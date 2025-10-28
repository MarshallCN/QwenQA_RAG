# gguf.py — Qwen GGUF chat with multi-session (load/save) via utils.py
import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import shutil
import gradio as gr
from llama_cpp import Llama

# Multi-session helpers from utils.py
from utils import mk_msg_dir, _as_dir, persist_messages, trim_by_tokens
# ===================== Model =====================
# You can swap to another GGUF by changing repo_id/filename.
model = Llama.from_pretrained(
    repo_id="bartowski/Qwen2.5-0.5B-Instruct-GGUF",
    filename="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
)

assistant_name = "Nova"
user_name = "Marshall"
persona = f"""Your name is {assistant_name}. Address the user as "{user_name}". Use Markdown; put code in fenced blocks with a language tag.""".strip()

# Where each conversation (session) persists its messages
BASE_MSG_DIR = Path("./msgs/msgs_QwenGGUF")
BASE_MSG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Qwen chat template (no tools) ----------
# def render_qwen(messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
#     """
#     Convert OpenAI-style messages to Qwen2.5 Instruct format:
#       <|im_start|>system ... <|im_end|>
#       <|im_start|>user ...   <|im_end|>
#       <|im_start|>assistant  (generation continues here)
#     """
#     # System prompt
#     if messages and messages[0].get("role") == "system":
#         sys_txt = messages[0]["content"]
#         rest = messages[1:]
#     else:
#         sys_txt = persona
#         rest = messages

#     parts = [f"<|im_start|>system\n{sys_txt}<|im_end|>\n"]
#     for m in rest:
#         role = m.get("role")
#         if role not in ("user", "assistant"):
#             continue
#         parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>\n")

#     if add_generation_prompt:
#         parts.append("<|im_start|>assistant\n")
#     return "".join(parts)

def render_qwen_trim(
    messages: List[Dict[str, str]],
    model,                             # llama_cpp.Llama 实例（用于 token 计数）
    n_ctx: Optional[int] = None,       # 不传则用 model.n_ctx()
    add_generation_prompt: bool = True,
    persona: str = "",
    reserve_new: int = 256,            # 希望生成的新 token 预算（上限）
    pad: int = 8,                      # 保险余量，避免越界
    hard_user_tail_chars: int = 2000,  # 还不够时，最后一条 user 文本的硬截断字符数
) -> Tuple[str, int]:
    """
    - 只保留 system + 最近的若干轮对话，使得 total_tokens + reserve_new + pad <= n_ctx
    - 若仍不够，则截短最后一条 user。
    - 返回 (prompt, safe_max_new)，safe_max_new 已确保不越界。
    """
    def _tok_len(txt: str) -> int:
        # 与 llama_cpp 的计数保持一致
        return len(model.tokenize(txt.encode("utf-8"), add_bos=True))

    if n_ctx is None:
        n_ctx = getattr(model, "n_ctx")() if callable(getattr(model, "n_ctx", None)) else model.n_ctx

    # 1) 拆出 system 与其余消息
    if messages and messages[0].get("role") == "system":
        sys_txt = messages[0]["content"]
        rest = messages[1:]
    else:
        sys_txt = persona
        rest = messages

    # 仅保留 user / assistant
    rest = [m for m in rest if m.get("role") in ("user", "assistant")]

    # 2) 生成函数：把 system + 若干轮对话渲染为 Qwen prompt
    def _render(sys_text: str, turns: List[Dict[str, str]], add_gen: bool) -> str:
        parts = [f"<|im_start|>system\n{sys_text}<|im_end|>\n"]
        for m in turns:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_gen:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    # 3) 先尝试保留全部轮次，从最老开始裁剪直到 fits
    kept = rest[:]  # 深拷贝
    while True:
        prompt = _render(sys_txt, kept, add_generation_prompt)
        used = _tok_len(prompt)

        # 计算还能安全生成的 token 数
        safe_max_new = max(1, n_ctx - used - pad)
        # 希望生成 reserve_new，但不能超过 safe_max_new
        if used + reserve_new + pad <= n_ctx:
            # 有余量，按 reserve_new 返回可生成上限
            return prompt, min(reserve_new, safe_max_new)

        # 没有余量——需要裁剪历史。如果可裁剪的 turns < 1，则进入硬截断
        if len(kept) <= 1:
            break  # 只剩最后一条，跳出去做硬截断

        # 从最早的一条开始丢；为避免打断成对语义，可一次丢两条（user+assistant）
        # 但如果开头不是成对，就按 1 条丢弃。
        drop_count = 2 if len(kept) >= 2 else 1
        # 保证留下至少 1 条（最后一条 user）用于上下文
        while drop_count > 0 and len(kept) > 1:
            kept.pop(0)
            drop_count -= 1

    # 4) 仍然不够：硬截断“最后一条 user”文本尾部
    #    目标：尽量保留最近语义，同时立刻释放 token 空间
    if kept and kept[-1]["role"] == "user":
        kept[-1] = {
            "role": "user",
            "content": kept[-1]["content"][-hard_user_tail_chars:]
        }
    elif kept:
        # 最后一条不是 user，则尽量截短它（通常是 assistant）
        kept[-1] = {
            "role": kept[-1]["role"],
            "content": kept[-1]["content"][-hard_user_tail_chars:]
        }

    # 重新渲染并最终给出安全 max_new
    prompt = _render(sys_txt, kept, add_generation_prompt)
    used = _tok_len(prompt)
    safe_max_new = max(1, n_ctx - used - pad)

    # 如果仍然超（极端长的 system），进一步把 system 也截短
    if used + pad > n_ctx:
        trimmed_sys = sys_txt[-hard_user_tail_chars:]
        prompt = _render(trimmed_sys, kept, add_generation_prompt)
        used = _tok_len(prompt)
        safe_max_new = max(1, n_ctx - used - pad)

    # 不允许返回负或 0
    return prompt, max(1, safe_max_new)


STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]

# ---------- Helpers for system + display ----------
def ensure_system(messages: Optional[List[Dict[str, str]]], sys_prompt: str) -> List[Dict[str, str]]:
    """Guarantee a system message at index 0 and keep it in sync with the UI textbox."""
    sys_prompt = (sys_prompt or persona).strip()
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": sys_prompt}]
    messages = list(messages)
    messages[0] = {"role": "system", "content": sys_prompt}
    return messages


def visible_chat(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Hide system from chat display for gr.Chatbot(type='messages')."""
    return [m for m in (messages or []) if m.get("role") in ("user", "assistant")]


# ---------- Session I/O ----------
def _load_latest(msg_id: str) -> List[Dict[str, str]]:
    p = Path(_as_dir(BASE_MSG_DIR, msg_id), "trimmed.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _init_sessions():
    sessions = [p.name for p in BASE_MSG_DIR.iterdir() if p.is_dir()] if BASE_MSG_DIR.exists() else []
    if len(sessions) == 0:
        # No history
        return gr.update(choices=[], value=None), [], "", [], []
    sessions.sort(reverse=True)
    msg_id = sessions[0]
    messages = _load_latest(msg_id)
    chat_hist = visible_chat(messages)
    return gr.update(choices=sessions, value=msg_id), sessions, msg_id, messages, chat_hist


def load_session(session_list, sessions):
    msg_id = session_list
    messages = _load_latest(msg_id)
    chat_hist = visible_chat(messages)
    return msg_id, messages, chat_hist, gr.update(choices=sessions, value=msg_id)


def start_new_session(sessions):
    msg_id = mk_msg_dir(BASE_MSG_DIR)
    sessions = list(sessions or []) + [msg_id]
    return [], [], "", msg_id, gr.update(choices=sessions, value=msg_id), sessions


def _on_rm_error(func, path, exc_info):
    try:
        if os.name == "nt":                    # Windows
            os.chmod(path, stat.S_IWRITE)      # 去掉只读
        else:                                  # Linux / macOS
            mode = os.stat(path).st_mode
            os.chmod(
                path,
                mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR  # 给所有者加 r,w,x
            )
        func(path)                              # 重试原操作，如 os.remove 或 os.rmdir
    except Exception:
        pass
        
def delete_session(msg_id, sessions):
    """Delete the currently selected session directory and refresh the list."""
    # Remove directory for current session
    if msg_id:
        try:
            shutil.rmtree(_as_dir(BASE_MSG_DIR, msg_id), onerror=_on_rm_error)
        except Exception:
            shutil.rmtree(_as_dir(BASE_MSG_DIR, msg_id), ignore_errors=True)
    # Re-scan sessions on disk
    if BASE_MSG_DIR.exists():
        sess = [p.name for p in BASE_MSG_DIR.iterdir() if p.is_dir()]
    else:
        sess = []
    sess.sort(reverse=True)
    if sess:
        new_id = sess[0]
        msgs = _load_latest(new_id)
        chat_hist = visible_chat(msgs)
        return msgs, chat_hist, "", new_id, gr.update(choices=sess, value=new_id), sess
    else:
        return [], [], "", "", gr.update(choices=[], value=None), []
        
def export_messages_to_json(messages, msg_id):
    base = Path("/data/exports") if Path("/data").exists() else Path("./exports")
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"chat_{stamp}.json"
    path = base / fname
    path.write_text(json.dumps(messages or [], ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)

def on_click_download(messages, msg_id):
    path = export_messages_to_json(messages, msg_id)
    return gr.update(value=path, visible=True)


# ---------- Generation callback ----------
def on_send(user_text: str,
            messages: List[Dict[str, str]],
            msg_id: str,
            sessions: List[str],
            sys_prompt: str,
            temperature: float,
            top_p: float,
            max_new_tokens: int,
            repetition_penalty: float):
    user_text = (user_text or "").strip()
    if not user_text:
        return gr.update(), messages, visible_chat(messages), msg_id, gr.update(choices=sessions, value=(msg_id or None)), sessions

    # 1) ensure system
    messages = ensure_system(messages, sys_prompt)

    # 2) session bookkeeping
    new_session = (len(messages) <= 1)  # only system exists
    if new_session and not msg_id:
        msg_id = mk_msg_dir(BASE_MSG_DIR)
        sessions = list(sessions or []) + [msg_id]
    if msg_id and msg_id not in (sessions or []):
        sessions = list(sessions or []) + [msg_id]
    sessions_update = gr.update(choices=sessions, value=msg_id)

    # 3) append user, render, generate
    messages = messages + [{"role": "user", "content": user_text}]
    # prompt = render_qwen(messages, add_generation_prompt=True)
    prompt, max_new = render_qwen_trim(
        messages=messages,
        model=model,        # llama_cpp.Llama 实例
        n_ctx=None,         # 不传用 model.n_ctx()
        add_generation_prompt=True,
        persona=persona,    # 你之前的 persona 变量
        reserve_new=max_new_tokens,  # 你希望的生成长度
        pad=16
    )
    

    try:
        result = model.create_completion(
            prompt=prompt,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new),
            repeat_penalty=float(repetition_penalty),
            stop=STOP_TOKENS,
        )
        reply = result['choices'][0]['text'].strip()
    except Exception:
        _out = model(
            prompt,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_new),
            repeat_penalty=float(repetition_penalty),
            stop=STOP_TOKENS,
        )
        if isinstance(_out, dict):
            reply = _out.get('choices', [{}])[0].get('text', '').strip()
        else:
            reply = str(_out).strip()

    # 4) append assistant + persist
    messages = messages + [{"role": "assistant", "content": reply}]

    if msg_id:
        msg_dir = _as_dir(BASE_MSG_DIR, msg_id)
        persist_messages(messages, msg_dir, archive_last_turn=True)

    return "", messages, visible_chat(messages), msg_id, sessions_update, sessions


# ===================== UI =====================
with gr.Blocks(title="Qwen GGUF — multi-session") as demo:
    gr.Markdown("## 🧠 Qwen Chat")

    with gr.Row():
        with gr.Column(scale=3):
            sys_prompt = gr.Textbox(
                label="System prompt",
                value=persona,
                lines=6,
                show_label=True,
            )
            with gr.Accordion("Generation settings", open=False):
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
                max_new_tokens = gr.Slider(16, 512, value=256, step=16, label="max_new_tokens")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.07, step=0.01, label="repetition_penalty")

            session_list = gr.Radio(choices=[], value=None, label="Conversations", interactive=True)
            new_btn = gr.Button("New session", variant="secondary")
            del_btn = gr.Button("Delete session", variant="stop")
            dl_btn = gr.Button("Download JSON", variant="secondary")
            dl_file = gr.File(label="", interactive=False, visible=False)

        with gr.Column(scale=9):
            chat = gr.Chatbot(
                label="Chat",
                height=560,
                render_markdown=True,
                type="messages",
            )
            user_box = gr.Textbox(
                label="Your message",
                placeholder="Type and press Enter…",
                autofocus=True,
            )
            send = gr.Button("Send", variant="primary")

    # States
    messages = gr.State([])   # includes system
    msg_id = gr.State("")
    sessions = gr.State([])

    # Events
    user_box.submit(
        on_send,
        inputs=[user_box, messages, msg_id, sessions, sys_prompt, temperature, top_p, max_new_tokens, repetition_penalty],
        outputs=[user_box, messages, chat, msg_id, session_list, sessions],
    )
    send.click(
        on_send,
        inputs=[user_box, messages, msg_id, sessions, sys_prompt, temperature, top_p, max_new_tokens, repetition_penalty],
        outputs=[user_box, messages, chat, msg_id, session_list, sessions],
    )

    new_btn.click(
        start_new_session,
        inputs=[sessions],
        outputs=[messages, chat, user_box, msg_id, session_list, sessions],
    )

    del_btn.click(
        delete_session,
        inputs=[msg_id, sessions],
        outputs=[messages, chat, user_box, msg_id, session_list, sessions],
    )

    session_list.change(
        load_session,
        inputs=[session_list, sessions],
        outputs=[msg_id, messages, chat, session_list],
    )

    dl_btn.click(
        on_click_download,
        inputs=[messages, msg_id],
        outputs=[dl_file],
    )

    demo.load(_init_sessions, None, outputs=[session_list, sessions, msg_id, messages, chat])

if __name__ == "__main__":
    demo.queue().launch()
