# ggufv2_with_RAG_CPU_Space.py — Qwen GGUF chat + on‑Space RAG (vector DB build + retrieval)
# Merged from your local RAG prototype (Chat_RAG_vecDB.py) and HF Space chat app (ggufv2.py).
# CPU‑only friendly: uses llama.cpp for generation; BGEM3 + FAISS for embeddings; optional CrossEncoder reranking.

from __future__ import annotations
import os
import json
import stat
import shutil
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import gradio as gr

# ==== LLM (generation) — llama.cpp GGUF (CPU) ====
from llama_cpp import Llama

# ==== RAG dependencies (CPU‑friendly) ====
# NOTE: These import torch under the hood; works fine on CPU Spaces.
import faiss  # make sure you install faiss-cpu
try:
    from FlagEmbedding import BGEM3FlagModel
except Exception as e:
    BGEM3FlagModel = None
    print("[WARN] FlagEmbedding not available. Install with: pip install FlagEmbedding")

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None
    print("[WARN] sentence-transformers CrossEncoder not available. Install with: pip install sentence-transformers")

# Optional langchain loaders for PDFs/TXT chunking
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==== Utilities imported from your utils.py (unchanged) ====
from utils import mk_msg_dir, _as_dir, persist_messages

# ===================== Paths & Constants =====================
assistant_name = "Nova"
persona = (
    f"Your name is {assistant_name}. Use Markdown; "
    f"put code in fenced blocks with a language tag."
).strip()

# Sessions (conversations)
BASE_MSG_DIR = Path("./msgs/msgs_QwenGGUF")
BASE_MSG_DIR.mkdir(parents=True, exist_ok=True)

# Vector store base dir (on Space persistent storage)
VEC_DIR_BASE = Path("./vectorstore/bgem3")
VEC_DIR_BASE.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_ID = "BAAI/bge-m3"
DEFAULT_TOPK = 8
DEFAULT_RERANK_TAKE = 3

STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]

# ===================== GGUF Model =====================
model = Llama.from_pretrained(
    repo_id="bartowski/Qwen2.5-0.5B-Instruct-GGUF",
    filename="Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
)

# ===================== RAG Globals =====================
BGEM3 = None  # will be lazy-loaded
RERANKER = None
INDEX = None
CORPUS: List[str] = []
CURRENT_DB_DIR: Optional[Path] = None

# ===================== Helpers: Qwen prompt with trimming =====================
def render_qwen_trim(
    messages: List[Dict[str, str]],
    model,                             # llama_cpp.Llama for token counting
    n_ctx: Optional[int] = None,
    add_generation_prompt: bool = True,
    persona: str = "",
    reserve_new: int = 256,
    pad: int = 16,
    hard_user_tail_chars: int = 2000,
) -> Tuple[str, int]:
    """
    Keep system + most recent turns so total <= n_ctx - pad; if still too long, hard-truncate last user.
    Returns (prompt, safe_max_new).
    """
    def _tok_len(txt: str) -> int:
        return len(model.tokenize(txt.encode("utf-8"), add_bos=True))

    if n_ctx is None:
        n_ctx = getattr(model, "n_ctx")() if callable(getattr(model, "n_ctx", None)) else model.n_ctx

    if messages and messages[0].get("role") == "system":
        sys_txt = messages[0]["content"]
        rest = messages[1:]
    else:
        sys_txt = persona
        rest = messages

    rest = [m for m in rest if m.get("role") in ("user", "assistant")]

    def _render(sys_text: str, turns: List[Dict[str, str]], add_gen: bool) -> str:
        parts = [f"<|im_start|>system\n{sys_text}<|im_end|>\n"]
        for m in turns:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        if add_gen:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    kept = rest[:]
    while True:
        prompt = _render(sys_txt, kept, add_generation_prompt)
        used = _tok_len(prompt)
        safe_max_new = max(1, n_ctx - used - pad)
        if used + reserve_new + pad <= n_ctx:
            return prompt, min(reserve_new, safe_max_new)
        if len(kept) <= 1:
            break
        drop_count = 2 if len(kept) >= 2 else 1
        while drop_count > 0 and len(kept) > 1:
            kept.pop(0)
            drop_count -= 1

    if kept:
        last = kept[-1]
        kept[-1] = {"role": last["role"], "content": last["content"][-hard_user_tail_chars:]}  # hard tail

    prompt = _render(sys_txt, kept, add_generation_prompt)
    used = _tok_len(prompt)
    safe_max_new = max(1, n_ctx - used - pad)

    if used + pad > n_ctx:
        trimmed_sys = sys_txt[-hard_user_tail_chars:]
        prompt = _render(trimmed_sys, kept, add_generation_prompt)
        used = _tok_len(prompt)
        safe_max_new = max(1, n_ctx - used - pad)

    return prompt, max(1, safe_max_new)

# ===================== Chat (UI) helpers =====================
def ensure_system(messages: Optional[List[Dict[str, str]]], sys_prompt: str) -> List[Dict[str, str]]:
    sys_prompt = (sys_prompt or persona).strip()
    if not messages or messages[0].get("role") != "system":
        return [{"role": "system", "content": sys_prompt}]
    messages = list(messages)
    messages[0] = {"role": "system", "content": sys_prompt}
    return messages


def visible_chat(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [m for m in (messages or []) if m.get("role") in ("user", "assistant")]


# ===================== RAG: model + DB ops =====================
def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def ensure_rag_models():
    global BGEM3, RERANKER
    if BGEM3 is None:
        if BGEM3FlagModel is None:
            raise RuntimeError("FlagEmbedding is not installed. Please add it to requirements.txt")
        # CPU mode
        BGEM3 = BGEM3FlagModel(EMBED_MODEL_ID, use_fp16=False, device='cpu')
    if RERANKER is None and CrossEncoder is not None:
        try:
            RERANKER = CrossEncoder("BAAI/bge-reranker-large")
        except Exception:
            # Fallback to base if -large is too heavy on CPU Space
            RERANKER = CrossEncoder("BAAI/bge-reranker-base")


def list_vector_dbs() -> List[str]:
    names = [p.name for p in VEC_DIR_BASE.iterdir() if p.is_dir()]
    return ["<New Vector DB>"] + sorted(names)


def _load_documents_from_folder(folder: str) -> List:
    docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix.lower() == ".txt":
            docs += TextLoader(str(path), encoding="utf-8").load()
        elif path.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(path)).load()
    return docs


def _split_docs(docs, chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(v, dtype=np.float32)
    faiss.normalize_L2(v)
    return v


def _load_db_to_memory(db_name: str) -> str:
    """Load selected DB (index + corpus) into memory globals for retrieval."""
    global INDEX, CORPUS, CURRENT_DB_DIR
    if db_name == "<New Vector DB>":
        INDEX, CORPUS, CURRENT_DB_DIR = None, [], None
        return "🆕 New vector DB will be created on upload."
    try:
        db_dir = VEC_DIR_BASE / db_name
        INDEX = faiss.read_index(str(db_dir / "index.faiss"))
        with open(db_dir / "corpus.pkl", "rb") as f:
            CORPUS = pickle.load(f)
        CURRENT_DB_DIR = db_dir
        # stats
        with open(db_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return f"📦 Loaded DB `{db_name}` — {meta.get('raw_docs', 0)} docs, {meta.get('total_chunks', 0)} chunks"
    except Exception as e:
        INDEX, CORPUS, CURRENT_DB_DIR = None, [], None
        return f"⚠️ Failed to load DB `{db_name}`: {e}"


def create_or_extend_index(files, selected_db: str):
    """Add uploaded PDFs/TXTs into a vector DB; de‑duplicate by content hash."""
    ensure_rag_models()
    global INDEX, CORPUS, CURRENT_DB_DIR

    # Stage a temp folder of the uploaded files so loaders can open them
    temp_dir = Path("./temp_docs")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, onerror=lambda f, p, ei: _on_rm_error(f, p, ei))
    temp_dir.mkdir(parents=True, exist_ok=True)

    for f in (files or []):
        src = Path(getattr(f, 'name', str(f)))
        shutil.copy(str(src), str(temp_dir / src.name))

    raw_docs = _load_documents_from_folder(str(temp_dir))
    chunks = _split_docs(raw_docs)
    new_corpus = [t.page_content for t in chunks]
    new_hashes = [_hash_text(t) for t in new_corpus]

    # Create new DB or load existing
    if selected_db == "<New Vector DB>":
        db_id = mk_msg_dir(VEC_DIR_BASE)
        CURRENT_DB_DIR = VEC_DIR_BASE / db_id
        CURRENT_DB_DIR.mkdir(parents=True, exist_ok=True)
        # infer dim
        dim = BGEM3.encode(["dim test"])['dense_vecs'].shape[1]
        INDEX = faiss.IndexFlatIP(dim)
        CORPUS = []
        existing_hashes = set()
    else:
        CURRENT_DB_DIR = VEC_DIR_BASE / selected_db
        INDEX = faiss.read_index(str(CURRENT_DB_DIR / "index.faiss"))
        with open(CURRENT_DB_DIR / "corpus.pkl", "rb") as f:
            CORPUS = pickle.load(f)
        meta = json.loads((CURRENT_DB_DIR / "meta.json").read_text(encoding="utf-8"))
        existing_hashes = set(meta.get("hashes", []))

    # De‑dup new chunks by hash
    filtered = [(c, h) for c, h in zip(new_corpus, new_hashes) if h not in existing_hashes]
    if not filtered:
        return "✅ No new (non-duplicate) chunks to add.", gr.update(choices=list_vector_dbs()), _show_db_stats(selected_db)

    add_corpus, add_hashes = zip(*filtered)
    dense = BGEM3.encode(list(add_corpus), batch_size=64)["dense_vecs"]
    # BGEM3 returns torch.Tensor; convert to numpy
    try:
        import torch
        if isinstance(dense, torch.Tensor):
            dense = dense.detach().cpu().numpy()
    except Exception:
        pass
    dense = _normalize(np.array(dense, dtype=np.float32))

    INDEX.add(dense)
    CORPUS.extend(add_corpus)
    all_hashes = list(existing_hashes) + list(add_hashes)

    faiss.write_index(INDEX, str(CURRENT_DB_DIR / "index.faiss"))
    with open(CURRENT_DB_DIR / "corpus.pkl", "wb") as f:
        pickle.dump(CORPUS, f)

    meta = {
        "model": EMBED_MODEL_ID,
        "dim": int(dense.shape[1]),
        "total_chunks": len(CORPUS),
        "raw_docs": len(raw_docs),
        "hashes": all_hashes,
    }
    (CURRENT_DB_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    db_name = CURRENT_DB_DIR.name
    return (
        f"✅ Added {len(add_corpus)} chunks to DB `{db_name}`.",
        gr.update(choices=list_vector_dbs(), value=db_name),
        _show_db_stats(db_name),
    )


def _show_db_stats(selected_db: str) -> str:
    if selected_db == "<New Vector DB>":
        return "🆕 New vector DB will be created on next upload."
    try:
        db_dir = VEC_DIR_BASE / selected_db
        meta = json.loads((db_dir / "meta.json").read_text(encoding="utf-8"))
        return f"📊 DB `{selected_db}`: {int(meta.get('raw_docs', 0))} docs, {int(meta.get('total_chunks', 0))} chunks"
    except Exception as e:
        return f"⚠️ Failed to load DB `{selected_db}`: {e}"


def retrieve_top_docs(query: str, top_k: int, rerank_take: int) -> Tuple[List[str], List[int], List[float]]:
    """Return top documents, indices, and (optionally reranked) scores."""
    ensure_rag_models()
    if INDEX is None or not CORPUS:
        return [], [], []
    qv = BGEM3.encode([query])["dense_vecs"]
    try:
        import torch
        if hasattr(qv, "detach"):
            qv = qv.detach().cpu().numpy()
    except Exception:
        pass
    qv = _normalize(np.array(qv, dtype=np.float32))

    D, I = INDEX.search(qv, int(top_k))
    cand = [CORPUS[i] for i in I[0]]

    # (Optional) rerank with CrossEncoder if available
    if RERANKER is not None and len(cand) > 1:
        pairs = [[query, c] for c in cand]
        scores = RERANKER.predict(pairs)
        order = np.argsort(scores)[::-1].tolist()
        cand = [cand[i] for i in order]
        I0 = [int(I[0][i]) for i in order]
        scores = [float(scores[i]) for i in order]
    else:
        scores = D[0].tolist()
        I0 = I[0].tolist()

    take = max(1, int(rerank_take))
    return cand[:take], I0[:take], scores[:take]


def build_user_with_context(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join(contexts)
    user_prompt = (
        "Answer the question based on the following context. If the answer cannot be found, say you don't know.\n\n"
        f"{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    return user_prompt

# ===================== Session / File ops =====================

def _on_rm_error(func, path, exc_info):
    try:
        if os.name == "nt":
            os.chmod(path, stat.S_IWRITE)
        else:
            mode = os.stat(path).st_mode
            os.chmod(path, mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        func(path)
    except Exception:
        pass


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


def delete_session(msg_id, sessions):
    if msg_id:
        try:
            shutil.rmtree(_as_dir(BASE_MSG_DIR, msg_id), onerror=_on_rm_error)
        except Exception:
            shutil.rmtree(_as_dir(BASE_MSG_DIR, msg_id), ignore_errors=True)
    sess = [p.name for p in BASE_MSG_DIR.iterdir() if p.is_dir()] if BASE_MSG_DIR.exists() else []
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

# ===================== Generation Callback (with optional RAG) =====================

def on_send(user_text: str,
            messages: List[Dict[str, str]],
            msg_id: str,
            sessions: List[str],
            sys_prompt: str,
            temperature: float,
            top_p: float,
            max_new_tokens: int,
            repetition_penalty: float,
            use_rag: bool,
            top_k: int,
            rerank_take: int,
            ):
    user_text = (user_text or "").strip()
    if not user_text:
        return gr.update(), messages, visible_chat(messages), msg_id, gr.update(choices=sessions, value=(msg_id or None)), sessions, gr.update(value="", visible=True)

    # 1) ensure system
    messages = ensure_system(messages, sys_prompt)

    # 2) session bookkeeping
    new_session = (len(messages) <= 1)
    if new_session and not msg_id:
        msg_id = mk_msg_dir(BASE_MSG_DIR)
        sessions = list(sessions or []) + [msg_id]
    if msg_id and msg_id not in (sessions or []):
        sessions = list(sessions or []) + [msg_id]
    sessions_update = gr.update(choices=sessions, value=msg_id)

    # 3) RAG (optional): augment the user's message with retrieved context
    rag_context_text = ""
    if use_rag:
        try:
            contexts, idxs, scores = retrieve_top_docs(user_text, int(top_k), int(rerank_take))
        except Exception as e:
            contexts, idxs, scores = [], [], []
            rag_context_text = f"⚠️ RAG retrieval failed: {e}"
        if contexts:
            rag_context_text = "\n\n".join([f"[{i+1}] {c.strip()[:1200]}" for i, c in enumerate(contexts)])
            user_text_aug = build_user_with_context(user_text, contexts)
        else:
            user_text_aug = user_text
    else:
        user_text_aug = user_text

    # 4) append user (raw) for UI/persist; use augmented only for model prompt
    visible_messages = messages + [{"role": "user", "content": user_text}]
    prompt_messages = messages + [{"role": "user", "content": user_text_aug}]

    prompt, max_new = render_qwen_trim(
        messages=prompt_messages,
        model=model,
        n_ctx=None,
        add_generation_prompt=True,
        persona=persona,
        reserve_new=max_new_tokens,
        pad=16,
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

    # 5) append assistant + persist
    messages = visible_messages + [{"role": "assistant", "content": reply}]

    if msg_id:
        msg_dir = _as_dir(BASE_MSG_DIR, msg_id)
        persist_messages(messages, msg_dir, archive_last_turn=True)

    return "", messages, visible_chat(messages), msg_id, sessions_update, sessions, gr.update(value=rag_context_text, visible=use_rag)


# ===================== UI =====================
with gr.Blocks(title="Qwen Chat with RAG (CPU Space)") as demo:
    gr.Markdown("## 🧠 Qwen Chat with RAG (BGEM3 + FAISS)")

    with gr.Row():
        with gr.Column(scale=3):
            sys_prompt = gr.Textbox(label="System prompt", value=persona, lines=6)
            with gr.Accordion("Generation settings", open=False):
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.01, label="top_p")
                max_new_tokens = gr.Slider(16, 512, value=256, step=16, label="max_new_tokens")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.07, step=0.01, label="repetition_penalty")
            with gr.Accordion("RAG settings", open=True):
                use_rag = gr.Checkbox(value=False, label="Use RAG for replies")
                db_selector = gr.Dropdown(label="Vector DB", choices=list_vector_dbs(), value="<New Vector DB>")
                top_k = gr.Slider(1, 20, value=DEFAULT_TOPK, step=1, label="Retrieve top‑k")
                rerank_take = gr.Slider(1, 10, value=DEFAULT_RERANK_TAKE, step=1, label="Rerank keep (Top‑N)")
                rag_status = gr.Textbox(label="RAG status / DB stats", interactive=False)
                with gr.Row():
                    file_box = gr.File(label="Upload documents (PDF/TXT)", file_types=[".pdf", ".txt"], file_count="multiple")
                add_btn = gr.Button("📚 Add to Vector DB")

            session_list = gr.Radio(choices=[], value=None, label="Conversations", interactive=True)
            new_btn = gr.Button("New chat", variant="secondary")
            del_btn = gr.Button("Delete chat", variant="stop")
            dl_btn = gr.Button("Download JSON", variant="secondary")
            dl_file = gr.File(label="", interactive=False, visible=False)

        with gr.Column(scale=9):
            chat = gr.Chatbot(label="Chat", height=560, render_markdown=True, type="messages")
            rag_ctx = gr.Textbox(label="📄 RAG context (Top‑N)", lines=8, interactive=False, show_copy_button=True, visible=False)
            user_box = gr.Textbox(label="Your message", placeholder="Type and press Enter…", autofocus=True)
            send = gr.Button("Send", variant="primary")

    # States
    messages = gr.State([])
    msg_id = gr.State("")
    sessions = gr.State([])
    # Toggle visibility of RAG UI when checkbox changes
    def toggle_rag_visibility(use):
        vis = bool(use)
        return (
            gr.update(visible=vis),  # db_selector
            gr.update(visible=vis),  # top_k
            gr.update(visible=vis),  # rerank_take
            gr.update(visible=vis),  # rag_status
            gr.update(visible=vis),  # file_box
            gr.update(visible=vis),  # add_btn
            gr.update(visible=vis),  # rag_ctx
        )

    use_rag.change(
        toggle_rag_visibility,
        inputs=[use_rag],
        outputs=[db_selector, top_k, rerank_take, rag_status, file_box, add_btn, rag_ctx],
    )

    # Events — RAG DB
    def on_db_change(db_name):
        return _load_db_to_memory(db_name), _show_db_stats(db_name)

    db_selector.change(on_db_change, inputs=[db_selector], outputs=[rag_status, rag_status])
    add_btn.click(create_or_extend_index, inputs=[file_box, db_selector], outputs=[rag_status, db_selector, rag_status])

    # Events — Chat
    user_box.submit(
        on_send,
        inputs=[user_box, messages, msg_id, sessions, sys_prompt, temperature, top_p, max_new_tokens, repetition_penalty, use_rag, top_k, rerank_take],
        outputs=[user_box, messages, chat, msg_id, session_list, sessions, rag_ctx],
    )
    send.click(
        on_send,
        inputs=[user_box, messages, msg_id, sessions, sys_prompt, temperature, top_p, max_new_tokens, repetition_penalty, use_rag, top_k, rerank_take],
        outputs=[user_box, messages, chat, msg_id, session_list, sessions, rag_ctx],
    )

    new_btn.click(start_new_session, inputs=[sessions], outputs=[messages, chat, user_box, msg_id, session_list, sessions])
    del_btn.click(delete_session, inputs=[msg_id, sessions], outputs=[messages, chat, user_box, msg_id, session_list, sessions])
    session_list.change(load_session, inputs=[session_list, sessions], outputs=[msg_id, messages, chat, session_list])

    dl_btn.click(on_click_download, inputs=[messages, msg_id], outputs=[dl_file])

    def _on_load():
        # refresh DB dropdown + sessions
        db_ui = gr.update(choices=list_vector_dbs(), value="<New Vector DB>")
        sess_ui, sess, cur_id, msgs, chat_hist = _init_sessions()
        return db_ui, "🆕 New vector DB will be created on next upload.", sess_ui, sess, cur_id, msgs, chat_hist

    demo.load(_on_load, None, outputs=[db_selector, rag_status, session_list, sessions, msg_id, messages, chat])

if __name__ == "__main__":
    demo.launch()
