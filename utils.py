# from __future__ import annotations
from pathlib import Path
import uuid
from datetime import datetime, timezone
import json, os
from typing import List, Dict, Tuple, Optional

# ============ 工具函数 ============
def mk_msg_dir(BASE_MSG_DIR) -> str:
    m_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    Path(BASE_MSG_DIR, m_id).mkdir(parents=True, exist_ok=True)
    return m_id  # 只返回 ID

def _as_dir(BASE_MSG_DIR, m_id: str) -> str:
    # 统一把传入值规整为 ./msgs/<ID>
    return Path(BASE_MSG_DIR, m_id)

def msg2hist(persona, msg):
    chat_history = []
    if msg != None:
        if len(msg)>0:
            chat_history = msg.copy()                 # 外层列表浅拷
            chat_history[0] = msg[0].copy()           # 这个字典单独拷
            chat_history[0]['content'] = chat_history[0]['content'][len(persona):]
    return chat_history
        
def render(tok, messages: List[Dict[str, str]]) -> str:
    """按 chat_template 渲染成最终提示词文本（不分词）。"""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
def _ensure_alternating(messages):
    if not messages:
        return
    if messages[0]["role"] != "user":
        raise ValueError("messages[0] 必须是 'user'（你的模板要求从 user 开始）")
    for i, m in enumerate(messages):
        expect_user = (i % 2 == 0)
        if (m["role"] == "user") != expect_user:
            raise ValueError(f"对话必须严格交替 user/assistant，在索引 {i} 处发现 {m['role']}")

def trim_by_tokens(tok, messages, prompt_budget):
    """
    只保留 messages[0]（persona 的 user）+ 一个“从奇数索引开始的后缀”，
    用二分法找到能放下的最长后缀。这样可保证交替不被破坏。
    """
    if not messages:
        return []

    # _ensure_alternating(messages)

    # 只有 persona 这一条时，直接返回
    if len(messages) == 1:
        return messages

    # 允许的后缀起点：奇数索引（index 1,3,5,... 都是 assistant），
    # 这样拼接到 index0(user) 后才能保持交替。
    cand_idx = [k for k in range(1, len(messages)) if k % 2 == 1]

    # 如果任何也放不下，就只留 persona
    best = [messages[0]]

    # 二分：起点越靠前 → 保留消息越多 → token 越大（单调）
    lo, hi = 0, len(cand_idx) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        k = cand_idx[mid]
        candidate = [messages[0]] + messages[k:]
        toks = len(tok(tok.apply_chat_template(candidate, tokenize=False),
                       add_special_tokens=False).input_ids)
        if toks <= prompt_budget:
            best = candidate     # 能放下：尝试保留更多（向左走）
            hi = mid - 1
        else:
            lo = mid + 1         # 放不下：丢更多旧消息（向右走）

    return best

# ============ 原子写 可能会和onedrive同步冲突============
# def atomic_write_json(path: Path, data) -> None:
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     with open(tmp, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#         f.flush()
#         os.fsync(f.fileno())
#     os.replace(tmp, path)  # 同目录原子替换

# 直接覆盖
def write_json_overwrite(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)    
        
# ============ 存储层 ============
class MsgStore:
    def __init__(self, base_dir: str | Path = "./msgs"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.archive = self.base / "archive.jsonl"  # 只追加
        self.trimmed = self.base / "trimmed.json"   # 当前上下文
        if not self.archive.exists():
            self.archive.write_text("", encoding="utf-8")
        if not self.trimmed.exists():
            self.trimmed.write_text("[]", encoding="utf-8")

    def load_trimmed(self) -> List[Dict[str, str]]:
        try:
            return json.loads(self.trimmed.read_text(encoding="utf-8"))
        except Exception:
            return []

    def save_trimmed(self, messages: List[Dict[str, str]]) -> None:
        write_json_overwrite(self.trimmed, messages)

    def append_archive(self, role: str, content: str, meta: dict | None = None) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "role": role, "content": content}
        if meta: rec["meta"] = meta
        with open(self.archive, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())

# ============ 显式保存（手动调用才落盘） ============
def persist_messages(
    messages: List[Dict[str, str]],
    store_dir: str | Path = "./msgs",
    archive_last_turn: bool = True,
) -> None:
    store = MsgStore(store_dir)
    # _ensure_alternating(messages)

    # 1) 覆写 trimmed.json（原子）
    store.save_trimmed(messages)

    # 2) 追加最近一轮到 archive.jsonl（可选）
    if not archive_last_turn:
        return

    # 从尾部向前找最近的一对 (user, assistant)
    pair = None
    for i in range(len(messages) - 2, -1, -1):
        if (
            messages[i]["role"] == "user"
            and i + 1 < len(messages)
            and messages[i + 1]["role"] == "assistant"
        ):
            pair = (messages[i]["content"], messages[i + 1]["content"])
            break

    if pair:
        u, a = pair
        store.append_archive("user", u)
        store.append_archive("assistant", a)
    # 若没有找到成对（比如你在生成前就调用了 persist），就只写 trimmed，不归档
