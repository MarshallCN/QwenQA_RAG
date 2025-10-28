import gradio as gr
import stat
import os, shutil, pickle, torch, json, hashlib
import faiss, numpy as np
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from utils import mk_msg_dir

# === 模型加载 ===
if gr.NO_RELOAD:
    BASE_DIR = r"C:\Users\c1052689\hug_models\Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(BASE_DIR, use_fast=False, local_files_only=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(BASE_DIR, quantization_config=bnb, device_map="auto", local_files_only=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=512)
    BGEM3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    reranker = CrossEncoder("BAAI/bge-reranker-large")

# === 向量库全局变量 ===
corpus = []
index = None
current_db_dir = None

vec_dir_base = './vectorstore/bgem3/'
embedding_model_id = 'BAAI/bge-m3'

# === 文档加载 & 向量构建 ===
def load_documents(folder: str):
    docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix == ".txt":
            docs += TextLoader(str(path), encoding="utf-8").load()
        elif path.suffix == ".pdf":
            docs += PyPDFLoader(str(path)).load()
    return docs

def split_docs(docs, chunk_size=512, chunk_overlap=64):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def on_rm_error(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def list_vector_dbs():
    db_list = [f.name for f in Path(vec_dir_base).iterdir() if f.is_dir()]
    return ["<New Vector DB>"] + db_list

def create_or_extend_index(docs, selected_db):
    global corpus, index, current_db_dir

    temp_dir = "temp_docs"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, onerror=on_rm_error)
    os.makedirs(temp_dir, exist_ok=True)

    for file in docs:
        src_path = file.name if hasattr(file, "name") else str(file)
        dst_path = os.path.join(temp_dir, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

    raw_docs = load_documents(temp_dir)
    chunks = split_docs(raw_docs)
    new_corpus = [t.page_content for t in chunks]
    new_hashes = [hash_text(t) for t in new_corpus]

    if selected_db == "<New Vector DB>":
        db_id = mk_msg_dir(Path(vec_dir_base))
        current_db_dir = os.path.join(vec_dir_base, db_id)
        os.makedirs(current_db_dir, exist_ok=True)
        index = faiss.IndexFlatIP(BGEM3.encode(["test"])["dense_vecs"].shape[1])
        corpus = []
        existing_hashes = set()
    else:
        current_db_dir = os.path.join(vec_dir_base, selected_db)
        index = faiss.read_index(os.path.join(current_db_dir, "index.faiss"))
        with open(os.path.join(current_db_dir, "corpus.pkl"), "rb") as f:
            corpus = pickle.load(f)
        with open(os.path.join(current_db_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
            existing_hashes = set(meta.get("hashes", []))

    # 去重
    filtered = [(c, h) for c, h in zip(new_corpus, new_hashes) if h not in existing_hashes]
    if not filtered:
        return "✅ No new (non-duplicate) chunks to add."

    add_corpus, add_hashes = zip(*filtered)
    dense = BGEM3.encode(add_corpus, batch_size=64)["dense_vecs"]
    if isinstance(dense, torch.Tensor):
        dense = dense.detach().cpu().numpy()
    dense = np.ascontiguousarray(dense, dtype=np.float32)
    faiss.normalize_L2(dense)

    index.add(dense)
    corpus.extend(add_corpus)
    all_hashes = list(existing_hashes) + list(add_hashes)

    faiss.write_index(index, os.path.join(current_db_dir, "index.faiss"))
    with open(os.path.join(current_db_dir, "corpus.pkl"), "wb") as f:
        pickle.dump(corpus, f)
    meta = {
        "model": embedding_model_id,
        "dim": int(dense.shape[1]),
        "total_chunks": len(corpus),
        "raw_docs": len(raw_docs),
        "hashes": all_hashes,
    }
    with open(os.path.join(current_db_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    db_stats = f"✅ Added {len(add_corpus)} new chunks to DB `{os.path.basename(current_db_dir)}`."
    db_list_update = gr.update(choices=list_vector_dbs())
    return db_stats, db_list_update

def build_prompt_corpus(top_docs, question):
    context_text = "\n\n".join(top_docs)
    user_prompt = f"""Answer the question based on the following context:

    {context_text}

    Question: {question}
    Answer:"""

    full_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return full_prompt

def ask_question(query):
    if not query.strip():
        return "❌ Please enter your questions", [], ""
    if index is None or len(corpus) == 0:
        return "⚠️ Please upload your documents", [], ""

    qv = np.array(BGEM3.encode([query])["dense_vecs"], dtype="float32")
    faiss.normalize_L2(qv)
    D, I = index.search(qv, 8)
    results = [corpus[i] for i in I[0]]
    pairs = [[query, c] for c in results]
    scores = reranker.predict(pairs)
    top_docs = [c for _, c in sorted(zip(scores, results), reverse=True)][:3]

    prompt = build_prompt_corpus(top_docs, query)
    out = pipe(
        prompt,
        max_new_tokens=1024,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        return_full_text=False,
    )
    reply = out[0]["generated_text"]
    context_display = "\n\n".join(
        f"[{i+1}] {doc.strip()[:1000]}" for i, doc in enumerate(top_docs)
    )
    return reply.strip(), context_display
    
def show_db_stats(selected_db):
    if selected_db == "<New Vector DB>":
        return "🆕 New vector DB will be created on next upload."
    try:
        db_dir = os.path.join(vec_dir_base, selected_db)
        with open(os.path.join(db_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
            chunk_num = int(meta.get("total_chunks", 0))
            docs_num = int(meta.get("raw_docs", 0))
        return f"📊 DB `{selected_db}`: {docs_num} docs, {chunk_num} chunks"
    except Exception as e:
        return f"⚠️ Failed to load DB `{selected_db}`: {e}"
    
with gr.Blocks(title="Qwen2.5 RAG Chat") as demo:
    gr.Markdown("## 🧠 Qwen2.5 BGEM3-RAG QA")

    with gr.Row():
        with gr.Column():
            file_box = gr.File(label="Upload documents (PDF or TXT)", file_types=[".pdf", ".txt"], file_count="multiple")
            db_selector = gr.Dropdown(label="Select or create vector DB", choices=list_vector_dbs(), value="<New Vector DB>")
            upload_btn = gr.Button("📚 Add to Vector DB")
            status = gr.Textbox(label="Status")

        with gr.Column():
            query = gr.Textbox(label="Enter your questions")
            ask_btn = gr.Button("Send")
            answer = gr.Textbox(label="🧠 Answer", lines=5)
            context = gr.Textbox(
                label="📄 Top-3 Reference Contexts",
                lines=10,
                interactive=False,
                show_copy_button=True,
                max_lines=20
            )
    db_selector.change(fn=show_db_stats, inputs=db_selector, outputs=status)
    upload_btn.click(fn=create_or_extend_index, inputs=[file_box, db_selector], outputs=[status, db_selector])
    ask_btn.click(fn=ask_question, inputs=query, outputs=[answer, context])
    demo.load(
        fn=lambda: gr.update(choices=list_vector_dbs()),
        inputs=None,
        outputs=db_selector
    )

if __name__ == '__main__':
    demo.launch(debug=True)
