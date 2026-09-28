"""RAG Studio: YouTube summariser + mind map + quiz + chat.
Lightweight build for Render's free tier (512 MB RAM): no torch, LLM via Hugging Face Inference API,
TF-IDF retrieval, per-visitor session state."""
import os, re, json, time, textwrap, tempfile
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
DEFAULT_MODEL = MODELS[0]
SERVER_TOKEN = os.environ.get("HF_TOKEN")                      # optional shared token (Render env var)
MAX_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "200000"))   # protects the tiny free CPU
LANGS = ["en", "en-US", "en-GB", "hi"]


# ───────────────────────── LLM ─────────────────────────
def llm(messages, model, token=None, temperature=0.3, max_tokens=1024, retries=4):
    from huggingface_hub import InferenceClient
    tok = token or SERVER_TOKEN
    if not tok:
        raise RuntimeError("No Hugging Face token. Paste yours in the 'HF token' box "
                           "(free at huggingface.co/settings/tokens).")
    client = InferenceClient(api_key=tok)
    for attempt in range(retries):
        try:
            r = client.chat_completion(messages=messages, model=model,
                                       temperature=max(temperature, 0.01), max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if "402" in msg or "credit" in msg:
                raise RuntimeError("Free Hugging Face credits are used up for this token. "
                                   "Try again after the monthly reset or use another token.")
            if "401" in msg or "unauthorized" in msg:
                raise RuntimeError("Hugging Face rejected the token. Check it and try again.")
            if any(x in msg for x in ("429", "503", "rate", "loading", "overloaded")) and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise


def parse_json(s):
    s = re.sub(r"^```(?:json)?|```$", "", s.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def fmt_time(t):
    if t is None:
        return ""
    t = int(t); h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ───────────────────────── Ingestion ─────────────────────────
def video_id_from(url):
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    raise ValueError("Couldn't find a video ID in that URL.")


def fetch_transcript(url):
    from youtube_transcript_api import YouTubeTranscriptApi
    vid = video_id_from(url)
    api = YouTubeTranscriptApi()
    proxy = os.environ.get("YT_PROXY_URL")     # optional: cloud IPs are often blocked by YouTube
    if proxy:
        from youtube_transcript_api.proxies import GenericProxyConfig
        api = YouTubeTranscriptApi(proxy_config=GenericProxyConfig(http_url=proxy, https_url=proxy))
    try:
        fetched = api.fetch(vid, languages=LANGS)
    except Exception as e:
        if "NoTranscriptFound" not in type(e).__name__:
            raise
        fetched = next(iter(api.list(vid))).fetch()
    return [(s.start, s.text) for s in fetched]


def text_to_segments(text):
    sents = re.split(r"(?<=[.!?।])\s+|\n+", text)
    return [(None, s.strip()) for s in sents if s.strip()]


def read_file(f):
    path = f if isinstance(f, str) else f.name
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    return open(path, encoding="utf-8", errors="ignore").read()


def chunk_segments(segs, max_chars=1200):
    chunks, cur, start, n = [], [], None, 0
    for t, txt in segs:
        txt = txt.replace("\n", " ")
        if start is None:
            start = t
        cur.append(txt); n += len(txt) + 1
        if n >= max_chars:
            chunks.append({"start": start, "text": " ".join(cur)})
            cur, start, n = [], None, 0
    if cur:
        chunks.append({"start": start, "text": " ".join(cur)})
    return chunks


def chunk_line(c):
    ts = f"[{fmt_time(c['start'])}] " if c["start"] is not None else ""
    return ts + c["text"]


def build_index(chunks):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, max_features=50000)
    return vec, vec.fit_transform([c["text"] for c in chunks])


def retrieve(st, question, k=5):
    q = st["vec"].transform([question])
    scores = (st["mat"] @ q.T).toarray().ravel()
    idx = [i for i in np.argsort(-scores)[:k] if scores[i] > 0] or list(range(min(k, len(st["chunks"]))))
    return [st["chunks"][i] for i in sorted(idx)]


def load_source(st, url, text, file):
    """Fills the per-visitor state dict `st`; returns a status message."""
    try:
        if url and url.strip():
            segs, title = fetch_transcript(url), "YouTube " + video_id_from(url)
        elif text and text.strip():
            segs, title = text_to_segments(text), "Pasted text"
        elif file:
            segs, title = text_to_segments(read_file(file)), "Uploaded file"
        else:
            return "⚠️ Provide a YouTube link, paste text, or upload a file."
    except Exception as e:
        return (f"⚠️ Could not get the text: `{type(e).__name__}`. YouTube often blocks cloud servers. "
                "Workaround: open the video → *…more* → *Show transcript*, copy it, and paste it into the text box.")
    chunks, total = [], 0
    for c in chunk_segments(segs):
        total += len(c["text"])
        if total > MAX_CHARS:
            break
        chunks.append(c)
    if not chunks:
        return "⚠️ No text found."
    vec, mat = build_index(chunks)
    st.update(chunks=chunks, vec=vec, mat=mat, title=title, notes=None, quiz=None)
    words = sum(len(c["text"].split()) for c in chunks)
    note = " (trimmed to the size limit)" if total > MAX_CHARS else ""
    return f"✅ Loaded **{title}**: {words:,} words in {len(chunks)} chunks{note}. Use the tabs below."


# ───────────────────────── Features ─────────────────────────
def split_blocks(text, size=6000):
    blocks, cur, n = [], [], 0
    for line in text.split("\n"):
        cur.append(line); n += len(line) + 1
        if n >= size:
            blocks.append("\n".join(cur)); cur, n = [], 0
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def condense(text, model, token, words=180):
    return llm([
        {"role": "system", "content": "You write dense, faithful study notes."},
        {"role": "user", "content": f"Condense this transcript excerpt into at most {words} words of bullet notes. "
                                    f"Keep concrete facts, names and numbers, and keep the [mm:ss] timestamps of key points.\n\n{text}"},
    ], model, token, max_tokens=500)


def get_notes(st, model, limit=10000):
    if st.get("notes"):
        return st["notes"]
    notes = "\n".join(chunk_line(c) for c in st["chunks"])
    for _ in range(3):
        if len(notes) <= limit:
            break
        notes = "\n".join(condense(b, model, st.get("token")) for b in split_blocks(notes))
    st["notes"] = notes
    return notes


def summarise(st, model):
    if not st.get("chunks"):
        return "⚠️ Load a video / text first."
    try:
        notes = get_notes(st, model)
        return llm([
            {"role": "system", "content": "You are an expert summariser. Use only the provided material."},
            {"role": "user", "content": "Write a summary in Markdown with these sections:\n"
                "## TL;DR (2-3 sentences)\n## Key points (6-10 bullets)\n"
                "## Chapters (timestamp - title; only if timestamps exist)\n## Key takeaways (3-5 bullets)\n\n"
                f"Material:\n{notes}"},
        ], model, st.get("token"), max_tokens=1500)
    except Exception as e:
        return f"⚠️ {e}"


COLORS = ["#4f46e5", "#0891b2", "#059669", "#d97706", "#dc2626"]


def tree_to_outline(node, depth=0):
    out = "  " * depth + ("# " if depth == 0 else "- ") + str(node.get("title", "")) + "\n"
    for ch in node.get("children", []) or []:
        out += tree_to_outline(ch, depth + 1)
    return out


def render_mindmap(tree):
    import graphviz
    g = graphviz.Digraph(format="png")
    g.attr(rankdir="LR", bgcolor="white", nodesep="0.25", ranksep="0.6", dpi="160")
    counter = [0]

    def add(node, parent, depth):
        title = str(node.get("title", ""))
        nid = f"n{counter[0]}"; counter[0] += 1
        color = COLORS[min(depth, len(COLORS) - 1)]
        label = "\n".join(textwrap.wrap(title, 26)) or title
        g.node(nid, label, shape="box", style="rounded,filled",
               fillcolor=color if depth < 2 else "white", fontcolor="white" if depth < 2 else "black",
               color=color, fontname="Helvetica", fontsize="13" if depth else "16")
        if parent:
            g.edge(parent, nid, color=color)
        for ch in node.get("children", []) or []:
            add(ch, nid, depth + 1)

    add(tree, None, 0)
    return g.render(os.path.join(tempfile.mkdtemp(), "mindmap"), cleanup=True)   # unique file per request


def make_mindmap(st, model, depth):
    if not st.get("chunks"):
        return None, "⚠️ Load a video / text first."
    try:
        notes = get_notes(st, model)
        raw = llm([
            {"role": "system", "content": "You turn material into hierarchical mind maps. Reply with JSON only."},
            {"role": "user", "content": 'Return ONLY JSON of the form {"title": str, "children": [{"title": str, "children": [...]}]}. '
                f"Max depth {int(depth)} levels below the root, 4-7 main branches, every label <= 6 words.\n\nMaterial:\n{notes}"},
        ], model, st.get("token"), temperature=0.2, max_tokens=1500)
        tree = parse_json(raw)
        outline = tree_to_outline(tree)
        try:
            img = render_mindmap(tree)
        except Exception as e:
            img, outline = None, f"(Graph render failed: {e})\n\n" + outline
        return img, "```\n" + outline + "```"
    except Exception as e:
        return None, f"⚠️ {e}"


QTYPES = {
    "Multiple choice": "multiple-choice with exactly 4 options (A-D) and one correct answer",
    "True / False": "true-or-false statements",
    "Short answer": "short-answer questions with a concise model answer",
}


def show_quiz(st, show_answers):
    qs = st.get("quiz")
    if not qs:
        return "Generate a quiz first."
    out = []
    for i, q in enumerate(qs, 1):
        out.append(f"**Q{i}. {q['question']}**\n")
        for j, o in enumerate(q.get("options") or []):
            o = re.sub(r"^[A-Da-d][).:]\s*", "", str(o))
            out.append(f"- {'ABCD'[j] if j < 4 else j + 1}) {o}")
        if show_answers:
            out.append(f"\n✅ **Answer:** {q['answer']}  \n💡 {q.get('explanation', '')}")
        out.append("\n---\n")
    return "\n".join(out)


def make_quiz(st, model, n, qtype, difficulty):
    if not st.get("chunks"):
        return "⚠️ Load a video / text first."
    try:
        notes = get_notes(st, model)
        raw = llm([
            {"role": "system", "content": "You write accurate quiz questions strictly from the given material. Reply with JSON only."},
            {"role": "user", "content": f"Write {int(n)} {difficulty.lower()} {QTYPES[qtype]} questions. Return ONLY JSON: "
                '{"questions": [{"question": str, "options": [str, ...] (empty list if not multiple choice), '
                '"answer": str, "explanation": str}]}\n\n' f"Material:\n{notes}"},
        ], model, st.get("token"), temperature=0.4, max_tokens=2500)
        st["quiz"] = parse_json(raw)["questions"]
        return show_quiz(st, False)
    except Exception as e:
        return f"⚠️ {e}"


def _clean(history, n=6):
    out = []
    for m in (history or [])[-n:]:
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            out.append({"role": m["role"], "content": m["content"]})
    return out


def chat(message, history, st, model):
    history = history or []
    if not message or not message.strip():
        return history, "", ""

    def reply(text, src=""):
        return history + [{"role": "user", "content": message},
                          {"role": "assistant", "content": text}], src, ""

    if not st.get("chunks"):
        return reply("⚠️ Load a video / text first.")
    try:
        recent = _clean(history)
        query = message
        if recent:
            convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in recent)
            query = llm([{"role": "user", "content": "Rewrite the LAST user question as one standalone question, "
                "using the conversation for context. Output only the question.\n\n"
                f"{convo}\nuser: {message}"}], model, st.get("token"), temperature=0, max_tokens=80).strip() or message
        hits = retrieve(st, query, k=5)
        context = "\n\n".join(chunk_line(c) for c in hits)
        msgs = [{"role": "system", "content": "Answer ONLY from the provided context and the earlier conversation. "
                 "If the answer is not in the context, say so. Cite timestamps like [mm:ss] when the context has them."}]
        msgs += recent
        msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {message}"})
        ans = llm(msgs, model, st.get("token"), max_tokens=700)
        src = "\n\n".join(f"**{('[' + fmt_time(c['start']) + ']') if c['start'] is not None else '•'}** {c['text'][:220]}…" for c in hits)
        return reply(ans, src)
    except Exception as e:
        return reply(f"⚠️ {e}")


# ───────────────────────── UI ─────────────────────────
def build_ui():
    import gradio as gr

    def load_handler(url, text, file, token):
        st = {"token": (token or "").strip() or None}      # brand-new state per Load click
        msg = load_source(st, url, text, file)
        return msg, st, [], ""

    def quiz_handler(st, model, n, qtype, diff):
        return make_quiz(st, model, n, qtype, diff), st, False

    with gr.Blocks(title="RAG Studio", theme=gr.themes.Soft()) as demo:
        st = gr.State({})          # one private copy per visitor
        gr.Markdown("# 🎬 RAG Studio\nSummarise a YouTube video, build a mind map, generate a quiz, or chat with it, "
                    "all grounded in the transcript.")
        with gr.Row():
            with gr.Column(scale=3):
                url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
                text = gr.Textbox(label="...or paste text / transcript", lines=4)
                file = gr.File(label="...or upload .txt / .pdf", file_types=[".txt", ".pdf", ".md"], type="filepath")
            with gr.Column(scale=1):
                model = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Model", allow_custom_value=True)
                token = gr.Textbox(label="HF token (optional)", type="password",
                                   placeholder="hf_... (uses your own free credits)" if SERVER_TOKEN else "hf_... (required)")
                load_btn = gr.Button("📥 Load", variant="primary")
        status = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("📝 Summary"):
                sum_btn = gr.Button("Summarise", variant="primary")
                sum_out = gr.Markdown()
                sum_btn.click(summarise, [st, model], sum_out)
            with gr.Tab("🧠 Mind map"):
                depth = gr.Slider(2, 4, value=3, step=1, label="Depth")
                mm_btn = gr.Button("Generate mind map", variant="primary")
                mm_img = gr.Image(label="Mind map", type="filepath")
                mm_outline = gr.Markdown()
                mm_btn.click(make_mindmap, [st, model, depth], [mm_img, mm_outline])
            with gr.Tab("❓ Quiz"):
                with gr.Row():
                    n = gr.Slider(3, 15, value=5, step=1, label="Questions")
                    qtype = gr.Dropdown(list(QTYPES), value="Multiple choice", label="Type")
                    diff = gr.Dropdown(["Easy", "Medium", "Hard"], value="Medium", label="Difficulty")
                q_btn = gr.Button("Generate quiz", variant="primary")
                reveal = gr.Checkbox(label="Show answers")
                q_out = gr.Markdown()
                q_btn.click(quiz_handler, [st, model, n, qtype, diff], [q_out, st, reveal])
                reveal.change(show_quiz, [st, reveal], q_out)
            with gr.Tab("💬 Chat (RAG)"):
                try:
                    chatbot = gr.Chatbot(type="messages", height=420, label="Chat about the loaded video")
                except TypeError:
                    chatbot = gr.Chatbot(height=420, label="Chat about the loaded video")
                question = gr.Textbox(label="Your question", placeholder="Ask anything, follow-ups are remembered ...")
                with gr.Row():
                    ask_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear chat")
                with gr.Accordion("Retrieved passages (latest answer)", open=False):
                    srcs = gr.Markdown()
                ask_btn.click(chat, [question, chatbot, st, model], [chatbot, srcs, question])
                question.submit(chat, [question, chatbot, st, model], [chatbot, srcs, question])
                clear_btn.click(lambda: ([], "", ""), None, [chatbot, srcs, question])

        load_btn.click(load_handler, [url, text, file, token], [status, st, chatbot, srcs])
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue(max_size=20, default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
