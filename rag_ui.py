import streamlit as st
from pathlib import Path
import chromadb
from chromadb.config import Settings
import ollama
import textwrap

LANG_PROMPTS = {
    "php": (
        "Du bist ein hilfsbereiter Senior-PHP-Entwickler. "
        "Antworte ausschließlich auf Basis der gelieferten Snippets. "
        "Zitiere IMMER Dateipfad und Zeilennummern. "
        "Wenn Information fehlt, sage es klar. "
        "Schlage konkrete Änderungen vor."
    ),
    "cpp": (
        "Du bist ein hilfsbereiter Senior-C++-Entwickler. "
        "Antworte ausschließlich auf Basis der gelieferten Snippets. "
        "Zitiere IMMER Dateipfad und Zeilennummern. "
        "Wenn Information fehlt, sage es klar. "
        "Schlage konkrete Änderungen vor und achte auf Header/Source-Kopplung."
    ),
    "generic": (
        "Du bist ein hilfsbereiter Senior-Softwareentwickler. "
        "Antworte ausschließlich auf Basis der gelieferten Snippets. "
        "Zitiere IMMER Dateipfad und Zeilennummern. "
        "Wenn Information fehlt, sage es klar. "
        "Schlage präzise Code-Änderungen vor."
    ),
}

def format_context(results, max_chars=12000, show_dist=False):
    if not results or not results.get("documents") or not results["documents"][0]:
        return ""
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results.get("distances", [[]])[0] if show_dist else [None] * len(docs)

    parts, total = [], 0
    for doc, meta, dist in zip(docs, metas, dists):
        path = meta.get("path")
        start = meta.get("start_line")
        end = meta.get("end_line")
        dist_txt = f"  [dist={dist:.3f}]" if (show_dist and isinstance(dist, (int, float))) else ""
        header = f"// {path}  (Zeilen {start}-{end}){dist_txt}\n"
        block = header + (doc or "")
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)

def auto_lang_from_meta(client, collection_name: str) -> str:
    try:
        coll = client.get_or_create_collection(collection_name)
        meta = getattr(coll, "metadata", None) or {}
        lang = (meta.get("lang") or "").lower()
        if lang in ("php", "cpp", "generic"):
            return lang
    except Exception:
        pass
    # Heuristik aus Name
    n = (collection_name or "").lower()
    if "php" in n: return "php"
    if "cpp" in n or "c++" in n or "cxx" in n: return "cpp"
    return "generic"

st.set_page_config(page_title="Ollama Code Assistant", page_icon="💬", layout="wide")
st.title("💬 Code Assistant mit Ollama (RAG)")

with st.sidebar:
    st.header("🔧 Einstellungen")
    db_path = st.text_input("Pfad zur ChromaDB", value="./chroma_db")

    # Chroma verbinden & Collections laden
    try:
        client = chromadb.PersistentClient(path=str(Path(db_path)), settings=Settings(anonymized_telemetry=False))
        cols = client.list_collections()
        collection_names = [c.name for c in cols] if cols else []
    except Exception as e:
        client = None
        collection_names = []
        st.error(f"Chroma-Verbindung fehlgeschlagen: {e}")

    # Dropdown für Collections (Default cpp_repo)
    default_coll = "cpp_repo"
    if default_coll not in collection_names:
        collection_names = [default_coll] + [n for n in collection_names if n != default_coll]
    collection = st.selectbox("Collection (Projekt)", options=collection_names or [default_coll], index=0)

    # Sprache automatisch aus Collection-Metadaten, Override möglich
    auto_lang = auto_lang_from_meta(client, collection) if client else "generic"
    lang = st.selectbox("Sprache/Prompt", ["php", "cpp", "generic"],
                        index=["php","cpp","generic"].index(auto_lang) if auto_lang in ("php","cpp","generic") else 2)

    # Modelle
    model = st.selectbox("Antwort-Modell", ["deepseek-r1:8b", "gpt-oss:20b", "llama3.1:8b"], index=0)
    embed_model = st.text_input("Embeddings-Modell", value="nomic-embed-text")

    topk = st.slider("Top-K Snippets", 1, 20, 8)
    max_chars = st.slider("Max Kontext-Zeichen", 2000, 30000, 12000, step=1000)
    show_dist = st.checkbox("Ähnlichkeitswerte anzeigen", value=False)
    show_ctx_box = st.checkbox("Verwendete Snippets anzeigen", value=True)

frage = st.text_area("Frage zur Codebasis", placeholder="Wie wird das Login validiert?")

col_run1, col_run2 = st.columns([1,1])
with col_run1:
    go = st.button("Antwort abrufen", type="primary")
with col_run2:
    ping = st.button("Embeddings testen")

if ping:
    try:
        emb = ollama.embeddings(model=embed_model, prompt="ping").get("embedding", [])
        st.success(f"Embeddings OK (len={len(emb)})")
    except Exception as e:
        st.error(f"Embedding-Test fehlgeschlagen: {e}")

if go:
    if not client:
        st.error("Keine Chroma-Verbindung.")
    elif not frage.strip():
        st.warning("Bitte gib eine Frage ein.")
    else:
        try:
            with st.spinner("Suche relevante Snippets…"):
                coll = client.get_or_create_collection(collection)
                q_emb = ollama.embeddings(model=embed_model, prompt=frage)["embedding"]
                results = coll.query(
                    query_embeddings=[q_emb],
                    n_results=topk,
                    include=["documents", "metadatas", "distances"]
                )
                ctx = format_context(results, max_chars=max_chars, show_dist=show_dist)

            if not ctx.strip():
                st.info("Keine passenden Snippets gefunden. Versuche eine spezifischere Frage oder erhöhe Top‑K.")
            else:
                if show_ctx_box:
                    code_lang = "php" if lang == "php" else "cpp" if lang == "cpp" else "text"
                    with st.expander("🔎 Verwendete Snippets (Kontext)"):
                        st.code(ctx, language=code_lang)

                system_prompt = LANG_PROMPTS.get(lang, LANG_PROMPTS["generic"])
                user_prompt = textwrap.dedent(f"""
                Beantworte die folgende Frage zur Codebasis mithilfe des Kontexts.

                ### Kontext (Snippets mit Pfaden und Zeilen)
                {ctx}

                ### Frage
                {frage}

                ### Anforderungen
                - Begründe kurz deine Antwort.
                - Zitiere IMMER Pfad + Zeilen.
                - Wenn Codeänderungen nötig sind: zeige konkrete Patches (Diff-ähnlich) oder Code-Auszüge.
                """)

                with st.spinner(f"Frage {model}…"):
                    msgs = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    resp = ollama.chat(model=model, messages=msgs)

                st.subheader("Antwort")
                st.markdown(resp["message"]["content"])

        except Exception as e:
            st.error(f"Fehler: {e}")
            st.stop()
