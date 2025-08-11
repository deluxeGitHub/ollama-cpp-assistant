import streamlit as st
from pathlib import Path
import subprocess, sys, textwrap
import chromadb
from chromadb.config import Settings
import ollama

# ---------------- Prompt-Vorlagen ----------------
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

# ---------------- Hilfsfunktionen ----------------
def format_context(results, max_chars=16000, show_dist=False):
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
    n = (collection_name or "").lower()
    if "php" in n: return "php"
    if "cpp" in n or "c++" in n or "cxx" in n: return "cpp"
    return "generic"

def run_index_subprocess(python_exe: str, script_path: Path, args: list[str]):
    proc = subprocess.Popen(
        [python_exe, str(script_path), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:  # type: ignore
        yield line.rstrip("\n")
    proc.wait()
    yield f"[Indexer beendet] Exit-Code: {proc.returncode}"

# ---------------- UI Setup ----------------
st.set_page_config(page_title="Ollama Code Assistant", page_icon="💬", layout="wide")
st.title("💬 Code Assistant mit Ollama (RAG)")

# Sidebar: Settings
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

    # Retrieval-Parameter
    topk = st.slider("Top-K Snippets", 1, 30, 20)
    max_chars = st.slider("Max Kontext-Zeichen", 2000, 30000, 16000, step=1000)
    show_dist = st.checkbox("Ähnlichkeitswerte anzeigen", value=False)
    show_ctx_box = st.checkbox("Verwendete Snippets anzeigen", value=True)

    # Embedding-Ping
    if st.button("Embeddings testen"):
        try:
            emb = ollama.embeddings(model=embed_model, prompt="ping").get("embedding", [])
            st.success(f"Embeddings OK (len={len(emb)})")
        except Exception as e:
            st.error(f"Embedding-Test fehlgeschlagen: {e}")

# --------- Chat State ---------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "last_ctx" not in st.session_state:
    st.session_state.last_ctx = ""  # zuletzt verwendeter Kontext (optional sichtbar)

# Chatverlauf rendern
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat-Eingabe (wie ChatGPT)
user_input = st.chat_input("Frage zur Codebasis eingeben…")
if user_input:
    # 1) User-Post in Verlauf
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Retrieval + Antwort erzeugen
    try:
        with st.spinner("Suche relevante Snippets…"):
            if not client:
                st.error("Keine Chroma-Verbindung.")
                st.stop()
            coll = client.get_or_create_collection(collection)
            q_emb = ollama.embeddings(model=embed_model, prompt=user_input)["embedding"]
            results = coll.query(
                query_embeddings=[q_emb],
                n_results=topk,
                include=["documents", "metadatas", "distances"]
            )
            ctx = format_context(results, max_chars=max_chars, show_dist=show_dist)
            st.session_state.last_ctx = ctx  # speichern für optionalen Blick

        if not ctx.strip():
            assistant_text = "Ich habe keine passenden Snippets gefunden. Versuche eine spezifischere Frage oder erhöhe die Anzahl Top‑K."
        else:
            if show_ctx_box:
                with st.expander("🔎 Verwendete Snippets (Kontext)"):
                    code_lang = "php" if lang == "php" else "cpp" if lang == "cpp" else "text"
                    st.code(ctx, language=code_lang)

            system_prompt = LANG_PROMPTS.get(lang, LANG_PROMPTS["generic"])
            user_prompt = textwrap.dedent(f"""
            Beantworte die folgende Frage zur Codebasis mithilfe des Kontexts.

            ### Kontext (Snippets mit Pfaden und Zeilen)
            {ctx}

            ### Frage
            {user_input}

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
            assistant_text = resp["message"]["content"]

        # 3) Assistant-Post in Verlauf + rendern
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)

    except Exception as e:
        st.error(f"Fehler: {e}")

# --------- Reindex-Sektion (default hidden) ---------
with st.expander("🧱 Index neu aufbauen / aktualisieren (Advanced)", expanded=False):
    st.caption("Für seltene Neu-Indexierungen – standardmäßig ausgeblendet.")
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        repo_path = st.text_input("Repo-Pfad (Ordner der Codebasis)", value="", key="repo_path")
    with col2:
        project_name = st.text_input("Projektname (Collection)", value=collection or "cpp_repo", key="project_name")
    with col3:
        index_lang = st.selectbox("Sprache fürs Indexing", ["php", "cpp", "generic"],
                                  index=["php","cpp","generic"].index(lang) if lang in ("php","cpp","generic") else 2,
                                  key="index_lang")

    if st.button("Index aufbauen/aktualisieren", key="idx_go"):
        if not repo_path.strip():
            st.warning("Bitte Repo-Pfad angeben.")
        else:
            index_script = Path("index_cpp.py" if index_lang == "cpp" else "index_repo.py")
            if not index_script.exists():
                st.error(f"{index_script.name} nicht gefunden neben rag_ui.py.")
            else:
                # ollama ping
                try:
                    _ = ollama.embeddings(model="nomic-embed-text", prompt="ping")["embedding"]
                except Exception as e:
                    st.error(f"Embeddings nicht verfügbar (Ollama läuft/Modell geladen?): {e}")
                    st.stop()

                cmd_args = [
                    "--repo", repo_path,
                    "--db", str(Path(db_path)),
                    "--project", project_name,
                    "--lang", index_lang,
                ]
                st.info(f"Starte {index_script.name} …")
                log_box = st.empty()
                log_lines = []
                for line in run_index_subprocess(sys.executable, index_script, cmd_args):
                    log_lines.append(line)
                    log_box.code("\n".join(log_lines[-200:]), language="text")
                st.success("Indexing abgeschlossen. Öffne/wechsel die Collection oben, um sie zu verwenden.")

# Kleiner Reset-Button (optional)
st.sidebar.divider()
if st.sidebar.button("🧹 Chat zurücksetzen"):
    st.session_state.messages = []
    st.session_state.last_ctx = ""
    st.rerun()
