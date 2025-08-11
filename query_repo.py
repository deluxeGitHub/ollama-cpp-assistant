import argparse
from pathlib import Path
import textwrap
import chromadb
from chromadb.config import Settings
import ollama

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
        "Schlage konkrete, präzise Code-Änderungen vor."
    ),
}

def pick_lang_from_collection(name: str) -> str:
    n = (name or "").lower()
    if "php" in n:
        return "php"
    if "cpp" in n or "c++" in n or "cxx" in n:
        return "cpp"
    return "generic"

def format_context(results, max_chars=12000, show_dist=False):
    if not results or not results.get("documents"):
        return ""
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results.get("distances", [[]])[0] if show_dist else [None] * len(docs)

    parts = []
    total = 0
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Pfad zu ChromaDB (persist)")
    ap.add_argument("--collection", default="php_repo", help="Chroma-Collection-Name")
    ap.add_argument("--lang", choices=["php", "cpp", "generic"], help="Erzwinge Prompt-Sprache (sonst auto)")
    ap.add_argument("--topk", type=int, default=8, help="Anzahl Snippets")
    ap.add_argument("--max-chars", type=int, default=12000, help="Max. Zeichen für Kontext")
    ap.add_argument("--model", default="gpt-oss:20b", help="Ollama-Modellname")
    ap.add_argument("--show-dist", action="store_true", help="Distanzen (Ähnlichkeiten) im Kontext anzeigen")
    ap.add_argument("frage", help="Deine Frage zur Codebasis")
    args = ap.parse_args()

    # System-Prompt auswählen
    lang = args.lang or pick_lang_from_collection(args.collection)
    system_prompt = LANG_PROMPTS.get(lang, LANG_PROMPTS["generic"])

    # Chroma verbinden
    client = chromadb.PersistentClient(
        path=str(Path(args.db)),
        settings=Settings(anonymized_telemetry=False)
    )
    coll = client.get_or_create_collection(args.collection)

    # Query-Embedding
    q_emb = ollama.embeddings(model="nomic-embed-text", prompt=args.frage)["embedding"]

    results = coll.query(
        query_embeddings=[q_emb],
        n_results=args.topk,
        include=["documents", "metadatas", "distances"]
    )

    if not results or not results.get("documents") or not results["documents"][0]:
        print("\n=== Antwort ===\n")
        print("Ich habe keine passenden Snippets gefunden. "
              "Versuche eine spezifischere Frage oder erhöhe --topk.")
        return

    ctx = format_context(results, max_chars=args.max_chars, show_dist=args.show_dist)

    user_prompt = textwrap.dedent(f"""
    Beantworte die folgende Frage zur Codebasis mithilfe des Kontexts.

    ### Kontext (Snippets mit Pfaden und Zeilen)
    {ctx}

    ### Frage
    {args.frage}

    ### Anforderungen
    - Begründe kurz deine Antwort.
    - Zitiere IMMER Pfad + Zeilen.
    - Wenn Codeänderungen nötig sind: zeige konkrete Patches (Diff-ähnlich) oder Code-Auszüge.
    """)

    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = ollama.chat(model=args.model, messages=msgs)
    print("\n=== Antwort ===\n")
    print(resp["message"]["content"])  # type: ignore

if __name__ == "__main__":
    main()
