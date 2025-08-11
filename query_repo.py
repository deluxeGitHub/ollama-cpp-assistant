import argparse
from pathlib import Path
import textwrap
import re

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
        "Schlage präzise Code-Änderungen vor."
    ),
}

SYMBOL_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_:]{2,}")

def pick_lang_from_meta(client, collection_name: str) -> str:
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

def expand_neighbors(results, window=1):
    """Nimmt zusätzlich angrenzende Treffer (selbe Datei) und passende Header/Source-Paare hinein."""
    if not results or not results.get("documents"):
        return results
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids   = results["ids"][0]

    by_file = {}
    for i, meta in enumerate(metas):
        path = meta.get("path")
        by_file.setdefault(path, []).append((i, int(meta.get("start_line") or 0), int(meta.get("end_line") or 0)))

    add_idx = set()
    for path, hits in by_file.items():
        hits.sort(key=lambda x: x[1])
        for i, (idx, s, e) in enumerate(hits):
            for off in range(1, window+1):
                if i-off >= 0: add_idx.add(hits[i-off][0])
                if i+off < len(hits): add_idx.add(hits[i+off][0])

    def mates(path: str):
        p = Path(path)
        base = p.stem
        if p.suffix in (".cpp", ".cc", ".cxx", ".c"):
            return [str(p.with_suffix(ext)) for ext in (".h", ".hpp", ".hh", ".hxx")]
        if p.suffix in (".h", ".hpp", ".hh", ".hxx"):
            return [str(p.with_suffix(ext)) for ext in (".cpp", ".cc", ".cxx")]
        return []

    paths = [m.get("path") for m in metas]
    for i, meta in enumerate(metas):
        for m in mates(meta.get("path")):
            if m in paths:
                j = paths.index(m)
                add_idx.add(j)

    keep = list(range(len(docs)))
    for j in sorted(add_idx):
        if j not in keep:
            keep.append(j)

    results["documents"][0] = [docs[i] for i in keep]
    results["metadatas"][0] = [metas[i] for i in keep]
    results["ids"][0]       = [ids[i] for i in keep]
    return results

def keyword_boost(results, question):
    tokens = {t.lower() for t in SYMBOL_RE.findall(question) if len(t) > 2}
    if not tokens or not results or not results.get("documents"):
        return results
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results.get("distances", [[]])[0] if results.get("distances") else [None]*len(docs)

    scored = []
    for doc, meta, dist in zip(docs, metas, dists):
        base = -dist if isinstance(dist, (int, float)) else 0.0  # kleinere Distanz ist besser
        path = (meta.get("path") or "").lower()
        name = (meta.get("name") or "").lower()
        text = (doc or "").lower()
        hits = 0
        for t in tokens:
            if t in path or t in name:
                hits += 2
            elif t in text:
                hits += 1
        scored.append((base + 0.2 * hits, doc, meta))
    scored.sort(reverse=True, key=lambda x: x[0])
    results["documents"][0] = [d for _, d, _ in scored]
    results["metadatas"][0] = [m for _, _, m in scored]
    return results

def mmr_diversify(docs, metas, k=12, lambda_div=0.7):
    """Einfaches MMR anhand Länge (Nutzen) + Pfad/Zeilen-Überlappung (Redundanz)."""
    scores = [len(d or "") for d in docs]
    selected, cand = [], list(range(len(docs)))
    while cand and len(selected) < k:
        if not selected:
            j = max(cand, key=lambda i: scores[i])
            selected.append(j); cand.remove(j); continue
        def redundancy(i):
            m = metas[i]; p = m.get("path"); s=(m.get("start_line") or 0); e=(m.get("end_line") or 0)
            red = 0
            for j in selected:
                mj = metas[j]; pj = mj.get("path"); sj=(mj.get("start_line") or 0); ej=(mj.get("end_line") or 0)
                if p == pj and not (e < sj or s > ej):
                    red += 1
            return red
        j = max(cand, key=lambda i: lambda_div*scores[i] - (1-lambda_div)*redundancy(i))
        selected.append(j); cand.remove(j)
    return [docs[i] for i in selected], [metas[i] for i in selected]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Pfad zu ChromaDB (persist)")
    ap.add_argument("--collection", default="cpp_repo", help="Collection/Projekt (z. B. 'afx')")
    ap.add_argument("--lang", choices=["php","cpp","generic"], help="Prompt-Sprache (sonst auto)")
    ap.add_argument("--topk", type=int, default=20, help="Anzahl initialer Snippets")
    ap.add_argument("--max-chars", type=int, default=16000, help="Max. Zeichen für Kontext")
    ap.add_argument("--model", default="deepseek-r1:8b", help="Ollama-Antwortmodell")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="Ollama-Embeddingsmodell")
    ap.add_argument("--show-dist", action="store_true", help="Ähnlichkeitswerte im Kontext anzeigen")
    ap.add_argument("--neighbors", type=int, default=1, help="Fenstergröße angrenzender Treffer je Datei")
    ap.add_argument("--hybrid", action="store_true", help="Keyword-Boost zusätzlich zur Embedding-Suche")
    ap.add_argument("--mmr", action="store_true", help="Diversität via MMR (kürzt auf ~12 Docs)")
    ap.add_argument("frage", help="Deine Frage zur Codebasis")
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=str(Path(args.db)), settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(args.collection)

    lang = args.lang or pick_lang_from_meta(client, args.collection)
    system_prompt = LANG_PROMPTS.get(lang, LANG_PROMPTS["generic"])

    q_emb = ollama.embeddings(model=args.embed_model, prompt=args.frage)["embedding"]
    results = coll.query(query_embeddings=[q_emb], n_results=args.topk, include=["documents", "metadatas", "distances", "ids"])

    if not results or not results.get("documents") or not results["documents"][0]:
        print("\n=== Antwort ===\nKeine passenden Snippets gefunden. Versuche eine spezifischere Frage oder erhöhe --topk.")
        return

    # Upgrades: Nachbarn, Hybrid-Boost, MMR
    if args.neighbors and args.neighbors > 0:
        results = expand_neighbors(results, window=args.neighbors)
    if args.hybrid:
        results = keyword_boost(results, args.frage)
    if args.mmr:
        docs = results["documents"][0]; metas = results["metadatas"][0]
        if len(docs) > 20:
            docs2, metas2 = mmr_diversify(docs, metas, k=12)
            results["documents"][0] = docs2
            results["metadatas"][0] = metas2

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
