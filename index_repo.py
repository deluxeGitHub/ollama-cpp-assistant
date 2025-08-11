import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import re
import hashlib
import time

import chromadb
from chromadb.config import Settings
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from tqdm import tqdm
import ollama

# ---------- Settings ----------
MAX_FILE_MB = 3.0
MAX_EMBED_CHARS = 8000
MIN_EMBED_CHARS = 20
EMBED_RETRIES = 3
RETRY_SLEEP_SEC = 0.6

CPP_EXTS = {".cpp", ".cc", ".cxx", ".c", ".hpp", ".hh", ".hxx", ".h", ".ipp", ".inl"}

IGNORE_DIRS = {
    ".git", ".idea", ".vscode", ".vs", ".cache", "build", "cmake-build-debug",
    "cmake-build-release", "out", "dist", "bin", "obj", "lib", "target",
    "third_party", "external", "deps", "_deps", "vcpkg_installed", ".conan", ".m2"
}
IGNORE_PATH_SUBSTRINGS = {
    "third_party/", "external/", "/_deps/", "vcpkg_installed/", "/.conan/",
}
IGNORE_FILE_EXT = {".min.cpp"}  # falls es sowas gibt – unwahrscheinlich, aber der Vollständigkeit halber


# ---------- Utils ----------
def sanitize_meta(meta: dict) -> dict:
    clean = {}
    for k, v in meta.items():
        if v is None:
            v = ""
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def should_skip(path: Path) -> bool:
    if path.suffix.lower() not in CPP_EXTS:
        return True
    parents_names_lower = {p.name.lower() for p in path.parents}
    if parents_names_lower & {d.lower() for d in IGNORE_DIRS}:
        return True
    pnorm = str(path).replace("\\", "/").lower()
    for sub in IGNORE_PATH_SUBSTRINGS:
        if sub.lower() in pnorm:
            return True
    try:
        if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            return True
    except Exception:
        return True
    if any(pnorm.endswith(ext) for ext in IGNORE_FILE_EXT):
        return True
    return False

def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def make_doc_id(path: Path, start: int, end: int, idx: int, text: str) -> str:
    h = hashlib.blake2s(text.encode("utf-8"), digest_size=6).hexdigest()
    return f"{path}:{start}-{end}#{idx}:{h}"

def _split_long_text(text: str, max_chars: int = MAX_EMBED_CHARS) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    lines = text.splitlines()
    out, buf, total = [], [], 0
    for ln in lines:
        add = len(ln) + 1
        if total + add > max_chars:
            if buf:
                out.append("\n".join(buf))
            buf, total = [ln], add
        else:
            buf.append(ln)
            total += add
    if buf:
        out.append("\n".join(buf))
    return [s for s in out if s and s.strip()]

def safe_embed_docs(ids: List[str], docs: List[str], metas: List[Dict]) -> Tuple[List[str], List[str], List[Dict], List[List[float]]]:
    out_ids, out_docs, out_metas, out_embs = [], [], [], []
    for _id, doc, meta in zip(ids, docs, metas):
        if not doc or len(doc.strip()) < MIN_EMBED_CHARS:
            continue
        for piece_idx, piece in enumerate(_split_long_text(doc, MAX_EMBED_CHARS)):
            if len(piece.strip()) < MIN_EMBED_CHARS:
                continue
            emb = None
            for attempt in range(EMBED_RETRIES):
                try:
                    r = ollama.embeddings(model="nomic-embed-text", prompt=piece)
                    emb = r.get("embedding", None)
                    if emb and isinstance(emb, list) and len(emb) > 0:
                        break
                except Exception:
                    pass
                time.sleep(RETRY_SLEEP_SEC * (attempt + 1))
            if not emb:
                # letzte Chance: stark verkürzt
                short = piece[:2000]
                try:
                    r = ollama.embeddings(model="nomic-embed-text", prompt=short)
                    emb = r.get("embedding", None) if isinstance(r, dict) else None
                except Exception:
                    emb = None
            if not emb:
                continue
            new_id = f"{_id}@{piece_idx}"
            out_ids.append(new_id)
            out_docs.append(piece)
            out_metas.append(meta)
            out_embs.append(emb)
    return out_ids, out_docs, out_metas, out_embs


# ---------- C++ Symbol Extraction ----------
# Tree-sitter-Knotentypen (cpp):
# - function_definition
# - declaration (mit function_declarator)
# - class_specifier / struct_specifier
# - namespace_definition
# - enum_specifier
NAME_NODES = {"type_identifier", "field_identifier", "identifier", "namespace_identifier"}

FUNC_SIG_RE = re.compile(r'([A-Za-z_][\w:]*)\s*\(', re.MULTILINE)

def _node_text(code: str, n) -> str:
    return code[n.start_byte:n.end_byte]

def _first_name_child_text(code: str, n) -> str:
    for c in n.children:
        if c.type in NAME_NODES:
            return _node_text(code, c)
    # tiefer suchen (heuristik)
    for c in n.children:
        t = _first_name_child_text(code, c)
        if t:
            return t
    return ""

def extract_cpp_symbols(code: str) -> List[Dict]:
    # 1) Tree-sitter
    try:
        lang: Language = get_language("cpp")
        parser = Parser()
        parser.set_language(lang)
        tree = parser.parse(bytes(code, "utf-8"))
        root = tree.root_node

        out: List[Dict] = []

        def walk(n, cls_or_ns: str | None = None):
            t = n.type

            # Klassen/Structs
            if t in ("class_specifier", "struct_specifier"):
                name = _first_name_child_text(code, n) or ""
                meta = {
                    "kind": "class" if t == "class_specifier" else "struct",
                    "name": name,
                    "start_line": n.start_point[0] + 1,
                    "end_line": n.end_point[0] + 1,
                }
                out.append({"text": _node_text(code, n), "meta": meta})
                # Kinder durchlaufen (für Methoden)
                for c in n.children:
                    walk(c, cls_or_ns=name or cls_or_ns)
                return  # schon tiefer gelaufen

            # Namespace
            if t == "namespace_definition":
                name = _first_name_child_text(code, n) or ""
                meta = {
                    "kind": "namespace",
                    "name": name,
                    "start_line": n.start_point[0] + 1,
                    "end_line": n.end_point[0] + 1,
                }
                out.append({"text": _node_text(code, n), "meta": meta})
                for c in n.children:
                    walk(c, cls_or_ns=name or cls_or_ns)
                return

            # Funktionsdefinition (mit Body)
            if t == "function_definition":
                name = _first_name_child_text(code, n) or ""
                if cls_or_ns and name and "::" not in name:
                    name = f"{cls_or_ns}::{name}"
                meta = {
                    "kind": "function",
                    "name": name,
                    "start_line": n.start_point[0] + 1,
                    "end_line": n.end_point[0] + 1,
                }
                out.append({"text": _node_text(code, n), "meta": meta})

            # Deklarationen (z. B. Header-Prototypen)
            if t == "declaration":
                txt = _node_text(code, n)
                # Heuristik: nur „funktionsartige“ Deklarationen
                if "(" in txt and ")" in txt and ";" in txt:
                    m = FUNC_SIG_RE.search(txt)
                    if m:
                        name = m.group(1)
                        if cls_or_ns and name and "::" not in name:
                            name = f"{cls_or_ns}::{name}"
                        meta = {
                            "kind": "decl",
                            "name": name,
                            "start_line": n.start_point[0] + 1,
                            "end_line": n.end_point[0] + 1,
                        }
                        out.append({"text": txt, "meta": meta})

            for c in n.children:
                walk(c, cls_or_ns)

        walk(root)
        if out:
            return out
    except Exception:
        pass

    # 2) Fallback: Regex & Chunking
    out: List[Dict] = []
    # sehr grob: freie Funktionssignaturen
    for m in FUNC_SIG_RE.finditer(code):
        s = code.rfind('\n', 0, m.start())
        s = 0 if s < 0 else s + 1
        e = code.find('\n{', m.end())
        if e == -1:
            e = code.find('\n;', m.end())
        if e == -1:
            e = min(len(code), s + 800)  # heuristischer cut
        text = code[s:e]
        start_line = code.count('\n', 0, s) + 1
        end_line = code.count('\n', 0, e) + 1
        out.append({"text": text, "meta": {"kind": "function", "name": m.group(1), "start_line": start_line, "end_line": end_line}})
    if out:
        return out

    # 3) Zeilen-Chunking
    lines = code.splitlines()
    chunk = 120
    chunks: List[Dict] = []
    for i in range(0, len(lines), chunk):
        text = "\n".join(lines[i:i+chunk])
        chunks.append({"text": text, "meta": {"kind": "chunk", "name": "", "start_line": i+1, "end_line": min(i+chunk, len(lines))}})
    return chunks


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Pfad zum C++-Repository")
    ap.add_argument("--db", required=True, help="Pfad zu ChromaDB (persist)")
    ap.add_argument("--collection", default="cpp_repo", help="Chroma-Collection-Name")
    ap.add_argument("--project", required=False, help="Projekt-/Collection-Name (z. B. 'afx')")
    ap.add_argument("--lang", default="cpp", choices=["php","cpp","generic"], help="Primärsprache des Projekts")
    args = ap.parse_args()
    collection_name = (args.project or args.collection or "cpp_repo")

    repo = Path(args.repo)
    dbdir = Path(args.db)
    dbdir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(dbdir), settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine", "project": args.project or collection_name, "lang": args.lang}
    )


    # C++-Dateien sammeln
    cpp_files = [p for p in repo.rglob("*") if p.is_file() and not should_skip(p)]
    print(f"Indexiere {len(cpp_files)} C/C++-Dateien…")

    for f in tqdm(cpp_files):
        try:
            code = read_file(f)
            symbols = extract_cpp_symbols(code)

            # Alte Chunks dieser Datei entfernen (inkrementell)
            coll.delete(where={"path": str(f)})

            ids, docs, metas = [], [], []
            for idx, sym in enumerate(symbols):
                doc = sym["text"]
                raw_meta = sym["meta"] | {"path": str(f), "project": args.project or collection_name, "lang": args.lang}
                meta = sanitize_meta(raw_meta)
                start = int(meta.get("start_line", 0) or 0)
                end = int(meta.get("end_line", 0) or 0)
                _id = make_doc_id(f, start, end, idx, doc)
                if doc and doc.strip():
                    ids.append(_id); docs.append(doc); metas.append(meta)

            if not ids:
                continue

            ids2, docs2, metas2, embs2 = safe_embed_docs(ids, docs, metas)
            if ids2 and len(ids2) == len(docs2) == len(metas2) == len(embs2):
                coll.add(ids=ids2, documents=docs2, metadatas=metas2, embeddings=embs2)
            else:
                if not ids2:
                    print(f"[WARN] Keine validen Embeddings für {f} erzeugt.")

        except Exception as e:
            print(f"[WARN] Fehler bei {f}: {e}")

    print("Fertig. ChromaDB gespeichert unter:", dbdir)


if __name__ == "__main__":
    main()
