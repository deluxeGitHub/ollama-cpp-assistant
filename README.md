PHP-Code-RAG mit Ollama – Minimalbeispiel

Dieses Mini-Projekt zeigt, wie du ein größeres PHP-Repository für Q&A nutzbar machst:

Index: PHP-Dateien parsen (per Tree‑sitter), Funktionen/Klassen extrahieren, Embeddings mit Ollama erzeugen, in ChromaDB speichern.

Query: Frage → Embedding → Top-Snippets holen → zusammen mit der Frage an ein Ollama-Chatmodell schicken.

Getestet mit: Python ≥3.10, Ollama (lokal), Modelle: nomic-embed-text (Embeddings) und llama3.1:8b (Chat).

Nutzung

# Modelle holen & Ollama starten

ollama pull nomic-embed-text
ollama pull llama3.1:8b

# Python 3.11 installieren (von python.org oder via winget)
winget install Python.Python.3.11

# Neues venv mit 3.11
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Index bauen

python index_repo.py --repo "C:\Users\samuel\source\repos\afx\src\afx_commons" --db .\chroma_db --project afx --lang cpp
# Fragen stellen
python query_repo.py --db ./chroma_db "Wie funktioniert das Login und wo werden Passwörter gehashed?"

# UI Starten
streamlit run rag_ui.py