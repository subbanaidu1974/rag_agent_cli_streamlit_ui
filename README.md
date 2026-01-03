# RAG Pipeline — README

This repository contains a Streamlit-based RAG preprocessing pipeline and CLI tools to generate and query Chroma collections.

Files:
- `tabbed_steps.py` — Streamlit UI: Crawl → Load → Split → Persist → View.
- `params_template.csv` — CSV template with pipeline parameters.
- `generate_chroma_from_csv.py` — CLI: read CSV, crawl, chunk, embed, persist to Chroma.
- `rag_agent_cli.py` — CLI RAG agent: load Chroma collection, retrieve top-k docs, synthesize an answer with LLM.

Prerequisites
- Python 3.8+
- OpenAI API key with access to the chosen model.

Install dependencies
```bash
python -m pip install --upgrade pip
pip install langchain openai chromadb python-dotenv numpy scikit-learn pandas streamlit
```

Environment
- Create a `.env` file in the repo root containing:
```text
OPENAI_API_KEY=sk-...
```
The scripts call `load_dotenv()` to pick this up.

CSV parameters
- See `params_template.csv` for expected columns:
  - `base_url,url,headless,max_depth,num_urls,crawl_matching_query,relevance_threshold,chunk_size,overlap`
- Edit or add rows to `params_template.csv` for each site/collection you want to create.

Run the Streamlit app (interactive UI)
```bash
streamlit run tabbed_steps.py
```

Generate Chroma collections from CSV (CLI)
- Dry run (parse CSV without persisting):
```bash
python generate_chroma_from_csv.py --csv params_template.csv --dry-run
```
- Persist collections (crawl → chunk → embed → persist):
```bash
python generate_chroma_from_csv.py --csv params_template.csv
```

RAG Agent CLI (query an existing collection)
```bash
python rag_agent_cli.py --coll-dir ./chroma_store/<collection_dir> --query "Your question" --k 5
```
If the collection exists under `./chroma_store/<collection>`, pass that path to `--coll-dir` and optionally `--coll-name`.

Notes & Troubleshooting
- If imports fail at runtime, ensure required packages are installed in the active environment.
- LangChain and Chroma packaging/layouts change across versions. If the CLI or Streamlit app shows import errors, run `pip show langchain` and share the version so the code can be adapted.
- If LLM calls fail, verify `OPENAI_API_KEY` and model access.

License
- No license included; treat as personal project code.

Contact
- For changes or additional CLI flags, open an issue or ask for updates.

## Architecture Diagrams

The repository includes Mermaid diagrams for the Streamlit UI flow and the CLI flow under `diagrams/`.

Files:
- `diagrams/streamlit_flow.mmd` — Streamlit UI flow (Crawl → Load → Split → Persist → View + RAG Agent)
- `diagrams/cli_flow.mmd` — CSV-driven generator and RAG CLI flow

You can render these Mermaid files with your preferred Mermaid tool (e.g., `mmdc` mermaid-cli) or paste the contents into an online Mermaid editor.

Inline Mermaid (Streamlit UI flow):
```mermaid
flowchart TD
  U[User] -->|Interacts| S[Streamlit UI (`tabbed_steps.py`)]
  S --> Crawl[Step 1: Crawl (crawler.py)]
  Crawl --> CrawledDocs[Stored: crawled_docs (preview + full_content)]
  S --> Load[Step 2: Load Documents]
  Load --> Docs[st.session_state.documents]
  S --> Split[Step 3: Split into Chunks]
  Split --> Chunks[st.session_state.chunks]
  Split --> PersistBtn[Step 4: Persist to Chroma (💾)]
  PersistBtn --> Embeds[OpenAIEmbeddings]
  Embeds --> ChromaPersist[Chroma collection (./chroma_store/<collection>)]
  ChromaPersist --> Backup[backup_stored_docs.jsonl]
  S --> View[Step 5: View Stored]
  View --> CollectionsViewer[Load / Inspect Chroma]
  S --> RAGPanel[RAG Agent (center pane)]
  RAGPanel -->|uses| ChromaPersist
  RAGPanel --> Retrieval[Vector retrieval (similarity_search / query)]
  Retrieval --> BuildCtx[Build context from top-K docs]
  BuildCtx --> Chat[LangChain ChatOpenAI]
  Chat --> Answer[LLM response shown to user]
  Note[Notes: .env loads OPENAI_API_KEY; store & match state in st.session_state] --> S
```

Inline Mermaid (CLI flow):
```mermaid
flowchart TD
  CSV[params_template.csv] --> CSVParser[generate_chroma_from_csv.py]
  CSVParser --> ForEachRow[For each row]
  ForEachRow --> CrawlCLI[Crawl (crawler.py)]
  CrawlCLI --> Pages[Collected pages]
  Pages -->|optional| SemanticFilter[OpenAIEmbeddings semantic pre-filter]
  SemanticFilter --> PagesFiltered
  PagesFiltered --> Chunking[Chunk into pieces]
  Chunking --> Docs[Create langchain Documents]
  Docs --> EmbeddingsCLI[OpenAIEmbeddings]
  EmbeddingsCLI --> PersistCLI[Chroma.from_documents -> ./chroma_store/<collection>]
  PersistCLI --> BackupCLI[backup_stored_docs.jsonl]
  ---
  # RAG CLI:
  UserCLI[User] --> RAGCLI[rag_agent_cli.py]
  RAGCLI --> LoadCollection[Load Chroma collection (persist_directory)]
  LoadCollection --> Retrieve[Retrieve top-K docs]
  Retrieve --> BuildCtxCLI[Assemble context]
  BuildCtxCLI --> ChatCLI[LangChain ChatOpenAI]
  ChatCLI --> PrintAnswer[Print LLM response to stdout]
```
