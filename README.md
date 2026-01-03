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
