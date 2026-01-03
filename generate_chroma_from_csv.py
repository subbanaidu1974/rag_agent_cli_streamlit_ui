#!/usr/bin/env python3
"""
CLI: read CSV of pipeline parameters and generate Chroma collections for each base_url.
Usage:
    python generate_chroma_from_csv.py --csv params.csv

The CSV should include columns (headers):
- base_url,url,headless,max_depth,num_urls,crawl_matching_query,relevance_threshold,chunk_size,overlap

This script uses the same crawler/chunking/embedding+Chroma persistence logic as the Streamlit app.
"""
import csv
import os
import re
import argparse
import time
from datetime import datetime

import numpy as np
# Load .env
from dotenv import load_dotenv
load_dotenv()
dotenv_loaded = True

# Chroma + embeddings / LLM clients (direct imports only)
from langchain_openai import OpenAIEmbeddings
from crawler import Crawler
import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from langchain_chroma import Chroma


CHROMA_DIR = "./chroma_store"


def sanitize_name(s):
    if not s:
        return 'chroma_index'
    return re.sub(r'[^0-9a-zA-Z_-]', '_', s)[:64] or 'chroma_index'


def chunk_text(text, chunk_size=2000, overlap=200):
    chunks = []
    i = 0
    idx = 0
    if not text:
        return chunks
    while i < len(text):
        chunk = text[i:i + chunk_size]
        if chunk and chunk.strip():
            chunks.append((idx, chunk))
            idx += 1
        i += max(1, chunk_size - overlap)
    return chunks


def process_row(row, dry_run=False):
    base_url = row.get('base_url') or row.get('url') or ''
    url = row.get('url') or base_url
    headless = str(row.get('headless', 'True')).lower() in ('1', 'true', 'yes')
    try:
        max_depth = int(row.get('max_depth', 2))
    except Exception:
        max_depth = 2
    try:
        num_urls = int(row.get('num_urls', 5))
    except Exception:
        num_urls = 5
    crawl_matching_query = row.get('crawl_matching_query', '') or ''
    try:
        relevance_threshold = float(row.get('relevance_threshold', 0.75))
    except Exception:
        relevance_threshold = 0.75
    try:
        chunk_size = int(row.get('chunk_size', 2000))
    except Exception:
        chunk_size = 2000
    try:
        overlap = int(row.get('overlap', 200))
    except Exception:
        overlap = 200

    coll_name = sanitize_name(base_url) if base_url else sanitize_name(url)
    coll_path = os.path.join(CHROMA_DIR, coll_name)

    print(f"\n=== Processing base_url={base_url} -> collection={coll_name} ===")
    print(f"Crawl: url={url} headless={headless} max_depth={max_depth} num_urls={num_urls}")

    if dry_run:
        print("Dry run; skipping crawl and persist")
        return

    # Crawl pages
    pages = []
    try:
        crawler = Crawler(headless=headless, timeout=60000, max_pages=num_urls)
        for itm in crawler.crawl_generator(start_url=url, max_depth=max_depth):
            text = itm.get('content') or ''
            if not text or not str(text).strip():
                continue
            pages.append({'url': itm.get('url'), 'content': text})
    except Exception as e:
        print(f"Crawler failed: {e}")

    if not pages:
        print("No pages crawled — skipping collection")
        return

    # Optional semantic filter on pages
    if crawl_matching_query and pages:
        try:
            emb = OpenAIEmbeddings()
            texts = [p['content'] for p in pages]
            doc_embs = emb.embed_documents(texts)
            q_emb = emb.embed(crawl_matching_query)
            _doc = np.array(doc_embs)
            _q = np.array(q_emb)
            dot = _doc.dot(_q)
            norms = (np.linalg.norm(_doc, axis=1) * np.linalg.norm(_q)) + 1e-12
            sims = dot / norms
            keep_idx = [i for i, s in enumerate(sims) if s >= relevance_threshold]
            pages = [pages[i] for i in keep_idx]
            print(f"After semantic filter: kept {len(pages)} pages")
        except Exception as e:
            print(f"Semantic filtering failed: {e}")

    # Chunk pages
    chunks = []
    for p in pages:
        for idx, c in chunk_text(p['content'], chunk_size=chunk_size, overlap=overlap):
            chunks.append({'text': c, 'source': p['url'], 'chunk_index': idx})

    if not chunks:
        print("No chunks produced; nothing to persist")
        return

    print(f"Produced {len(chunks)} chunks — creating documents and embeddings")

    # Create langchain Documents
    docs = []
    for c in chunks:
        meta = {'source': c['source'], 'chunk_index': c['chunk_index'], 'base_url': base_url, 'timestamp': datetime.utcnow().isoformat()}
        docs.append(Document(page_content=c['text'], metadata=meta))

    # Ensure CHROMA_DIR exists
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(coll_path, exist_ok=True)

    # Create embeddings and persist to Chroma
    try:
        emb_client = OpenAIEmbeddings()
    except Exception as e:
        print(f"Failed to init embeddings client: {e}")
        return

    try:
        vect = Chroma.from_documents(documents=docs, embedding=emb_client, persist_directory=coll_path, collection_name=coll_name)
        # Some Chroma variants require explicit persist
        if hasattr(vect, 'persist'):
            try:
                vect.persist()
            except Exception:
                pass
        print(f"Persisted collection '{coll_name}' with {len(docs)} documents at {coll_path}")
    except Exception as e:
        print(f"Failed to persist Chroma collection: {e}")


def main(csv_path, dry_run=False):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        print("No rows found in CSV")
        return
    for row in rows:
        process_row(row, dry_run=dry_run)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', '-c', default='params_template.csv', help='CSV file with pipeline params')
    ap.add_argument('--dry-run', action='store_true', help='Parse CSV but do not persist')
    args = ap.parse_args()
    main(args.csv, dry_run=args.dry_run)
