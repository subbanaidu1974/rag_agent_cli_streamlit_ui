#!/usr/bin/env python3
"""
CLI RAG Agent: query a Chroma collection and synthesize an answer via LangChain ChatOpenAI.

Usage:
    python rag_agent_cli.py --coll-dir ./chroma_store/<collection> --query "your question" --k 5
    or
    python rag_agent_cli.py --coll-dir ./chroma_store --coll-name <collection> --query "..."

Requires: langchain, openai, python-dotenv
"""
import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def load_backup(coll_path):
    p = Path(coll_path) / 'backup_stored_docs.jsonl'
    if not p.exists():
        return None
    out = []
    try:
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return None
    return out


def load_vectordb(coll_dir, coll_name=None):
    embeddings = None
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        embeddings = None
    # Try to load Chroma
    try:
        if coll_name:
            vect = Chroma(persist_directory=str(coll_dir), collection_name=coll_name, embedding_function=embeddings)
        else:
            vect = Chroma(persist_directory=str(coll_dir), embedding_function=embeddings)
        return vect
    except Exception:
        # try without collection_name
        try:
            vect = Chroma(persist_directory=str(coll_dir))
            return vect
        except Exception:
            return None


def retrieve_from_vect(vect, q, k=5):
    results = []
    try:
        if hasattr(vect, 'similarity_search_with_score'):
            hits = vect.similarity_search_with_score(q, k=k)
            for doc, score in hits:
                text = getattr(doc, 'page_content', doc) if doc is not None else ''
                meta = getattr(doc, 'metadata', {}) if doc is not None else {}
                results.append({'document': text, 'score': float(score) if score is not None else None, 'metadata': meta})
            return results
        elif hasattr(vect, 'similarity_search'):
            hits = vect.similarity_search(q, k=k)
            for doc in hits:
                text = getattr(doc, 'page_content', doc) if doc is not None else ''
                meta = getattr(doc, 'metadata', {}) if doc is not None else {}
                results.append({'document': text, 'score': None, 'metadata': meta})
            return results
        else:
            col = getattr(vect, '_collection', None)
            if col is not None and hasattr(col, 'query'):
                items = col.query(query_texts=[q], n_results=k)
                docs = items.get('documents', [[]])[0]
                metas = items.get('metadatas', [[]])[0]
                dists = items.get('distances', [[]])[0] if items.get('distances') is not None else (items.get('scores', [[]])[0] if items.get('scores') is not None else None)
                for i, doc in enumerate(docs):
                    sc = dists[i] if dists and i < len(dists) else None
                    results.append({'document': doc, 'score': float(sc) if sc is not None else None, 'metadata': metas[i] if i < len(metas) else {}})
                return results
    except Exception:
        return []
    return []


def synthesize_answer(docs, q, model_name='gpt-4o-mini', temperature=0.0):
    # Build context
    context_parts = []
    for i, d in enumerate(docs):
        txt = d.get('document') or ''
        meta = d.get('metadata') or {}
        score = d.get('score')
        header = f"[Source {i+1}] score={score} metadata={meta}"
        context_parts.append(f"{header}\n{txt}")
    context_text = "\n\n---\n\n".join(context_parts)

    system_prompt = "You are an assistant that answers user queries using the provided context. Use the context to answer concisely and cite sources by number (e.g. [Source 1])."
    user_prompt = f"Context:\n{context_text}\n\nUser Query: {q}\n\nProvide a brief answer and list sources used."

    try:
        try:
            chat = ChatOpenAI(model_name=model_name, temperature=temperature)
        except TypeError:
            chat = ChatOpenAI(model=model_name, temperature=temperature)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        # Try several invocation patterns
        resp = None
        if hasattr(chat, 'predict_messages'):
            resp = chat.predict_messages(messages)
        elif callable(chat):
            resp = chat(messages)
        elif hasattr(chat, 'generate'):
            try:
                resp = chat.generate(messages=[messages])
            except Exception:
                resp = chat.generate([messages])
        elif hasattr(chat, 'predict'):
            try:
                resp = chat.predict(messages=messages)
            except Exception:
                resp = chat.predict(str(user_prompt))
        else:
            raise RuntimeError('No supported invocation on ChatOpenAI')

        # Extract text
        answer = None
        if resp is not None:
            if hasattr(resp, 'content'):
                answer = resp.content
            else:
                gen = getattr(resp, 'generations', None)
                if gen and isinstance(gen, (list, tuple)) and len(gen) > 0:
                    try:
                        first = gen[0]
                        if isinstance(first, (list, tuple)) and len(first) > 0:
                            cand = first[0]
                            answer = getattr(cand, 'text', None) or getattr(cand, 'generation_text', None)
                        else:
                            answer = getattr(first, 'text', None) or getattr(first, 'generation_text', None)
                    except Exception:
                        answer = None
                else:
                    try:
                        answer = str(resp)
                    except Exception:
                        answer = None
        return answer
    except Exception as e:
        raise e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coll-dir', required=True, help='Path to collection directory (persist_directory)')
    ap.add_argument('--coll-name', help='Optional collection name inside the directory')
    ap.add_argument('--query', '-q', required=True, help='Query to run')
    ap.add_argument('--k', type=int, default=5, help='Top K documents to retrieve')
    ap.add_argument('--model', default='gpt-4o-mini', help='LLM model name')
    ap.add_argument('--temperature', type=float, default=0.0, help='LLM temperature')
    args = ap.parse_args()

    coll_dir = Path(args.coll_dir)
    if not coll_dir.exists():
        print(f'Collection directory not found: {coll_dir}')
        return

    vect = load_vectordb(coll_dir, coll_name=args.coll_name)
    results = []
    if vect is not None:
        results = retrieve_from_vect(vect, args.query, k=args.k)

    if not results:
        # try backup JSONL
        backup = load_backup(str(coll_dir if args.coll_name is None else coll_dir / args.coll_name))
        if backup:
            # simple top-k slice
            results = backup[: args.k]

    if not results:
        print('No documents found for this collection or query')
        return

    ans = synthesize_answer(results, args.query, model_name=args.model, temperature=args.temperature)
    if ans:
        print('\n--- LLM Response ---\n')
        print(ans)
        print('\n--------------------\n')
    else:
        print('LLM did not return an answer')


if __name__ == '__main__':
    main()
