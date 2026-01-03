import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from crawler import Crawler
import re
import os
# Numeric / similarity libraries
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load .env
from dotenv import load_dotenv
load_dotenv()
dotenv_loaded = True

# Chroma + embeddings / LLM clients (direct imports only)
from langchain_openai import OpenAIEmbeddings

import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from langchain_chroma import Chroma



CHROMA_DIR = "./chroma_store"
import pandas as pd

# Helper wrappers to construct/load Chroma across versions
def create_chroma_from_documents(documents, embeddings, coll_path, coll_name):
    if Chroma is None:
        return None
    vect = None
    try:
        try:
            vect = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=coll_path, collection_name=coll_name)
        except Exception:
            try:
                vect = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=coll_path)
            except Exception:
                try:
                    vect = Chroma(documents=documents, embedding_function=embeddings, persist_directory=coll_path, collection_name=coll_name)
                except Exception:
                    vect = None
    except Exception:
        vect = None
    return vect

def load_chroma_collection(coll_path, coll_name, embeddings):
    if Chroma is None:
        return None
    vect = None
    try:
        try:
            vect = Chroma(persist_directory=coll_path, collection_name=coll_name, embedding_function=embeddings)
        except Exception:
            try:
                vect = Chroma(persist_directory=coll_path, embedding_function=embeddings)
            except Exception:
                try:
                    vect = Chroma(persist_directory=coll_path)
                except Exception:
                    vect = None
    except Exception:
        vect = None
    return vect

def load_backup_stored_docs(coll_name):
    """Load the fallback JSONL backup written during persistence, if available."""
    try:
        p = os.path.join(CHROMA_DIR, coll_name, 'backup_stored_docs.jsonl')
        if not os.path.exists(p):
            return None
        import json
        out = []
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out
    except Exception:
        return None
# Page config
st.set_page_config(page_title="Document Processing Pipeline", layout="wide")

# Initialize session state
if 'active_step' not in st.session_state:
    st.session_state.active_step = 0
if 'crawled_content' not in st.session_state:
    st.session_state.crawled_content = ''
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'stored_docs' not in st.session_state:
    st.session_state.stored_docs = []
if 'url' not in st.session_state:
    st.session_state.url = ''
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 2000
if 'overlap' not in st.session_state:
    st.session_state.overlap = 200
if 'matching_query' not in st.session_state:
    st.session_state.matching_query = ''
st.markdown("""
<style>
    .main {
        background: linear-gradient(to bottom right, #f8fafc, #f1f5f9);
    }
    .step-complete {
        background-color: #10b981;
        color: white;
        padding: 20px;
        border-radius: 50%;
        text-align: center;
        font-size: 24px;
        margin: 0 auto;
        width: 64px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .step-active {
        background-color: #3b82f6;
        color: white;
        padding: 20px;
        border-radius: 50%;
        text-align: center;
        font-size: 24px;
        margin: 0 auto;
        width: 64px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: center;
        transform: scale(1.1);
    }
    .step-inactive {
        background-color: #e2e8f0;
        color: #94a3b8;
        padding: 20px;
        border-radius: 50%;
        text-align: center;
        font-size: 24px;
        margin: 0 auto;
        width: 64px;
        height: 64px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .content-box {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .info-box {
        background-color: #dbeafe;
        border: 1px solid #93c5fd;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .chunk-box {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .chunk-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 12px;
        max-height: 520px;
        overflow-y: auto;
        padding: 8px;
        border: 1px solid #e6eef8;
        border-radius: 8px;
        background: #ffffff;
    }
    .chunk-item {
        background: #ffffff;
        border: 1px solid #eef2ff;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 1px 2px rgba(2,6,23,0.04);
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    .chunk-item .meta { color: #64748b; font-size: 12px; }
    .chunk-item .text { color: #0f172a; font-size: 14px; white-space: pre-wrap; overflow: hidden; }
    /* Table view for chunks */
    .chunk-table-wrapper { max-height: 520px; overflow: auto; border: 1px solid #e6eef8; border-radius: 8px; }
    table.chunk-table { width: 100%; border-collapse: collapse; }
    table.chunk-table th, table.chunk-table td { padding: 10px; border-bottom: 1px solid #eef2ff; text-align: left; vertical-align: top; }
    table.chunk-table th { position: sticky; top: 0; background: #f8fafc; z-index: 2; font-weight: 600; }
    table.chunk-table td a { color: #2563eb; text-decoration: none; }
    .success-box {
        background-color: #d1fae5;
        border: 1px solid #6ee7b7;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    /* Button color overrides for specific buttons */
    /* Try multiple attribute selectors to catch Streamlit's button rendering variations */
    button[aria-label*="Start Crawling"],
    button[aria-label*="Start"],
    button[title*="Start Crawling"],
    button[title*="Start"]
    {
        background-color: #06b6d4 !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        border: none !important;
    }
    button[aria-label*="Stop Crawl"],
    button[aria-label*="Stop"],
    button[title*="Stop Crawl"],
    button[title*="Stop"]
    {
        background-color: #ef4444 !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        border: none !important;
    }
    /* Fallback: style the first two buttons inside the first wide column area (best-effort)
       This may affect other buttons but increases chance of styling the targeted buttons */
    .stApp .main > div[role="main"] div[data-testid="stVerticalBlock"] button:nth-of-type(1),
    .stApp .main > div[role="main"] div[data-testid="stVerticalBlock"] button:nth-of-type(2) {
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Chroma connector (moved from Step 5) ---
with st.sidebar:
    st.header("Chroma DB")
    # Toggle between pipeline UI and collections viewer
    mode_default = st.session_state.get('sidebar_mode', 'Crawl Pipeline')
    mode = st.radio("Mode", ["Crawl Pipeline", "Collections", "RAG Agent"], index=0 if mode_default == 'Crawl Pipeline' else (1 if mode_default == 'Collections' else 2), key="sidebar_mode_radio")
    st.session_state['sidebar_mode'] = mode
    if Chroma is None:
        st.info("Chroma vector store not available. Install langchain_community or langchain-chroma.")
    else:
        st.info("Manage and load persisted collections from the 'Collections' mode (center pane).")

# If the sidebar is set to show Collections, render the collections viewer in the center
mode = st.session_state.get('sidebar_mode', 'Crawl Pipeline')
if mode == 'Collections':
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("Collections Viewer")
    if Chroma is None:
        st.info("Chroma vector store not available. Install langchain_community or langchain-chroma to enable collection inspection.")
    else:
        # allow user to refresh the list of collections
        if st.button("🔄 Refresh collections", key="collections_refresh_btn"):
            st.session_state.pop('collections_list', None)
            try:
                st.experimental_rerun()
            except Exception:
                pass

        collections = st.session_state.get('collections_list')
        if collections is None:
            try:
                collections = []
                if os.path.exists(CHROMA_DIR):
                    for name in sorted(os.listdir(CHROMA_DIR)):
                        p = os.path.join(CHROMA_DIR, name)
                        if os.path.isdir(p):
                            collections.append(name)
            except Exception:
                collections = []
            st.session_state['collections_list'] = collections

        if not collections:
            st.info(f"No persisted Chroma collections found under {CHROMA_DIR}")
        else:
            sel = st.selectbox("Choose collection to inspect", options=collections, key="collections_view_select")
            if sel:
                coll_path = os.path.join(CHROMA_DIR, sel)
                embeddings = None
                if OpenAIEmbeddings is not None:
                    try:
                        embeddings = OpenAIEmbeddings()
                    except Exception:
                        embeddings = None
                vect = None
                try:
                    vect = Chroma(persist_directory=coll_path, collection_name=sel, embedding_function=embeddings)
                except Exception:
                    try:
                        vect = Chroma(persist_directory=coll_path, embedding_function=embeddings)
                    except Exception as e:
                        st.error(f"Failed to load Chroma collection: {e}")

                if vect is not None:
                    st.session_state['chroma_conn'] = vect
                    st.session_state['chroma_selected'] = sel
                    st.info(f"Loaded collection: {sel}")
                    items = None
                    try:
                        if hasattr(vect, 'get'):
                            items = vect.get()
                    except Exception:
                        items = None
                    if items is None:
                        try:
                            col = getattr(vect, '_collection', None)
                            if col is not None and hasattr(col, 'get'):
                                items = col.get()
                        except Exception:
                            items = None

                    if items is None:
                        st.info('Could not enumerate items from this Chroma version. View stored documents from pipeline state below.')
                    else:
                        try:
                            docs = items.get('documents', []) or []
                            metas = items.get('metadatas', []) or []
                            # If Chroma returned no documents or returned the same document repeated,
                            # prefer the JSONL backup produced during persistence (if present).
                            try:
                                docs_strs = [str(d) for d in docs]
                            except Exception:
                                docs_strs = []
                            if (not docs_strs) or (len(docs_strs) > 1 and len(set(docs_strs)) == 1):
                                backup = load_backup_stored_docs(sel)
                                if backup:
                                    docs = [b.get('document') for b in backup]
                                    metas = [b.get('metadata') or b.get('metadata', {}) for b in backup]
                            rows = []
                            for i, d in enumerate(docs):
                                meta = metas[i] if i < len(metas) else {}
                                try:
                                    content_len = len(d) if isinstance(d, str) else 0
                                except Exception:
                                    content_len = 0
                                doc_preview = (d[:300] + '...') if isinstance(d, str) and len(d) > 300 else d
                                rows.append({'id': f'{sel}_{i}', 'content_length': content_len, 'document': doc_preview, 'metadata': meta})
                            if rows:
                                st.subheader(f'Contents of {sel}')
                                df = pd.DataFrame(rows)
                                st.dataframe(df, height=400)
                            else:
                                st.info('Collection appears empty')
                        except Exception:
                            st.info('Unable to render collection contents from this Chroma client.')

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# RAG Agent mode - center panel query against Chroma collections
if mode == 'RAG Agent':
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("RAG Agent")
    st.write("Run a semantic query against persisted Chroma collections.")

    # List available collections from disk
    try:
        collections = []
        if os.path.exists(CHROMA_DIR):
            for name in sorted(os.listdir(CHROMA_DIR)):
                p = os.path.join(CHROMA_DIR, name)
                if os.path.isdir(p):
                    collections.append(name)
    except Exception:
        collections = []

    sel = st.selectbox("Choose collection", options=collections if collections else ["(none)"], key="rag_center_select")
    if sel == "(none)":
        sel = None

    k = st.number_input("Top K results", min_value=1, max_value=100, value=5, key="rag_k")
    q = st.text_input("Query", key="rag_query")

    if st.button("Run RAG", key="run_rag_btn_center"):
        if not q:
            st.error("Enter a query to run the RAG agent")
        else:
            vect = None
            # try to use an already-loaded connection
            if st.session_state.get('chroma_conn') is not None and st.session_state.get('chroma_selected') == sel:
                vect = st.session_state.get('chroma_conn')

            # otherwise attempt to load the collection
            if vect is None and sel is not None:
                embeddings = None
                if OpenAIEmbeddings is not None:
                    try:
                        embeddings = OpenAIEmbeddings()
                    except Exception:
                        embeddings = None
                vect = load_chroma_collection(os.path.join(CHROMA_DIR, sel), sel, embeddings)

            results = []
            try:
                if vect is not None:
                    # Preferred: similarity_search_with_score
                    try:
                        if hasattr(vect, 'similarity_search_with_score'):
                            hits = vect.similarity_search_with_score(q, k=k)
                            for doc, score in hits:
                                text = getattr(doc, 'page_content', doc) if doc is not None else ''
                                meta = getattr(doc, 'metadata', {}) if doc is not None else {}
                                results.append({'document': text, 'score': float(score) if score is not None else None, 'metadata': meta})
                        elif hasattr(vect, 'similarity_search'):
                            hits = vect.similarity_search(q, k=k)
                            for doc in hits:
                                text = getattr(doc, 'page_content', doc) if doc is not None else ''
                                meta = getattr(doc, 'metadata', {}) if doc is not None else {}
                                results.append({'document': text, 'score': None, 'metadata': meta})
                        else:
                            # Try low-level collection query (different Chroma wrappers expose different APIs)
                            col = getattr(vect, '_collection', None)
                            if col is not None and hasattr(col, 'query'):
                                try:
                                    items = col.query(query_texts=[q], n_results=k)
                                    docs = items.get('documents', [[]])[0]
                                    metas = items.get('metadatas', [[]])[0]
                                    dists = items.get('distances', [[]])[0] if items.get('distances') is not None else (items.get('scores', [[]])[0] if items.get('scores') is not None else None)
                                    for i, doc in enumerate(docs):
                                        sc = dists[i] if dists and i < len(dists) else None
                                        results.append({'document': doc, 'score': float(sc) if sc is not None else None, 'metadata': metas[i] if i < len(metas) else {}})
                                except Exception:
                                    pass
                    except Exception as e:
                        st.warning(f"Chroma client query raised: {e}")

                # Fallback: use backup JSONL with stored embeddings
                if not results and sel is not None:
                    backup = load_backup_stored_docs(sel)
                    if backup and OpenAIEmbeddings is not None:
                        emb_client = None
                        try:
                            emb_client = OpenAIEmbeddings()
                        except Exception:
                            emb_client = None
                        if emb_client is not None:
                            try:
                                q_emb = None
                                try:
                                    q_emb = emb_client.embed(q)
                                except Exception:
                                    try:
                                        q_emb = emb_client.embed_documents([q])
                                        if isinstance(q_emb, list):
                                            q_emb = q_emb[0]
                                    except Exception:
                                        q_emb = None

                                if q_emb is not None:
                                    import numpy as _np
                                    scored = []
                                    for item in backup:
                                        doc_text = item.get('document', '')
                                        doc_emb = item.get('embedding')
                                        meta = item.get('metadata', {})
                                        if doc_emb is None:
                                            scored.append({'document': doc_text, 'score': None, 'metadata': meta})
                                        else:
                                            try:
                                                _d = _np.array(doc_emb)
                                                _q = _np.array(q_emb)
                                                sc = float((_d.dot(_q)) / (_np.linalg.norm(_d) * _np.linalg.norm(_q) + 1e-12))
                                            except Exception:
                                                sc = 0.0
                                            scored.append({'document': doc_text, 'score': sc, 'metadata': meta})
                                    # sort by score if present
                                    with_score = [s for s in scored if s.get('score') is not None]
                                    if with_score:
                                        with_score.sort(key=lambda x: x['score'], reverse=True)
                                        results = with_score[:k]
                                    else:
                                        results = scored[:k]
                            except Exception:
                                pass

                # Display results by generating an LLM answer using the retrieved docs as context
                if results:
                    # Build context from top-k retrieved docs
                    docs_for_prompt = results[:int(k)] if k and isinstance(k, int) else results
                    context_parts = []
                    for i, d in enumerate(docs_for_prompt):
                        txt = d.get('document') or ''
                        meta = d.get('metadata') or {}
                        score = d.get('score')
                        header = f"[Source {i+1}] score={score} metadata={meta}"
                        context_parts.append(f"{header}\n{txt}")
                    context_text = "\n\n---\n\n".join(context_parts)

                    # Use LangChain ChatOpenAI to synthesize an answer from retrieved docs
                    if ChatOpenAI is None or HumanMessage is None or SystemMessage is None:
                        st.error('LangChain ChatOpenAI or message schema not available; install langchain.')
                    else:
                        # ChatOpenAI will pick up OPENAI_API_KEY from environment
                        api_key = os.getenv('OPENAI_API_KEY')
                        if not api_key:
                            st.error('OPENAI_API_KEY not set in environment; set it to enable LLM responses.')
                        else:
                            system_prompt = "You are an assistant that answers user queries using the provided context. Use the context to answer concisely and cite sources by number (e.g. [Source 1])."
                            user_prompt = f"Context:\n{context_text}\n\nUser Query: {q}\n\nProvide a brief answer and list sources used."

                            with st.spinner('Generating answer from LLM (LangChain ChatOpenAI)...'):
                                try:
                                    try:
                                        chat = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.0)
                                    except TypeError:
                                        chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)

                                    messages = [
                                        SystemMessage(content=system_prompt),
                                        HumanMessage(content=user_prompt)
                                    ]

                                    # Try multiple invocation patterns for different LangChain versions
                                    resp = None
                                    try:
                                        # Preferred: predict_messages (returns an AIMessage)
                                        if hasattr(chat, 'predict_messages'):
                                            resp = chat.predict_messages(messages)
                                        # Some versions make the instance callable
                                        elif callable(chat):
                                            resp = chat(messages)
                                        # Newer versions may expose `generate`
                                        elif hasattr(chat, 'generate'):
                                            try:
                                                resp = chat.generate(messages=[messages])
                                            except Exception:
                                                resp = chat.generate([messages])
                                        # Fallback: predict (may accept text or messages)
                                        elif hasattr(chat, 'predict'):
                                            try:
                                                resp = chat.predict(messages=messages)
                                            except Exception:
                                                resp = chat.predict(str(user_prompt))
                                        else:
                                            raise RuntimeError('No supported invocation on ChatOpenAI')
                                    except Exception as ex_call:
                                        raise ex_call

                                    # Extract text from common LangChain response shapes
                                    answer = None
                                    if resp is not None:
                                        # AIMessage-like (has .content)
                                        if hasattr(resp, 'content'):
                                            answer = resp.content
                                        else:
                                            # LLMResult-like (generations)
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
                                                # fallback to string
                                                try:
                                                    answer = str(resp)
                                                except Exception:
                                                    answer = None

                                    if answer:
                                        st.subheader('LLM Response')
                                        st.write(answer)
                                    else:
                                        st.error('LLM did not return an answer.')
                                except Exception as e:
                                    st.error(f'LLM call failed: {e}')
                else:
                    st.info('No results found for this query.')

            except Exception as e:
                st.error(f"RAG query failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Steps definition
steps = [
    {"name": "Crawl Website", "icon": "🌐", "color": "#3b82f6"},
    {"name": "Load Documents", "icon": "📄", "color": "#10b981"},
    {"name": "Split into Chunks", "icon": "✂️", "color": "#8b5cf6"},
    {"name": "Persist to ChromaDB", "icon": "💾", "color": "#f97316"},
    {"name": "View Stored", "icon": "👁️", "color": "#ec4899"}
]

# Progress Steps
cols = st.columns(5)
for idx, step in enumerate(steps):
    with cols[idx]:
        # show visual indicator for current/complete state (single badge, no duplicate button)
        if idx < st.session_state.active_step:
            st.markdown(f'<div style="text-align:center; margin-top:6px;"><span class="step-complete">{step["icon"]}</span><div style="font-size:12px;">{step["name"]}</div></div>', unsafe_allow_html=True)
        elif idx == st.session_state.active_step:
            st.markdown(f'<div style="text-align:center; margin-top:6px;"><span class="step-active">{step["icon"]}</span><div style="font-size:12px;">{step["name"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="text-align:center; margin-top:6px;"><span class="step-inactive">{step["icon"]}</span><div style="font-size:12px;">{step["name"]}</div></div>', unsafe_allow_html=True)

# st.markdown("---")

# Step 1: Crawl Website
if st.session_state.active_step == 0:
    # st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("Step 1: Crawl Website")
    # place description and headless checkbox on the same row
    c_desc, c_headless = st.columns([3, 1])
    with c_desc:
        st.write("Enter a URL to simulate crawling web content")
    with c_headless:
        headless = st.checkbox("Run browser headless", value=True, key="headless_input")

    # URL input below the description
    url = st.text_input("URL", placeholder="https://example.com", value=st.session_state.url, key="url_input")
    st.session_state.url = url
    # Matching query input (optional)
    # matching = st.text_input("Matching Query (optional)", placeholder="Enter a query to match content", value=st.session_state.matching_query, key="matching_query_input")
    # st.session_state.matching_query = matching

    # Numeric crawl controls on the next row
    c1, c2 = st.columns([1, 1])
    with c1:
        max_depth = st.number_input("Max depth", min_value=0, max_value=10, value=2, key="max_depth_input")
    with c2:
        num_urls = st.number_input("Number of URLs to crawl", min_value=1, max_value=10000, value=5, key="num_urls_input")

    # Expose crawl matching query and threshold up-front so user can enter before crawling
    st.markdown('**Optional: pre-set a Crawl Matching Query**')
    crawl_mq_input = st.text_input('Crawl Matching Query (optional)', value=st.session_state.get('crawl_matching_query', ''), key='crawl_matching_query_input')
    st.session_state['crawl_matching_query'] = crawl_mq_input
    crawl_threshold_input = st.slider('Crawl Threshold', 0.0, 1.0, float(st.session_state.get('relevance_threshold', 0.75)), step=0.01, key='crawl_relevance_threshold_input')
    st.session_state['relevance_threshold'] = crawl_threshold_input

    # stop/crawling flags
    st.session_state.setdefault('stop_crawl', False)
    st.session_state.setdefault('crawling', False)

    # Start / Stop buttons left-aligned under Max depth
    btn_area, _ = st.columns([1, 3])
    with btn_area:
        b1, b2 = st.columns([1, 1])
        start_pressed = b1.button("🚀 Start Crawling", key="start_crawl_btn")
        stop_pressed = b2.button("Stop Crawl", key="stop_crawl_btn")

    if stop_pressed:
        st.session_state['stop_crawl'] = True

    if start_pressed:
        if not url:
            st.error("Please enter a URL to crawl")
        else:
            # reset state
            st.session_state['stop_crawl'] = False
            st.session_state['crawling'] = True
            st.session_state['crawled_docs'] = []

            rows = []
            table_ph = st.empty()
            progress_ph = st.empty()

            table_ph = st.empty()
            # helper to render rows into a pandas DataFrame and show as table
            def render_table(rows_list):
                # filter out rows that have empty or whitespace-only content
                filtered = [r for r in rows_list if r.get('content') and str(r.get('content')).strip() != '']
                if not filtered:
                    table_ph.write("No results yet")
                    return
                df = pd.DataFrame(filtered)
                # limit preview length
                if 'content' in df.columns:
                    df['content'] = df['content'].astype(str).str.slice(0, 1000)
                # add/format content length column and present columns in a helpful order
                if 'content_length' in df.columns:
                    try:
                        df = df[['url', 'type', 'depth', 'content_length', 'content']]
                    except Exception:
                        pass
                table_ph.dataframe(df, height=420)
            with st.spinner("Crawling website (streaming results)..."):
                try:
                    crawler = Crawler(headless=headless, timeout=60000, max_pages=int(num_urls))
                    for itm in crawler.crawl_generator(start_url=url, max_depth=int(max_depth)):
                        # allow user to request stop
                        if st.session_state.get('stop_crawl'):
                            progress_ph.markdown("**Crawl stopped by user**")
                            break

                        u = itm.get('url')
                        typ = itm.get('type', 'html')
                        depth_v = itm.get('depth', 0)
                        raw_content = itm.get('content', '') or ''
                        preview = raw_content[:1000].replace('\n', ' ') if raw_content else ''
                        content_len = len(raw_content) if raw_content else 0
                        # store both a preview for UI and the full content for loading/chunking
                        doc = {'url': u, 'type': typ, 'depth': depth_v, 'content': preview, 'full_content': raw_content, 'content_length': content_len}
                        st.session_state.setdefault('crawled_docs', []).append(doc)
                        rows.append(doc)

                        progress_ph.markdown(f"**Crawled:** {len(rows)}")
                        try:
                            render_table(rows)
                        except Exception:
                            table_ph.write(rows)
                except Exception as e:
                    st.error(f"Crawl failed: {e}")

            st.session_state['crawling'] = False
            # set preview
            if st.session_state.get('crawled_docs'):
                st.session_state.crawled_content = st.session_state['crawled_docs'][0].get('content', '')

            # After crawl completes, if a crawl matching query was provided, automatically apply
            # the semantic filter so subsequent Load step sees only matched pages.
            crawl_mq = st.session_state.get('crawl_matching_query', '') or ''
            crawl_threshold = float(st.session_state.get('relevance_threshold', 0.75))
            if crawl_mq:
                crawled = st.session_state.get('crawled_docs', [])
                texts = [d.get('full_content') or d.get('content') or '' for d in crawled]
                if any(texts) and OpenAIEmbeddings is not None:
                    emb_client = None
                    try:
                        emb_client = OpenAIEmbeddings()
                    except Exception as e:
                        st.session_state['emb_init_error'] = str(e)
                        emb_client = None

                    if emb_client is not None:
                        doc_embs = None
                        try:
                            doc_embs = emb_client.embed_documents(texts)
                        except Exception:
                            try:
                                doc_embs = emb_client.embed(texts)
                            except Exception:
                                doc_embs = None

                        q_emb = None
                        try:
                            q_emb = emb_client.embed_documents([crawl_mq])
                            if isinstance(q_emb, list) and q_emb:
                                q_emb = q_emb[0]
                        except Exception:
                            try:
                                q_emb = emb_client.embed(crawl_mq)
                            except Exception:
                                q_emb = None

                        if doc_embs is not None and q_emb is not None:
                            scores = []
                            try:
                                import numpy as _np
                                _doc = _np.array(doc_embs)
                                _q = _np.array(q_emb)
                                if cosine_similarity is not None:
                                    sims = cosine_similarity([_q], _doc)[0]
                                else:
                                    dot = _doc.dot(_q)
                                    norms = (_np.linalg.norm(_doc, axis=1) * _np.linalg.norm(_q))
                                    sims = dot / (norms + 1e-12)
                                scores = [float(s) for s in sims]
                            except Exception:
                                def _dot(a, b):
                                    return sum(x * y for x, y in zip(a, b))
                                def _norm(a):
                                    return sum(x * x for x in a) ** 0.5
                                for emb in doc_embs:
                                    try:
                                        s = _dot(emb, q_emb) / ((_norm(emb) * _norm(q_emb)) + 1e-12)
                                    except Exception:
                                        s = 0.0
                                    scores.append(float(s))

                            keep_idx = [i for i, sc in enumerate(scores) if sc >= float(crawl_threshold)]
                            st.session_state['crawled_keep_idx'] = keep_idx
                            st.session_state['crawled_preview_scores'] = scores
            else:
                # no matching query provided — clear any previous filters
                if 'crawled_keep_idx' in st.session_state:
                    del st.session_state['crawled_keep_idx']
                if 'crawled_preview_scores' in st.session_state:
                    del st.session_state['crawled_preview_scores']

            # advance to Load step (filtered state already set)
            st.session_state.active_step = 1
            try:
                st.rerun()
            except Exception:
                pass
    st.markdown('</div>', unsafe_allow_html=True)

    # Crawl-stage semantic filter UI (filter by matching query over full_content)
    crawled = st.session_state.get('crawled_docs', [])
    if crawled:
        st.markdown('---')
        st.write('Optional: Filter crawled documents by semantic query (operates on full page content)')
        cq_col1, cq_col2 = st.columns([3, 1])
        with cq_col1:
            crawl_mq = st.text_input('Crawl Matching Query', value=st.session_state.get('crawl_matching_query', ''), key='crawl_matching_query')
            st.session_state['crawl_matching_query'] = crawl_mq
        with cq_col2:
            crawl_threshold = st.slider('Threshold', 0.0, 1.0, float(st.session_state.get('relevance_threshold', 0.75)), step=0.01, key='crawl_relevance_threshold')
            st.session_state['relevance_threshold'] = crawl_threshold

        if st.button('🔎 Filter Crawled Docs'):
            if not crawl_mq:
                st.error('Enter a query to filter crawled documents')
            else:
                # prepare texts (use full_content when available)
                texts = [d.get('full_content') or d.get('content') or '' for d in crawled]
                if not any(texts):
                    st.error('No crawled content available to filter')
                elif OpenAIEmbeddings is None:
                    st.error('Embedding client not available to perform semantic filtering.')
                else:
                    emb_client = None
                    try:
                        emb_client = OpenAIEmbeddings()
                    except Exception as e:
                        st.session_state['emb_init_error'] = str(e)
                        emb_client = None

                    if emb_client is None:
                        st.error('Failed to initialize embeddings client. See debug info.')
                    else:
                        # compute document embeddings
                        doc_embs = None
                        try:
                            doc_embs = emb_client.embed_documents(texts)
                        except Exception:
                            try:
                                doc_embs = emb_client.embed(texts)
                            except Exception:
                                doc_embs = None

                        # compute query embedding
                        q_emb = None
                        try:
                            q_emb = emb_client.embed_documents([crawl_mq])
                            if isinstance(q_emb, list) and q_emb:
                                q_emb = q_emb[0]
                        except Exception:
                            try:
                                q_emb = emb_client.embed(crawl_mq)
                            except Exception:
                                q_emb = None

                        if doc_embs is None or q_emb is None:
                            st.error('Failed to compute embeddings for query or documents')
                        else:
                            # compute similarity scores (numpy preferred)
                            scores = []
                            try:
                                import numpy as _np
                                _doc = _np.array(doc_embs)
                                _q = _np.array(q_emb)
                                if cosine_similarity is not None:
                                    sims = cosine_similarity([_q], _doc)[0]
                                else:
                                    dot = _doc.dot(_q)
                                    norms = (_np.linalg.norm(_doc, axis=1) * _np.linalg.norm(_q))
                                    sims = dot / (norms + 1e-12)
                                scores = [float(s) for s in sims]
                            except Exception:
                                # pure-python fallback
                                def _dot(a, b):
                                    return sum(x * y for x, y in zip(a, b))
                                def _norm(a):
                                    return sum(x * x for x in a) ** 0.5
                                for emb in doc_embs:
                                    try:
                                        s = _dot(emb, q_emb) / ((_norm(emb) * _norm(q_emb)) + 1e-12)
                                    except Exception:
                                        s = 0.0
                                    scores.append(float(s))

                            keep_idx = [i for i, sc in enumerate(scores) if sc >= float(crawl_threshold)]
                            st.session_state['crawled_keep_idx'] = keep_idx
                            st.session_state['crawled_preview_scores'] = scores
                            st.success(f'Filtered {len(keep_idx)} / {len(texts)} crawled documents (threshold {crawl_threshold})')
                            try:
                                st.experimental_rerun()
                            except Exception:
                                pass

        # Render two tables side-by-side: Live feed (all crawled with previews) and Matched (filtered) if present
        live_col, match_col = st.columns([2, 1])
        # Live feed
        try:
            with live_col:
                df_live = pd.DataFrame(crawled)
                if 'content' in df_live.columns:
                    df_live['content'] = df_live['content'].astype(str).str.slice(0, 1000)
                st.subheader('Crawled URLs (live feed)')
                st.dataframe(df_live, height=420)
        except Exception:
            with live_col:
                st.write(crawled)

        # Matched / filtered table
        if 'crawled_keep_idx' in st.session_state and st.session_state.get('crawled_keep_idx'):
            try:
                kidx = st.session_state.get('crawled_keep_idx')
                filtered_rows = [crawled[i] for i in kidx if i < len(crawled)]
                dfk = pd.DataFrame(filtered_rows)
                if 'content' in dfk.columns:
                    dfk['content'] = dfk['content'].astype(str).str.slice(0, 1000)
                scores_list = st.session_state.get('crawled_preview_scores', [])
                try:
                    dfk['score'] = [scores_list[i] if i < len(scores_list) else None for i in kidx]
                except Exception:
                    pass
                with match_col:
                    st.subheader('Matched URLs')
                    st.dataframe(dfk, height=420)
            except Exception:
                with match_col:
                    st.write('No matched rows')

# Step 2: Load Documents
elif st.session_state.active_step == 1:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("Step 2: Load Documents")
    st.write("Crawled content preview:")

    # show crawled items table first (live feed)
    crawled = st.session_state.get('crawled_docs', [])
    crawled_with_content = [d for d in crawled if d.get('content') and str(d.get('content')).strip() != '']
    if crawled_with_content:
        try:
            df_live = pd.DataFrame(crawled_with_content)
            if 'content' in df_live.columns:
                df_live['content'] = df_live['content'].astype(str).str.slice(0, 1000)
            st.subheader('Crawled URLs (live feed)')
            st.dataframe(df_live, height=300)
        except Exception:
            st.write(crawled_with_content)

    # Then show crawl matching controls (query, threshold, filter button) stacked vertically
    st.markdown('**Filter crawled pages by semantic query (operates on full page content)**')
    crawl_mq = st.text_input('Crawl Matching Query', value=st.session_state.get('crawl_matching_query', ''), key='crawl_matching_query_step2')
    st.session_state['crawl_matching_query'] = crawl_mq
    crawl_threshold = st.slider('Threshold', 0.0, 1.0, float(st.session_state.get('relevance_threshold', 0.75)), step=0.01, key='crawl_relevance_threshold_step2')
    st.session_state['relevance_threshold'] = crawl_threshold
    filter_btn = st.button('🔎 Filter Crawled Docs (Step 2)')

    if filter_btn:
        if not crawl_mq:
            st.error('Enter a query to filter crawled documents')
        else:
            texts = [d.get('full_content') or d.get('content') or '' for d in crawled]
            if not any(texts):
                st.error('No crawled content available to filter')
            elif OpenAIEmbeddings is None:
                st.error('Embedding client not available to perform semantic filtering.')
            else:
                emb_client = None
                try:
                    emb_client = OpenAIEmbeddings()
                except Exception as e:
                    st.session_state['emb_init_error'] = str(e)
                    emb_client = None

                if emb_client is None:
                    st.error('Failed to initialize embeddings client. See debug info.')
                else:
                    doc_embs = None
                    try:
                        doc_embs = emb_client.embed_documents(texts)
                    except Exception:
                        try:
                            doc_embs = emb_client.embed(texts)
                        except Exception:
                            doc_embs = None

                    q_emb = None
                    try:
                        q_emb = emb_client.embed_documents([crawl_mq])
                        if isinstance(q_emb, list) and q_emb:
                            q_emb = q_emb[0]
                    except Exception:
                        try:
                            q_emb = emb_client.embed(crawl_mq)
                        except Exception:
                            q_emb = None

                    if doc_embs is None or q_emb is None:
                        st.error('Failed to compute embeddings for query or documents')
                    else:
                        scores = []
                        try:
                            import numpy as _np
                            _doc = _np.array(doc_embs)
                            _q = _np.array(q_emb)
                            if cosine_similarity is not None:
                                sims = cosine_similarity([_q], _doc)[0]
                            else:
                                dot = _doc.dot(_q)
                                norms = (_np.linalg.norm(_doc, axis=1) * _np.linalg.norm(_q))
                                sims = dot / (norms + 1e-12)
                            scores = [float(s) for s in sims]
                        except Exception:
                            def _dot(a, b):
                                return sum(x * y for x, y in zip(a, b))
                            def _norm(a):
                                return sum(x * x for x in a) ** 0.5
                            for emb in doc_embs:
                                try:
                                    s = _dot(emb, q_emb) / ((_norm(emb) * _norm(q_emb)) + 1e-12)
                                except Exception:
                                    s = 0.0
                                scores.append(float(s))

                        keep_idx = [i for i, sc in enumerate(scores) if sc >= float(crawl_threshold)]
                        st.session_state['crawled_keep_idx'] = keep_idx
                        st.session_state['crawled_preview_scores'] = scores
                        st.success(f'Filtered {len(keep_idx)} / {len(texts)} crawled documents (threshold {crawl_threshold})')
                        try:
                            st.experimental_rerun()
                        except Exception:
                            pass

    # After controls, show Matched URLs table (if any)
    crawled_keep_idx = st.session_state.get('crawled_keep_idx') if 'crawled_keep_idx' in st.session_state else None
    if crawled_keep_idx:
        try:
            filtered = [crawled[i] for i in crawled_keep_idx if i < len(crawled)]
            df = pd.DataFrame(filtered)
            if 'content' in df.columns:
                df['content'] = df['content'].astype(str).str.slice(0, 1000)
            try:
                scores_list = st.session_state.get('crawled_preview_scores', [])
                df['score'] = [scores_list[i] if i < len(scores_list) else None for i in crawled_keep_idx]
            except Exception:
                pass
            st.subheader('Matched URLs')
            st.dataframe(df, height=300)
        except Exception:
            st.write('No matched results')
    # (Semantic filtering on crawled documents removed) 

    if st.button("📄 Load Documents", type="primary"):
        crawled = st.session_state.get('crawled_docs', [])
        docs = []
        crawled_keep_idx = st.session_state.get('crawled_keep_idx') if 'crawled_keep_idx' in st.session_state else None
        # if a filtered set exists, only load those documents
        if crawled_keep_idx:
            for i in crawled_keep_idx:
                if i < len(crawled):
                    d = crawled[i]
                    content = d.get('full_content') if d.get('full_content') is not None else (d.get('content', '') or '')
                    source = d.get('url', '') or ''
                    docs.append({
                        'id': f'doc_{i}',
                        'content': content,
                        'metadata': {
                            'source': source,
                            'type': d.get('type'),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
        else:
            for idx, d in enumerate(crawled):
                # prefer full_content stored by the crawler, fall back to preview
                content = d.get('full_content') if d.get('full_content') is not None else (d.get('content', '') or '')
                source = d.get('url', '') or ''
                docs.append({
                    'id': f'doc_{idx}',
                    'content': content,
                    'metadata': {
                        'source': source,
                        'type': d.get('type'),
                        'timestamp': datetime.now().isoformat()
                    }
                })

        if not docs:
            st.warning('No crawled documents to load. Run the Crawl step first.')

        else:
            st.success(f'Loaded {len(docs)} documents (matching query will be used later for semantic filtering)')

        st.session_state.documents = docs
        st.session_state.active_step = 2
        try:
            st.rerun()
        except Exception:
            pass
    st.markdown('</div>', unsafe_allow_html=True)

# Step 3: Split into Chunks
elif st.session_state.active_step == 2:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("Step 3: Split into Chunks")
    st.write("Configure chunking parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk Size", 100, 5000, st.session_state.chunk_size)
        st.session_state.chunk_size = chunk_size
    with col2:
        overlap = st.slider("Overlap", 0, 200, st.session_state.overlap)
        st.session_state.overlap = overlap
    
    if st.button("✂️ Create Chunks", type="primary"):
        all_chunks = []
        for doc in st.session_state.get('documents', []):
            text = doc.get('content', '')
            doc_meta = doc.get('metadata', {})
            i = 0
            chunk_index = 0
            while i < len(text):
                chunk_text = text[i:i + chunk_size]
                if chunk_text:
                    chunk_id = f"chunk_{len(all_chunks)+1}"
                    all_chunks.append({
                        'id': chunk_id,
                        'text': chunk_text,
                        'source': doc_meta.get('source', ''),
                        'chunk_index': chunk_index
                    })
                    chunk_index += 1
                i += chunk_size - overlap

        st.session_state.chunks = all_chunks
        st.session_state.active_step = 3
        try:
            st.rerun()
        except Exception:
            pass
    st.markdown('</div>', unsafe_allow_html=True)

# Step 4: Persist to ChromaDB
elif st.session_state.active_step == 3:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("Step 4: Persist to ChromaDB")
    st.write(f"Created {len(st.session_state.chunks)} chunks:")
    
    # Render chunks in a responsive scrollable grid
    chunks = st.session_state.get('chunks', [])
    if chunks:
        # render a scrollable table with headers: Chunk | URL | Content
        rows = []
        for idx, chunk in enumerate(chunks):
            chunk_preview = chunk['text'][:400] if len(chunk['text']) > 400 else chunk['text']
            safe_preview = str(chunk_preview).replace('<', '&lt;').replace('>', '&gt;')
            chunk_num = idx + 1
            source = chunk.get('source', '') or ''
            safe_source = str(source).replace('"', '%22')
            content_len = len(chunk.get('text', '') or '')
            rows.append((chunk_num, source, content_len, safe_preview))

        # build a simple HTML table for chunks
        table_html = ['<div class="chunk-table-wrapper">', '<table class="chunk-table">', '<thead><tr><th>Chunk</th><th>URL</th><th>Content Length</th><th>Content Preview</th></tr></thead>', '<tbody>']
        for r in rows:
            num, src, clen, txt = r
            if src:
                url_cell = f'<a href="{src}" target="_blank" rel="noreferrer">{src}</a>'
            else:
                url_cell = '<span style="color:#94a3b8">(no source)</span>'
            table_html.append(f'<tr><td>{num}</td><td>{url_cell}</td><td>{clen}</td><td>{txt}...</td></tr>')
        table_html.append('</tbody></table></div>')
        st.markdown('\n'.join(table_html), unsafe_allow_html=True)
    else:
        st.info('No chunks created yet — run Split into Chunks first')
    # Persist: simple single-button persist to DB (no semantic filtering UI)
    if st.button("💾 Persist to DB", type="primary"):
        # persist all chunks to Chroma (or JSONL fallback)
        stored = []
        base_url = st.session_state.get('url') or ''
        coll_name = re.sub(r'[^0-9a-zA-Z_-]', '_', base_url)[:64] or 'chroma_index'
        chunks = st.session_state.get('chunks', [])
        for idx, chunk in enumerate(chunks):
            meta = {
                'source': chunk.get('source'),
                'chunk_index': chunk.get('chunk_index'),
                'total_chunks': len(chunks),
                'base_url': base_url,
                'index': coll_name,
                'matched': True,
            }
            stored.append({
                'id': chunk['id'],
                'document': chunk['text'],
                'metadata': meta,
            })

        # compute embeddings if available
        try:
            if OpenAIEmbeddings is not None and len(stored) > 0:
                try:
                    emb_client = OpenAIEmbeddings()
                except Exception:
                    emb_client = None
                if emb_client is not None:
                    texts_for_emb = [s['document'] for s in stored]
                    embs = None
                    try:
                        embs = emb_client.embed_documents(texts_for_emb)
                    except Exception:
                        try:
                            embs = emb_client.embed(texts_for_emb)
                        except Exception:
                            embs = None
                    if embs is not None and len(embs) == len(stored):
                        for k, e in enumerate(embs):
                            stored[k]['embedding'] = e
        except Exception:
            pass

        # persist
        try:
            os.makedirs(CHROMA_DIR, exist_ok=True)
        except Exception:
            pass
        persisted = False
        if Chroma is not None:
            embeddings = None
            if OpenAIEmbeddings is not None:
                try:
                    embeddings = OpenAIEmbeddings()
                except Exception:
                    embeddings = None
            doc_objs = []
            for s in stored:
                if Document is not None:
                    doc_objs.append(Document(page_content=s['document'], metadata=s['metadata']))
                else:
                    class _D:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata
                    doc_objs.append(_D(page_content=s['document'], metadata=s['metadata']))
            coll_path = os.path.join(CHROMA_DIR, coll_name)
            vectordb = create_chroma_from_documents(doc_objs, embeddings, coll_path, coll_name)
            if vectordb is not None:
                st.session_state.stored_docs = stored
                st.session_state.vectordb = vectordb
                st.success(f"Persisted {len(stored)} chunks to Chroma collection '{coll_name}'")
                try:
                    df_full = pd.DataFrame([{'id': s['id'], 'document': s['document'], 'metadata': s['metadata']} for s in stored])
                    st.subheader('Persisted Chunks (full content)')
                    st.dataframe(df_full, height=400)
                except Exception:
                    st.write('Persisted chunks (could not render table)')
                persisted = True
        if not persisted:
            try:
                coll_path = os.path.join(CHROMA_DIR, coll_name)
                os.makedirs(coll_path, exist_ok=True)
                backup_path = os.path.join(coll_path, 'backup_stored_docs.jsonl')
                import json
                with open(backup_path, 'w', encoding='utf-8') as f:
                    for s in stored:
                        f.write(json.dumps(s, ensure_ascii=False) + '\n')
                st.session_state.stored_docs = stored
                st.success(f"Saved {len(stored)} chunks to {backup_path} (backup - Chroma persist not available)")
                try:
                    df_full = pd.DataFrame([{'id': s['id'], 'document': s['document'], 'metadata': s['metadata']} for s in stored])
                    st.subheader('Persisted Chunks (backup - full content)')
                    st.dataframe(df_full, height=400)
                except Exception:
                    st.write('Persisted chunks (backup) (could not render table)')
            except Exception as e:
                st.error(f"Failed to persist chunks: {e}")

        st.session_state.active_step = 4
        try:
            st.rerun()
        except Exception:
            pass

    
    
    st.markdown('</div>', unsafe_allow_html=True)

# Step 5: View Stored Documents
elif st.session_state.active_step == 4:
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    st.subheader("Step 5: View Stored Documents")
    # Center page Chroma collection loader
    st.markdown('**Chroma Collections**')
    if Chroma is None:
        st.info("Chroma vector store not available. Install langchain_community or langchain-chroma to enable collection inspection.")
    else:
        try:
            collections = []
            if os.path.exists(CHROMA_DIR):
                for name in sorted(os.listdir(CHROMA_DIR)):
                    p = os.path.join(CHROMA_DIR, name)
                    if os.path.isdir(p):
                        collections.append(name)
        except Exception:
            collections = []

        if not collections:
            st.info(f"No persisted Chroma collections found under {CHROMA_DIR}")
        else:
            sel = st.selectbox("Choose collection to inspect", options=collections, key="center_chroma_select")
            # load and display the selected collection immediately
            if sel:
                coll_path = os.path.join(CHROMA_DIR, sel)
                embeddings = None
                if OpenAIEmbeddings is not None:
                    try:
                        embeddings = OpenAIEmbeddings()
                    except Exception:
                        embeddings = None
                vect = load_chroma_collection(coll_path, sel, embeddings)
                if vect is None:
                    st.error(f"Failed to load Chroma collection: {sel}")

                if vect is not None:
                    st.session_state['chroma_conn'] = vect
                    st.session_state['chroma_selected'] = sel
                    st.info(f"Loaded collection: {sel}")
                    # try to enumerate items
                    items = None
                    try:
                        if hasattr(vect, 'get'):
                            items = vect.get()
                    except Exception:
                        items = None
                    if items is None:
                        try:
                            col = getattr(vect, '_collection', None)
                            if col is not None and hasattr(col, 'get'):
                                items = col.get()
                        except Exception:
                            items = None

                    if items is None:
                        st.info('Could not enumerate items from this Chroma version. View stored documents from pipeline state below.')
                    else:
                        try:
                            docs = items.get('documents', []) or []
                            metas = items.get('metadatas', []) or []
                            rows = []
                            for i, d in enumerate(docs):
                                meta = metas[i] if i < len(metas) else {}
                                try:
                                    content_len = len(d) if isinstance(d, str) else 0
                                except Exception:
                                    content_len = 0
                                doc_preview = (d[:300] + '...') if isinstance(d, str) and len(d) > 300 else d
                                rows.append({'id': f'{sel}_{i}', 'content_length': content_len, 'document': doc_preview, 'metadata': meta})
                            if rows:
                                st.subheader(f'Contents of {sel}')
                                df = pd.DataFrame(rows)
                                st.dataframe(df, height=300)
                            else:
                                st.info('Collection appears empty')
                        except Exception:
                            st.info('Unable to render collection contents from this Chroma client.')
    
    st.markdown(f"""
    <div class="success-box">
        <p style='color: #065f46; font-weight: 600;'>✓ Successfully stored {len(st.session_state.stored_docs)} documents in ChromaDB</p>
    </div>
    """, unsafe_allow_html=True)
    
    for idx, doc in enumerate(st.session_state.stored_docs):
        doc_preview = doc['document'][:150] if len(doc['document']) > 150 else doc['document']
        with st.expander(f"📄 Document ID: {doc['id']} - Chunk {doc['metadata']['chunk_index'] + 1}/{doc['metadata']['total_chunks']}"):
            st.write("**Content:**")
            st.write(doc_preview + "...")
            st.write("**Embedding:**")
            st.code(doc.get('embedding', '(no embedding)'), language=None)
            st.write("**Metadata:**")
            st.json(doc['metadata'])
    
    if st.button("🔄 Start Over", type="secondary"):
        st.session_state.active_step = 0
        st.session_state.crawled_content = ''
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.stored_docs = []
        st.session_state.url = ''
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Info Box
st.markdown("""
<div class="info-box">
    <p style='color: #1e40af; font-weight: 600; margin-bottom: 5px;'>ℹ️ Pipeline Overview:</p>
    <p style='color: #1e40af; font-size: 14px;'>This demonstrates a typical RAG (Retrieval-Augmented Generation) preprocessing pipeline. Each step transforms your data to make it searchable and retrievable for AI applications.</p>
</div>
""", unsafe_allow_html=True)