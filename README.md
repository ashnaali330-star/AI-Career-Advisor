# AI Career Advisor Agent

A modular Streamlit application that parses a resume, extracts a structured candidate profile with Gemini, retrieves job requirements using local embeddings and FAISS, then generates a role-specific skills comparison, learning roadmap, role recommendations, and follow-up chat responses.

## Features

- PDF, DOCX, and TXT resume extraction (including Word tables)
- LLM-backed resume analysis with structured JSON output
- Local job-role knowledge base with sentence-transformer embeddings and persistent FAISS vector search
- Retrieved role context supplied to gap analysis, recommendations, roadmap generation, and chat
- Six included role documents, plus a custom target role field
- Session-based resume context and chat history
- Google Gemini API integration through the official `google-genai` SDK
- Automatic fallback from `gemini-2.5-flash` to `gemini-3.5-flash-lite` when the configured model is unavailable
- Friendly handling for missing credentials, unsupported/empty documents, API failures, and retrieval errors

## Architecture

```text
User
 ↓
Streamlit UI
 ↓
Resume Parser
 ↓
Resume Analyzer (LLM)
 ↓
Career Agent → focused tools
 ↓
RAG Retriever ← Job Role Knowledge Base
 ↓                  ↓
LLM             Embeddings → FAISS vector database
 ↓
Career Recommendations → Skill Gap → Learning Roadmap → Career Chat
```

## Folder structure

```text
ai-career-agent/
├── app.py                 # Streamlit interface and session workflow
├── requirements.txt       # Runtime dependencies
├── .env.example           # Provider configuration template
├── data/job_roles/        # Human-readable role knowledge sources
├── rag/                   # Loading, chunking, embedding, vector search, retrieval
├── agent/                 # Prompt templates, tools, and orchestration
├── resume/                # PDF/DOCX/TXT parsing and profile display helpers
├── utils/                 # Gemini client, environment settings, and safe error messages
└── vector_db/             # Generated local FAISS index and metadata
```

## Technologies

Python, Streamlit, Google Gemini API (`google-genai`), Sentence Transformers (`all-MiniLM-L6-v2` by default), FAISS, pypdf, python-docx, and python-dotenv.

## How RAG works

The retriever loads the text files under `data/job_roles`, divides them into overlapping word chunks, embeds each chunk, and stores normalized vectors in a local FAISS inner-product index. On first query or when index files are missing, it rebuilds the index. A query is embedded with the same model; nearest chunks are returned with their source filenames. The agent puts those retrieved passages into the LLM prompt so advice about role requirements is grounded in the included knowledge base. The index files are generated in `vector_db/` and can be rebuilt by deleting them.

## How the agent and tools work

`CareerAgent` orchestrates smaller operations rather than putting all logic in one prompt. The tools in `agent/tools.py` run resume analysis, compare a profile with retrieved role context, generate a roadmap, and recommend roles. Chat retrieves role context for the current target and supplies the session profile and recent conversation. Tool functions are Python functions called by the agent workflow; provider-native function calling is not required for this workflow.

## Installation

Use Python 3.10 or newer. In a terminal, from the directory containing this project:

```powershell
cd ai-career-agent
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` after creating the environment with `python3 -m venv .venv`.

## Environment variables

Copy `.env.example` to `.env`, then set:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key created in Google AI Studio. Never commit it. |
| `GEMINI_MODEL` | Gemini model name. Defaults to `gemini-2.5-flash`; if unavailable, the agent automatically retries once with `gemini-3.5-flash-lite`. |
| `EMBEDDING_MODEL` | Sentence Transformers model used for local role embeddings. |

The Gemini client and model response handling are isolated in `utils/llm_client.py`. The rest of the agent uses provider-neutral `generate_response()` and `generate_json()` methods.

## Run

```powershell
streamlit run app.py
```

The first RAG request downloads/loads the embedding model and builds the local role index. Network access may be required for the first model download and for remote LLM calls.

## Example workflow

1. Configure `.env` and start Streamlit.
2. Upload a text-based PDF, DOCX, or TXT resume.
3. Select a target role and click **Analyze Resume**.
4. Review the extracted candidate profile and click **Run skill gap analysis**.
5. Inspect skill statuses, retrieved source passages, and the generated roadmap.
6. Request role recommendations and ask follow-up questions in AI Career Chat.

## Where the course concepts appear

- **LLM / API integration:** `utils/helpers.py` loads configuration and creates the Gemini client; `utils/llm_client.py` calls the official Gemini SDK; `agent/tools.py` sends task prompts through provider-neutral methods.
- **Prompt engineering:** separate, task-specific prompts in `agent/prompts.py` set evidence and output requirements.
- **RAG:** `rag/` loads role texts, chunks them, retrieves relevant embedded passages, and grounds agent operations.
- **Embeddings / vector database:** `rag/embeddings.py` uses Sentence Transformers; `rag/vector_store.py` persists a FAISS index.
- **Agent workflow / tools:** `agent/career_agent.py` selects and orchestrates Python tool functions in `agent/tools.py`.
- **Document processing:** `resume/parser.py` extracts text from supported resume file formats.
- **Frontend:** `app.py` uses Streamlit widgets, tabs, progress states, tables, and chat.



## Demo questions and answers

**What makes this RAG?** Role documents are embedded, indexed, retrieved by semantic similarity, and inserted into LLM prompts at query time.

**Why use a vector database?** It finds semantically related passages even when the wording in a query differs from the role document wording.

**Does the model invent resume details?** The resume prompt explicitly asks for evidence-only extraction. LLMs can still err, so the profile should be reviewed; the UI labels it as extracted analysis.

**How are gaps decided?** The LLM compares profile evidence with role requirements retrieved from the knowledge base and returns a status plus evidence/explanation when available.

**Where are the uploaded resumes stored?** The app keeps parsed text and profile data in Streamlit's current session state and does not save the upload to the project's disk.

**What happens if I add a role document?** Place a `.txt` role document in `data/job_roles`; the index is rebuilt automatically the next time retrieval runs.

**What happens if Gemini 2.5 Flash is not enabled for my account?** The client retries with Gemini 3.5 Flash-Lite for model-unavailable errors. Invalid-key and quota errors are reported without exposing provider error text or credentials.
