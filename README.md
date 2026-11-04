# Female Wellness & Lifestyle MCP Server

A production-oriented, modular, and evidence-grounded **Model Context Protocol (MCP)** server tailored for a Female Wellness & Lifestyle AI Platform.

Built with **FastMCP / MCPServer**, **ChromaDB**, **Sentence Transformers**, and the **Anthropic Claude API**, this server connects AI host applications (such as Claude Desktop or autonomous agents) to verified female health research, life-stage context, and deterministic clinical safety guardrails over standard I/O streams (`stdio`).

---

## 1. Project Overview

The platform addresses female health, hormonal rhythms, longevity, athletic periodization, and maternal wellness across **six primary life-stage journeys**:

```text
                               Female Wellness AI Platform
                              Six Personalized Life-Stage Journeys
  ┌──────────────────────┬──────────────────────┬──────────────────────┐
  │  Beauty & Radiance   │  Cycle & Fertility   │ Athlete Performance  │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │Pregnancy & Postpartum│    Perimenopause     │  Lifelong Thriving   │
  └──────────────────────┴──────────────────────┴──────────────────────┘
```

1. **Beauty & Radiance**: Connects internal health with external vitality. Emphasizes gut-skin axis health, epidermal barrier nutrition, cellular hydration, hair follicle health, and restorative sleep.
2. **Cycle & Fertility**: Establishes hormonal baselines and fertility awareness. Focuses on the four phases of the menstrual cycle, basal body temperature (BBT) tracking, ovulation awareness, fertile window education, and luteal phase support.
3. **Athlete Performance**: Optimizes training loads and athletic recovery around female physiology, Heart Rate Variability (HRV), autonomic nervous system tracking, cycle-based periodization, and RED-s prevention.
4. **Pregnancy & Postpartum**: Supports maternal wellness throughout pregnancy and Fourth Trimester recovery. Covers trimester-specific nutrition (choline, folate, iron, DHA), symptom education, and pelvic floor restoration.
5. **Perimenopause**: Navigates midlife hormonal transitions with clarity. Emphasizes vasomotor symptom thermoregulation (hot flashes, night sweats), sleep preservation, neurosteroid fluctuations, and bone mineral density preservation.
6. **Lifelong Thriving**: Supports healthy aging, functional mobility, sarcopenia prevention, cognitive neuroprotection, and long-term vitality.

> [!IMPORTANT]
> These six categories represent **product journeys and wellness domains**, not clinical medical diagnoses.

---

## 2. System Architecture

The platform uses a **dual-mode architecture** operating over the Model Context Protocol:

1. **Host-Orchestrated Reasoning (Primary MCP Mode)**: An MCP client (such as Claude Desktop) invokes retrieval tools (`search_wellness_knowledge`, `get_wellness_recommendations`), inspects resources (`wellness://...`), and performs its own language reasoning directly over the returned citations.
2. **Server-Side Generation Mode**: Headless clients or automated agents invoke `ask_wellness_question`, which executes end-to-end RAG retrieval, passes evidence to the internal Anthropic Claude service, and returns synthesized answers with citations.

### Architectural Diagram

```text
MCP Client (Claude Desktop / CLI Host)
                  │
                  │ STDIO (JSON-RPC 2.0)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 Python FastMCP Server                       │
│                                                             │
│  ┌───────────────────────┐       ┌───────────────────────┐  │
│  │   Life Stage Engine   │       │  Safety & Escalation  │  │
│  │ (life_stage_service)  │       │   (safety_service)    │  │
│  └──────────┬────────────┘       └──────────┬────────────┘  │
│             │                               │               │
│  ┌──────────▼───────────────────────────────▼────────────┐  │
│  │                    RAG Pipeline                       │  │
│  │  Retriever ──► Sentence-Aware Chunker ──► Doc Loader  │  │
│  └──────────────────────────┬────────────────────────────┘  │
└─────────────────────────────┼───────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
     ┌───────────────────┐         ┌───────────────────┐
     │     ChromaDB      │         │   Anthropic API   │
     │   Vector Store    │         │ (Claude-3.5-Sonnet│
     │ (HNSW Cosine idx) │         │  Grounded Synth)  │
     └─────────┬─────────┘         └───────────────────┘
               │
               ▼
     ┌───────────────────┐
     │   Knowledge Base  │
     │  (PDF, DOCX, MD)  │
     └───────────────────┘
```

### Critical Protocol Rule: STDIO Purity
FastMCP communicates with MCP clients using standard input and output (`stdio`). Any arbitrary print statement or third-party log written to `stdout` corrupts the JSON-RPC framing and disconnects the client. In this architecture:
- `sys.stdout` is reserved **exclusively** for MCP JSON-RPC protocol messages.
- All application logs, diagnostics, and vector database outputs are routed strictly to `sys.stderr`.
- API keys (`sk-ant-...`) are intercepted and masked by a custom logging filter.

---

## 3. Technology Stack

| Technology | Version / Spec | Purpose |
| :--- | :--- | :--- |
| **Python** | 3.12+ (tested on 3.13.5) | Core application runtime |
| **FastMCP (`mcp[cli]`)** | `>=1.3.0` (mcp 2.x compatible) | MCP protocol server, tools, resources, and prompt registry |
| **Anthropic Python SDK** | `>=0.40.0` | Claude API integration for server-side grounded synthesis |
| **ChromaDB** | `>=0.5.0` | Embedded local vector database with persistent SQLite/Parquet backing |
| **Sentence Transformers** | `>=3.0.0` | Local dense vector embedding generation |
| **Embedding Model** | `BAAI/bge-small-en-v1.5` | 384-dimensional dense embeddings running locally on CPU |
| **PyMuPDF (`pymupdf`)** | `>=1.24.0` | Page-accurate text extraction from PDF research papers |
| **python-docx** | `>=1.1.0` | Structural text extraction from DOCX documents |
| **Pydantic v2 & Settings** | `>=2.7.0` | Centralized settings, validation, and JSON schemas |
| **Pytest & Pytest-Asyncio** | `>=8.0.0` | Comprehensive unit and integration test suite |
| **Ruff** | `>=0.6.0` | High-speed linting and code formatting |

---

## 4. Directory Structure

```text
mcp-server/
├── .env.example                               # Environment variable template
├── .gitignore                                 # Git ignore (excludes chroma_db/, .venv, .env)
├── LICENSE                                    # MIT License
├── README.md                                  # Comprehensive documentation
├── pyproject.toml                             # PEP 621 package metadata & configuration
│
├── src/
│   └── female_wellness/
│       ├── __init__.py                        # Package root
│       ├── server.py                          # FastMCP entry point & STDIO transport runner
│       ├── config.py                          # Centralized Pydantic Settings
│       │
│       ├── core/                              # Core constants and infrastructure
│       │   ├── __init__.py
│       │   ├── constants.py                   # Life stage IDs, categories, system defaults
│       │   ├── exceptions.py                  # Domain exception hierarchy (FemaleWellnessError)
│       │   └── logging_config.py              # Stderr-only logging with secret key redaction
│       │
│       ├── mcp/                               # Model Context Protocol endpoints
│       │   ├── __init__.py
│       │   ├── tools/                         # 7 MCP Tools
│       │   │   ├── __init__.py
│       │   │   ├── life_stage_tools.py        # Journey discovery & feature inspection
│       │   │   ├── knowledge_tools.py         # Semantic search, sources, stats
│       │   │   └── recommendation_tools.py    # Evidence-grounded QA & recommendations
│       │   ├── resources/                     # 3 MCP Resources (wellness://...)
│       │   │   ├── __init__.py
│       │   │   └── knowledge_resources.py    # Static/dynamic context endpoints
│       │   └── prompts/                       # 3 MCP Prompt Templates
│       │       ├── __init__.py
│       │       └── wellness_prompts.py        # Question answering & recommendation prompts
│       │
│       ├── rag/                               # RAG & Vector Subsystem
│       │   ├── __init__.py
│       │   ├── embeddings.py                  # SentenceTransformer wrapper (bge-small-en-v1.5)
│       │   ├── vector_store.py                # BaseVectorStore abstraction & ChromaVectorStore
│       │   ├── retriever.py                   # Semantic search & threshold filter
│       │   ├── document_loader.py             # PDF, DOCX, TXT, MD loader with page tracking
│       │   ├── chunking.py                    # Sentence-aware chunker with overlap
│       │   └── pipeline.py                    # Ingestion coordinator & context assembler
│       │
│       ├── services/                          # Business logic layer
│       │   ├── __init__.py
│       │   ├── anthropic_service.py           # Claude API integration & error handling
│       │   ├── life_stage_service.py          # 6 Life-stage journey engine & privacy rules
│       │   └── safety_service.py              # Deterministic red-flag & boundary engine
│       │
│       ├── schemas/                           # Pydantic v2 schemas
│       │   ├── __init__.py
│       │   ├── documents.py                   # Document, chunk, and collection schemas
│       │   ├── recommendations.py             # Answer, recommendation, and item models
│       │   └── wellness.py                    # Citations, context, and user profile schemas
│       │
│       └── utils/                             # Utilities
│           ├── __init__.py
│           └── text_utils.py                  # Unicode normalization & sentence splitting
│
├── knowledge_base/
│   ├── raw/                                   # Source documents organized by journey
│   │   ├── beauty_radiance/
│   │   ├── cycle_fertility/
│   │   ├── athlete_performance/
│   │   ├── pregnancy_postpartum/
│   │   ├── perimenopause/
│   │   ├── lifelong_thriving/
│   │   └── general/
│   └── processed/                             # Ingestion artifacts & caches
│
├── chroma_db/                                 # Persistent ChromaDB storage (auto-created)
│
├── scripts/                                   # Operational CLI utilities
│   ├── ingest_documents.py                    # Batch document ingestion CLI (--dry-run supported)
│   ├── clear_vector_store.py                  # Vector store reset utility
│   └── inspect_collection.py                  # Collection inspection and record viewer
│
└── tests/                                     # Comprehensive test suite (74 tests)
    ├── __init__.py
    ├── test_config.py                         # Settings, logging, and exceptions
    ├── test_document_loader.py                # PDF, DOCX, TXT, MD loading & metadata
    ├── test_chunking.py                       # Sentence boundaries & overlap preservation
    ├── test_embeddings.py                     # Vector dimension, normalization, and ranking
    ├── test_vector_store.py                   # ChromaDB CRUD, persistence, filtering
    ├── test_retriever.py                      # Retrieval, threshold cutoffs, RAG pipeline
    ├── test_life_stages.py                    # 6 Journeys, validation, privacy non-inference
    ├── test_anthropic_service.py              # Claude generation & exception translation
    ├── test_mcp_tools.py                      # MCP tools, resources, and prompt execution
    └── test_safety.py                         # Medical red-flags & prescriptive boundaries
```

---

## 5. Installation & Setup

### Prerequisites
- Linux, macOS, or Windows (WSL recommended on Windows).
- Python 3.12 or higher.

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd mcp-server

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

# 4. Install the package in editable mode with development dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 6. Environment Configuration

Copy the provided `.env.example` file to create your local `.env`:

```bash
cp .env.example .env
```

### Configurable Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `ANTHROPIC_API_KEY` | *(Optional)* | Anthropic Claude API key (`sk-ant-...`). Optional if Claude Desktop handles reasoning; required for server-side `ask_wellness_question`. |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Claude model identifier for server-side generation. |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db` | Local filesystem directory where ChromaDB stores SQLite and Parquet files. |
| `CHROMA_COLLECTION_NAME` | `female_wellness_knowledge`| ChromaDB collection name. |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence Transformers Hugging Face identifier or local checkpoint path. |
| `CHUNK_SIZE` | `800` | Target maximum character length for each document chunk. |
| `CHUNK_OVERLAP` | `150` | Character overlap between consecutive chunks. |
| `TOP_K` | `5` | Default number of chunks retrieved per search. |
| `SIMILARITY_THRESHOLD` | `0.35` | Minimum cosine similarity score ($1.0 - \text{distance}$) for evidence acceptance. |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Always written to stderr. |

---

## 7. Knowledge Ingestion CLI

Documents are placed into `knowledge_base/raw/` grouped by journey folder:
```text
knowledge_base/raw/
├── beauty_radiance/
├── cycle_fertility/
├── athlete_performance/
├── pregnancy_postpartum/
├── perimenopause/
├── lifelong_thriving/
└── general/
```

### Supported Document Formats
- **PDF (`.pdf`)**: Extracted page-by-page using PyMuPDF (`pymupdf`), tracking 1-based page numbers.
- **Word (`.docx`)**: Body paragraphs and tables extracted using `python-docx`.
- **Markdown (`.md`)**: Structured headers and sections extracted.
- **Plain Text (`.txt`)**: Text cleaned and normalized.

### Running Ingestion

```bash
# 1. Dry run: Verify discovery, page counts, and chunk planning without modifying ChromaDB
python scripts/ingest_documents.py --dry-run

# 2. Live Ingestion: Chunk, embed, and index into ChromaDB
python scripts/ingest_documents.py

# 3. Inspect Collection: Verify indexed chunk counts and view sample records
python scripts/inspect_collection.py --limit 3

# 4. Clear Vector Store: Reset the collection if re-indexing from scratch
python scripts/clear_vector_store.py --force
```

---

## 8. MCP Tools, Resources, and Prompts Catalog

### A. MCP Tools

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| `get_available_life_stages` | *None* | Returns all six supported female wellness journeys with summaries. |
| `get_life_stage_details` | `life_stage: str` | Returns comprehensive specifications, supported wellness categories, and features for a given journey. |
| `search_wellness_knowledge`| `query: str`, `life_stage: str?`, `category: str?`, `top_k: int = 5` | Semantic similarity search across indexed documents with metadata filtering. Returns text, source, page number, and similarity score. |
| `ask_wellness_question` | `question: str`, `life_stage: str`, `user_context: dict?` | Evaluates medical safety rules, retrieves relevant evidence, and generates an evidence-grounded answer with citations. |
| `get_wellness_recommendations` | `life_stage: str`, `category: str?`, `user_context: dict?` | Retrieves actionable, evidence-backed lifestyle, nutrition, and recovery recommendations for the selected journey. |
| `get_knowledge_sources` | `life_stage: str?`, `category: str?` | Catalogs indexed source documents, page counts, and coverage. |
| `get_knowledge_base_stats` | *None* | Returns aggregate statistics: total chunks, documents, available journeys, and categories. |

### B. MCP Resources

| Resource URI | MIME Type | Description |
| :--- | :--- | :--- |
| `wellness://life-stages` | `application/json` | Complete JSON specifications for all six life-stage journeys. |
| `wellness://knowledge-base` | `application/json` | Real-time statistics, document counts, and categories indexed in ChromaDB. |
| `wellness://safety-guidelines`| `text/markdown` | Ethical guidelines, non-prescriptive boundaries, and red-flag escalation principles. |

### C. MCP Prompts

| Prompt Name | Arguments | Description |
| :--- | :--- | :--- |
| `wellness_question_prompt` | `question: str`, `life_stage: str` | Guides the host LLM to formulate compassionate, evidence-grounded answers citing sources and acknowledging biological variation. |
| `life_stage_recommendation_prompt`| `life_stage: str`, `category: str = "general"` | Directs generation of phase-appropriate nutrition, sleep, and exercise guidance. |
| `evidence_based_wellness_prompt` | `topic: str`, `life_stage: str` | Instructs the model to synthesize physiological mechanisms and study findings. |

---

## 9. Running the MCP Server

### Standalone CLI Execution

```bash
# Run server using Python module
python -m female_wellness.server

# Or use the installed console entrypoint
female-wellness-mcp
```
The server will start listening on `stdio` for JSON-RPC 2.0 protocol messages. Diagnostic logs will appear on `stderr`.

---

## 10. Claude Desktop Integration

Claude Desktop communicates with MCP servers by spawning them as background subprocesses over `stdio`.

### Locating Your Claude Desktop Configuration

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Configuration Snippet

Add the following configuration to `claude_desktop_config.json`. Use the **absolute path** to your virtual environment's Python executable and project directory:

```json
{
  "mcpServers": {
    "female-wellness": {
      "command": "/home/rasel/Projects_hub/mcp-server/.venv/bin/python",
      "args": [
        "-m",
        "female_wellness.server"
      ],
      "cwd": "/home/rasel/Projects_hub/mcp-server",
      "env": {
        "ANTHROPIC_MODEL": "claude-3-5-sonnet-20241022",
        "CHROMA_PERSIST_DIRECTORY": "/home/rasel/Projects_hub/mcp-server/chroma_db",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

> [!TIP]
> After modifying `claude_desktop_config.json`, fully quit and restart Claude Desktop. The hammer icon in the bottom right of the input box will display the registered `female-wellness` tools.

---

## 11. Female Health & Medical Safety Engine

Health guidance requires deterministic safeguards. The `SafetyService` operates **upstream of the RAG pipeline and LLM**, ensuring safety rules are executed with zero hallucination risk:

```text
User Question
      │
      ▼
┌──────────────────────────────────────────────┐
│  Safety Gate (safety_service.py)             │
│  • Preeclampsia screening (headache + aura)  │
│  • Hemorrhage / Ectopic screening            │
│  • DVT screening (unilateral calf swelling)  │
│  • Acute Cardiac screening (chest pain)      │
│  • Postpartum Sepsis (fever + foul lochia)   │
│  • Prescription dosage adjustment blocker    │
│  • Self-diagnosis request blocker            │
└──────┬────────────────────────────────┬──────┘
       │                                │
  [Red Flag Detected]              [Safe Query]
       │                                │
       ▼                                ▼
Immediate Clinical Alert         Proceed to RAG & LLM
(Halts general chit-chat)
```

### Safety Rules Enforced:
1. **Educational Scope**: Guidance strictly supports lifestyle, nutrition, and wellness habits.
2. **Never Diagnose**: Inquiries asking *"Do I have endometriosis/PCOS?"* are redirected to licensed healthcare providers for clinical imaging and biomarker tests.
3. **Never Prescribe or Modify Dosages**: Questions requesting medication dosages (e.g. Metformin, Levothyroxine) are blocked.
4. **Physiological Individuality**: Never assumes all women have regular 28-day cycles, nor that all users are fertile, sexually active, or pregnant.
5. **Privacy by Default**: The system **never** infers pregnancy or menopausal status automatically. Only user-provided profile context is incorporated.

---

## 12. Testing

The project maintains a 100% passing test suite across all 10 phases. External LLM calls are mocked, allowing tests to run completely offline without an API key.

```bash
# Run all 74 unit and integration tests
pytest tests/ -v

# Run with coverage report
pytest --cov=src/female_wellness tests/

# Run linter checks
ruff check src tests scripts
```

### Test Suite Coverage

| Test Module | Tests | Focus Area |
| :--- | :--- | :--- |
| `test_config.py` | 9 | Pydantic settings, overlap validation, secret masking, stderr logging |
| `test_document_loader.py` | 6 | PDF, DOCX, TXT, MD loading, life-stage inference, error handling |
| `test_chunking.py` | 4 | Sentence boundary retention, overlap seeding, Chroma serialization |
| `test_embeddings.py` | 6 | 384-dimension vector output, unit normalization, similarity ranking |
| `test_vector_store.py` | 8 | ChromaDB CRUD, persistence, cosine distance metric, metadata filtering |
| `test_retriever.py` | 5 | Semantic retrieval, life stage isolation, similarity cutoffs, RAG context |
| `test_life_stages.py` | 8 | All 6 journeys, normalization, feature specs, privacy non-inference |
| `test_anthropic_service.py` | 7 | Grounded answer generation, refusal on empty evidence, rate limit & timeout handling |
| `test_mcp_tools.py` | 10 | Live execution of all 7 tools, 3 resources, and 3 prompt templates |
| `test_safety.py` | 11 | Preeclampsia, DVT, hemorrhage, cardiac alerts, dosage blocking, disclaimers |
| **Total** | **74** | **100% Passing** |

---

## 13. Troubleshooting & Common Issues

### 1. Claude Desktop shows `Server disconnected` or JSON-RPC errors
- **Cause**: Standard output (`stdout`) was polluted by third-party print statements or library logs.
- **Solution**: Ensure all logging uses `female_wellness.core.logging_config.get_logger()`. Logs must write strictly to `sys.stderr`.

### 2. `ModuleNotFoundError: No module named 'female_wellness'`
- **Cause**: The package was not installed into the active virtual environment.
- **Solution**: Activate your `.venv` and run `pip install -e .` from the repository root.

### 3. `AnthropicAPIKeyMissingError` when invoking `ask_wellness_question`
- **Cause**: No Anthropic API key was configured in `.env`.
- **Solution**: Provide an `ANTHROPIC_API_KEY=sk-ant-...` in `.env` for server-side generation. If running in Claude Desktop, Claude Desktop can reason over `search_wellness_knowledge` directly without needing a server-side key.

### 4. `CHUNK_OVERLAP must be strictly less than CHUNK_SIZE`
- **Cause**: Misconfigured chunk settings in `.env`.
- **Solution**: Ensure `CHUNK_OVERLAP < CHUNK_SIZE` (default is 800 size, 150 overlap).

---

## 14. Production Roadmap & Future Improvements

1. **PostgreSQL pgvector Migration**: Swap `ChromaVectorStore` with a PostgreSQL `pgvector` implementation using the modular `BaseVectorStore` interface for distributed multi-tenant workloads.
2. **Hybrid Search (BM25 + Dense Vectors)**: Integrate sparse lexical search (BM25 or Reciprocal Rank Fusion) alongside dense BGE embeddings to enhance keyword retrieval for rare medical terminology.
3. **Async Batch Ingestion**: Leverage `asyncio` worker pools for parallelizing multi-thousand document PDF parsing during enterprise document ingestion.
4. **Biometric Device Ingestion**: Add wearable data loaders (Apple HealthKit, Oura, Whoop) to stream real-time HRV and resting heart rate data into the `Athlete Performance` journey.

---

## 15. License

This project is licensed under the MIT License — see the [`LICENSE`](file:///home/rasel/Projects_hub/mcp-server/LICENSE) file for details.
