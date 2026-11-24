# NexaRAG — Research Paper Question Answering with Advanced RAG

NexaRAG is a Retrieval-Augmented Generation (RAG) project that answers questions from AI/ML research papers. It processes PDF documents, stores searchable chunks in ChromaDB, retrieves relevant context, and generates answers using locally served Ollama models.

The notebook currently downloads and processes three research papers:
- *Attention Is All You Need*
- *BERT: Pre-training of Deep Bidirectional Transformers*
- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*

## Project workflow

```text
Research paper PDFs
        |
        v
PDF ingestion
  |             |                 |
  v             v                 v
Text          Tables          Image metadata
PyMuPDF       pdfplumber       PyMuPDF
  |             |                 |
  v             v                 v
Text chunks   Markdown tables   Metadata chunks
        \         |              /
         \        |             /
          v       v            v
       Recursive Character Text Splitter
          (chunk size: 800; overlap: 150)
                       |
                       v
             Ollama Embeddings
             qwen3-embedding:0.6b
                       |
                       v
                 ChromaDB
                       |
          +------------+-------------+
          |                          |
          v                          v
      Basic RAG                 Advanced retrieval
   Semantic similarity       BM25 + semantic hybrid
          |                   Multi-query experiment
          |                   Context compression experiment
          |                          |
          +------------+-------------+
                       |
                       v
                Ollama LLM
                  qwen3.5:2b
                       |
                       v
                  Generated answer

          LangGraph agent workflow:
          route -> retrieve -> grade step
          -> generate -> grounding check
          -> retry generation when required

          RAGAS evaluation:
          Faithfulness, Answer Relevancy,
          Context Precision, Context Recall
```

## Main features

- **PDF ingestion:** Downloads the three research papers from arXiv.
- **Text extraction:** Uses PyMuPDF to extract page-level text and preserve page and paper metadata.
- **Table extraction:** Uses pdfplumber in the main ingestion flow and converts extracted tables into Markdown text.
- **Image handling:** Extracts image metadata (paper, page, dimensions and content type) for indexing. Vision-LLM image descriptions and CLIP embedding helper functions are present in the notebook, but are not connected to the main indexing pipeline.
- **Chunking:** Uses `RecursiveCharacterTextSplitter` with a chunk size of 800 characters and 150-character overlap.
- **Embeddings:** Uses the local Ollama embedding model `qwen3-embedding:0.6b`.
- **Vector database:** Uses persistent ChromaDB storage.
- **Basic RAG:** Uses semantic similarity retrieval from ChromaDB followed by LLM answer generation.
- **Advanced retrieval experiments:** Includes BM25, semantic retrieval, weighted hybrid retrieval, Multi-Query retrieval and contextual compression.
- **Agentic workflow:** Uses LangGraph to route questions, retrieve context, generate answers and check whether answers are grounded in retrieved documents.
- **Evaluation:** Uses RAGAS metrics to assess faithfulness, answer relevancy, context precision and context recall. The notebook also contains retrieval metrics and hybrid-weight experiments.

## Current model and retrieval configuration

| Setting | Value |
|---|---|
| Generation model | `qwen3.5:2b` |
| Embedding model | `qwen3-embedding:0.6b` |
| Vector database | ChromaDB |
| Chunk size | 800 characters |
| Chunk overlap | 150 characters |
| Default top-k | 5 |
| Hybrid weight in the final hybrid setup | BM25: 0.6, Semantic: 0.4 |
| ChromaDB collection | `research_papers` |

The notebook also tests other BM25/semantic weight combinations. The 60:40 setting is the one used in the later Advanced RAG comparison section.

## Technologies

- Python
- LangChain
- LangGraph
- Ollama
- ChromaDB
- BM25 (`rank-bm25`)
- PyMuPDF
- pdfplumber
- RAGAS
- Hugging Face Transformers / CLIP helper functions

## Setup and execution

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd NexaRAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama separately from the official website:

https://ollama.com/

Start the Ollama service, then download the models used by the notebook:

```bash
ollama pull qwen3.5:2b
ollama pull qwen3-embedding:0.6b
```

The notebook uses the local Ollama service at its default address (`http://localhost:11434`).

### 5. Run the notebook

Open `NexaRAG.ipynb` in Jupyter Notebook, JupyterLab or Google Colab and execute the cells in order.

The notebook currently contains Google Colab/Linux shell commands for installing and starting Ollama. If running locally on Windows, skip or adapt those environment-specific cells and start Ollama separately before running the notebook.

The research PDFs are downloaded by the notebook into `research_pdfs/`. ChromaDB data is stored in `chroma_research_db/`.

## Evaluation

The notebook evaluates retrieval and answer quality using:

| Metric | What it measures |
|---|---|
| Faithfulness | Whether answer claims are supported by retrieved context |
| Answer Relevancy | Whether the answer addresses the question |
| Context Precision | Whether relevant retrieved context is ranked effectively |
| Context Recall | How much of the reference information is covered by retrieved context |

It also calculates Precision@K, Recall@K and Mean Reciprocal Rank (MRR) for retriever comparisons.

**Evaluation note:** The notebook currently uses a small, manually defined evaluation set. Treat its scores as preliminary experimental results, not as a general performance guarantee. LLM-based evaluation can vary between runs.

## Important implementation notes

- In the LangGraph section, `GRADE_DOCS = False` by default. This means the document-relevance grading function is skipped during the default run. The workflow still includes answer-grounding checks and a generation retry path.
- Multi-Query retrieval, contextual compression, table fallback extractors, Vision-LLM descriptions and CLIP embedding functions are explored or defined in the notebook; they are not all connected to the final end-to-end RAG chain.
- The Basic-vs-Advanced evaluation section should be checked so that the contexts recorded for RAGAS are exactly the contexts used to generate each answer. In the current notebook, the Basic RAG chain and the separately recorded Basic RAG contexts use different retrieval settings (`TOP_K` versus `k=3`).

## Repository structure

```text
NexaRAG/
├── NexaRAG.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Future improvements

- Enable and evaluate document relevance grading in the LangGraph workflow.
- Align the contexts captured for evaluation with the contexts actually passed to each generation chain.
- Expand the evaluation dataset and repeat experiments to assess score stability.
- Integrate Vision-LLM figure descriptions and CLIP embeddings into the indexing and retrieval pipeline.
- Connect table fallback extraction methods to the main ingestion workflow.

## Disclaimer

This repository is an experimental learning and portfolio project. Generated answers should be checked against the source research papers.
