# RAG Knowledge System

A retrieval-augmented generation system for ingesting, indexing, retrieving and answering questions from documents with source citations.

## Public Repository Note

This repository is a sanitised and re-engineered public implementation derived from patterns used in private local-first projects.

The original implementations contain organisation-specific workflows, private operational data, credentials, local infrastructure details, and environment-specific integrations that are not suitable for public release.

This public version therefore reconstructs the core engineering patterns using synthetic or publicly available sample data. The focus is on demonstrating transferable architecture and engineering practices, including ingestion, chunking, embedding, retrieval, filtering, reranking, testing, and reproducibility.

The public repository is not intended to reproduce the private production environment or its data.

## Status

Core retrieval and source-grounded generation pipeline implemented.

Current capabilities:

- validated text and Markdown ingestion
- character, word, token and paragraph-aware chunking
- Hugging Face sentence embeddings
- in-memory vector storage
- cosine similarity search
- metadata filtering
- top-k semantic retrieval
- hybrid semantic and keyword retrieval
- second-stage reranking
- end-to-end retrieval pipeline
- automated tests
- ground-truth retrieval evaluation
- Hit Rate@K, Recall@K and Mean Reciprocal Rank (MRR)
- runnable semantic retrieval benchmark
- source-grounded answer generation
- source provenance and citation tracking

## Architecture

```mermaid
flowchart LR
    A[Document] --> B[Parse]
    B --> C[Chunk]
    C --> D[Embed]
    D --> E[Vector Store]
    E --> F[Semantic Retrieve]
    F --> G[Hybrid Score]
    G --> H[Rerank]
    H --> I[Top-K Results]
    I --> J[Build Grounded Context]
    J --> K[Generate]
    K --> L[Answer with Source Citations]
```

## Retrieval Evaluation

The repository includes a small labelled retrieval benchmark using a synthetic corpus and manually defined ground-truth relevant chunks.

Metrics:

- Hit Rate@K
- Recall@K
- Mean Reciprocal Rank (MRR)

Run the benchmark with:

```powershell
python .\eval\run_retrieval_eval.py
```

Current semantic retrieval baseline:

```
hit_rate@3: 1.000
recall@3: 1.000
mrr: 1.000
```

The benchmark is intentionally small and is intended to provide a reproducible baseline for comparing semantic, hybrid and reranked retrieval strategies.

## Source-grounded Generation

The generation layer turns retrieved chunks into grounded answers while preserving source provenance.

The current implementation:

- formats retrieved chunks into numbered source context
- instructs the language model to answer only from retrieved evidence
- requires citations using `[Source N]`
- returns both the generated answer and its supporting sources
- avoids calling the language model when no retrieval results are available
- keeps the language model interface injectable for provider-independent testing

Example flow:

```
Question
  -> Retriever
  -> SearchResult[]
  -> Grounded Context
  -> Prompt
  -> LLM
  -> GenerationResult
     - answer
     - sources
```
