# RAG Knowledge System

A retrieval-augmented generation system for ingesting, indexing, retrieving and answering questions from documents with source citations.

## Public Repository Note

This repository is a sanitised and re-engineered public implementation based on patterns developed in private local-first projects.

All private, organisational, credential, and personally identifiable data has been removed. The public version uses synthetic or publicly available sample data only.

The public implementation focuses on architecture, testing, reproducibility, and transferable engineering patterns rather than reproducing private production data or environment-specific integrations.

## Status

Under active development.

## Project Goals

This project explores:

- document ingestion and parsing
- metadata-aware chunking
- embedding and vector retrieval
- hybrid retrieval and reranking
- source-grounded responses
- retrieval evaluation
- structured outputs
- API deployment and testing

## Architecture

Planned high-level flow:

```mermaid
flowchart LR
    A[Document] --> B[Parse]
    B --> C[Chunk]
    C --> D[Embed]
    D --> E[Vector Store]
    E --> F[Retrieve]
    F --> G[Rerank]
    G --> H[Generate]
    H --> I[Cite Sources]
