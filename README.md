# HyperGraphRAG

Hypergraph-based RAG (Retrieval-Augmented Generation) system

Hypergraphs extend traditional graphs by allowing edges (hyperedges) to connect multiple nodes simultaneously. This enables the modeling of complex, high-order relationships, providing richer context for retrieval tasks compared to standard binary graphs.

> **Note**: Unlike **Ontology-Augmented Generation (OAG)**, which aims for lossless information representation based on strict schemas, this approach emphasizes capturing flexible, latent connections between entities. For mission-critical domains requiring zero information loss, OAG might be more appropriate.

## Overview

This project implements a Hypergraph RAG system using Neo4j for both graph structure and vector search.

![Hypergraph Visualization](docs/images/graph_visualization_01.png)
*An overall view of the constructed hypergraph.*

![Hypergraph Visualization](docs/images/graph_visualization_02.png)
*A zoomed-in view of the constructed hypergraph.*

## Features

- Document data insertion and hypergraph structure creation
- Query-based search and related information retrieval

## Installation

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Or install in development mode
uv sync --dev
```

### Using pip

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file and set the following variables:

```
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Configuration
OPENAI_API_KEY=your_openai_api_key  # Required if using OpenAI
LLM_MODEL=your_llm_model_name
LLM_BASE_URL=  # For self-hosted LLMs (e.g., http://localhost:8000/v1)

# Embedding Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_BASE_URL=  # For self-hosted embedding servers
EMBEDDING_DIMENSION=  # Optional: Embedding dimension (auto-detected if not set)

# Logging (optional)
LOG_LEVEL=INFO
```

## Usage

```python
import asyncio
from hypergraphrag import HyperGraphRAG

async def main():
    # Initialize client
    rag = HyperGraphRAG()

    # Insert data
    # batch_size and max_concurrent_tasks control processing speed vs. load
    await rag.add(
        documents=["Document 1", "Document 2", ...],
        batch_size=5,             
        max_concurrent_tasks=5    
    )

    # Query search
    results = await rag.query(query_text="Your search question")
    
    print(results)

if __name__ == "__main__":
    asyncio.run(main())
```

See `examples/basic_usage.py` for a complete example.

**Example Output:**

```
🔍 Querying Hypergraph RAG...

Query 1: What other titles did the magazine Science Fantasy appear under?
------------------------------------------------------------
...
📊 Results:
   - Chunks found: 2
   - Hyperedges found: 11
   - Entities found: 9

🔝 Top 5 Hyperedges:
   1. Content: Impulse was changed to SF Impulse for the last few issues.
      Entities: Impulse, SF Impulse
      Source Chunk ID: ...
   2. Content: Kyril Bonfiglioli changed the title of Science Fantasy to Impulse in early 1966.
      Entities: Impulse, Kyril Bonfiglioli, Science Fantasy
      Source Chunk ID: ...
   3. Content: Science Fantasy appeared under the titles Impulse and SF Impulse.
      Entities: Science Fantasy, Impulse, SF Impulse
   ...
```

## Performance Tuning

For larger datasets, you can adjust the following parameters in `add`:

- **`batch_size`**: Number of chunks processed in a single LLM batch (Recommended: 5-20).
- **`max_concurrent_tasks`**: Number of batches processed in parallel (Recommended: 3-10).

The system uses a pipeline approach: as soon as a batch is processed, it is immediately stored in Neo4j (Graph + Vectors). This ensures data is saved progressively even during long operations.

## Self-Hosted Support

This project supports various LLM and Embedding services. It automatically detects the appropriate backend based on your configuration (API Key, Base URL).

### LLM Services
- **OpenAI**: Set `OPENAI_API_KEY`.
- **Local/Self-Hosted (Ollama, vLLM, LocalAI)**: Set `LLM_BASE_URL` (and optionally `LLM_API_KEY`).
  - Example `LLM_BASE_URL`: `http://localhost:11434/v1` (Ollama), `http://localhost:8000/v1` (vLLM)

### Embedding Services
- **Sentence Transformers (Local)**: Default if no API key or Base URL is provided.
- **OpenAI Embeddings**: Set `OPENAI_API_KEY` (and ensure `EMBEDDING_BASE_URL` is empty).
- **Custom HTTP / Self-Hosted**: Set `EMBEDDING_BASE_URL`.

## Evaluation Results

Evaluation performed on **HotpotQA** dataset (1000 samples, graph generation and evaluation with **gpt-5-nano-2025-08-07**).

- **Average Precision**: 0.7006
- **Average Recall**: 0.6309
- **Average SP-F1**: 0.6221

## Project Structure

```
hypergraphrag/
├── hypergraphrag/          # Main package
│   ├── __init__.py
│   ├── client.py          # Main client class
│   ├── models.py          # Pydantic models
│   ├── neo4j_manager.py   # Neo4j integration (Graph + Vectors)

│   ├── llm_extractor.py   # LLM-based extraction
│   ├── embedding.py        # Embedding generation
│   ├── text_processor.py  # Text processing
│   └── prompts.py         # LLM prompts
├── tests/                 # Test suite
├── examples/              # Usage examples
├── docs/                  # Documentation
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
