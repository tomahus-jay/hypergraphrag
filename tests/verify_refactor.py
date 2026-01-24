import asyncio
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the module explicitly so we can patch the object on it
try:
    from hypergraphrag import client
except ImportError:
    # If package structure is an issue, try appending parent
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from hypergraphrag import client

async def verify():
    print("Verifying refactor...")
    
    # Patch Neo4jManager and EmbeddingGenerator in the client module namespace
    with patch.object(client, "Neo4jManager") as MockNeo4j, \
         patch.object(client, "EmbeddingGenerator") as MockEmbed:
        
        # Setup mocks
        mock_neo4j_instance = MockNeo4j.return_value
        mock_embed_instance = MockEmbed.return_value
        
        # Configure specific mock returns
        mock_embed_instance.dimension = 384
        mock_embed_instance.generate_embedding.return_value = [0.1]*384
        
        # Mock step 0: BM25
        mock_neo4j_instance.get_top_documents_by_bm25.return_value = ["doc1", "doc2"]

        # Mock step 1: Hybrid Search
        # This replaces search_vectors for the initial entity search
        mock_neo4j_instance.search_vectors_with_document_bias.return_value = [
            {"name": "SpaceX", "score": 0.9}
        ]
        
        # Mock get_best_hyperedges_with_entities
        mock_neo4j_instance.get_best_hyperedges_with_entities.return_value = [
            {
                "hyperedge_id": "h1",
                "content": "SpaceX launches rockets.",
                "chunk_id": "c1",
                "metadata": "{}",
                "score": 0.9,
                "entity_names": ["SpaceX", "Rocket"]
            }
        ]
        
        # Chunks retrieval
        mock_neo4j_instance.get_chunks_by_ids.return_value = [
            {
                "id": "c1",
                "content": "Full chunk content...",
                "metadata": {"source": "test"}
            }
        ]
        
        # Initialize client (uses mocks)
        rag_client = client.HyperGraphRAG(
            neo4j_uri="bolt://mock",
            embedding_model="mock"
        )
        
        # Run query
        result = await rag_client.query("test query")
        
        print("\nQuery Result Structure:")
        print(f"Query: {result.query}")
        print(f"Hyperedges Found: {len(result.hyperedges)}")
        
        if len(result.hyperedges) > 0:
            he = result.hyperedges[0]
            print(f"Hyperedge 1 Content: {he.content}")
            print(f"Hyperedge 1 ChunkID: {he.chunk_id}")
            if he.chunk:
                 print(f"Hyperedge 1 Source Chunk: {he.chunk.id}")
        
        # Assertions
        assert len(result.hyperedges) == 1
        assert result.hyperedges[0].hyperedge_id == "h1"
        assert result.hyperedges[0].chunk_id == "c1"
        assert result.hyperedges[0].chunk is not None
        assert result.hyperedges[0].chunk.id == "c1"
        
        # Verify calls
        mock_neo4j_instance.get_top_documents_by_bm25.assert_called_once()
        mock_neo4j_instance.search_vectors_with_document_bias.assert_called_once()
        mock_neo4j_instance.get_best_hyperedges_with_entities.assert_called_once()
        
        print("\nVerification Successful!")

if __name__ == "__main__":
    asyncio.run(verify())
