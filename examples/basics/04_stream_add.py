"""Example: Stream Add documents into Hypergraph RAG with Text Logging"""
import asyncio

from torch import batch_norm
from hypergraphrag import HyperGraphRAG

async def main():
    # Initialize client
    rag = HyperGraphRAG(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Example documents to add (Same as basic_add.py)
    documents = [
        """
        Artificial Intelligence (AI) is a technology designed to enable computer systems 
        to mimic human intelligence. Machine Learning is a branch of AI that learns from 
        data to recognize patterns and make predictions. Deep Learning is a subfield of 
        Machine Learning that uses neural networks.
        """,
        """
        Natural Language Processing (NLP) is a technology that enables computers to 
        understand and process human language. It has various applications including 
        text analysis, sentiment analysis, and machine translation. Recently, Large 
        Language Models (LLMs) have revolutionized the NLP field.
        """,
        """
        RAG (Retrieval-Augmented Generation) is an AI model that combines retrieval and 
        generation. It retrieves relevant information from external knowledge bases to 
        generate more accurate answers. Hypergraph RAG represents complex relationships 
        between documents using hypergraphs.
        """,
        """
        Vector databases are specialized databases designed to store and search high-dimensional 
        vectors efficiently. They are essential for similarity search in machine learning 
        applications. Popular vector databases include Qdrant, Pinecone, and Weaviate.
        """,
        """
        Graph databases store data in nodes and edges, representing relationships between entities. 
        Neo4j is a popular graph database that uses Cypher query language. Graph databases are 
        ideal for complex relationship queries and network analysis.
        """
    ]
    
    # Metadata
    metadata = [
        {"source": "ai_intro", "category": "basic", "topic": "AI"},
        {"source": "nlp_intro", "category": "basic", "topic": "NLP"},
        {"source": "rag_intro", "category": "advanced", "topic": "RAG"},
        {"source": "vector_db", "category": "intermediate", "topic": "Database"},
        {"source": "graph_db", "category": "intermediate", "topic": "Database"}
    ]
    
    print(f"🚀 Starting Stream Add...")
    
    # Add data using stream method with manual logging
    # Using batch_size=1 to see granular updates for this small dataset
    async for update in rag.add_stream(
        documents=documents,
        metadata=metadata,
        batch_size=1,
        max_concurrent_tasks=5
    ):
        status = update["status"]
        
        if status == "chunking_complete":
            print(f"📦 Chunking Complete: Total {update['total_chunks']} chunks ready.")
            
        elif status == "processing":
            progress = update["progress"]
            current = update["completed_batches"]
            total = update["total_batches"]
            stats = update["total_stats"]
            
            # Print single line log
            print(
                f"🔄 [Processing] Batch {current}/{total} ({progress:.1f}%) "
                f"| Entities: {stats['entities']} | Hyperedges: {stats['hyperedges']} | Chunks: {stats['chunks']}"
            )
            
        elif status == "complete":
            print("\n✅ Add Completed Successfully!")
            print("📊 Final Statistics:")
            final_stats = update["total_stats"]
            print(f"   - Total Chunks:     {final_stats['chunks']}")
            print(f"   - Total Entities:   {final_stats['entities']}")
            print(f"   - Total Hyperedges: {final_stats['hyperedges']}")
    
    # Clean up resources
    rag.close()

if __name__ == "__main__":
    asyncio.run(main())
