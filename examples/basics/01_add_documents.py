"""Example: Add documents into Hypergraph RAG"""
import asyncio
from hypergraphrag import HyperGraphRAG

async def main():
    # Initialize client
    rag = HyperGraphRAG(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Example documents to add
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
    
    # Add data with metadata
    print(f"📝 Adding documents into Hypergraph RAG...")
    doc_ids = await rag.add(
        documents=documents,
        metadata=[
            {"source": "ai_intro", "category": "basic", "topic": "AI"},
            {"source": "nlp_intro", "category": "basic", "topic": "NLP"},
            {"source": "rag_intro", "category": "advanced", "topic": "RAG"},
            {"source": "vector_db", "category": "intermediate", "topic": "Database"},
            {"source": "graph_db", "category": "intermediate", "topic": "Database"}
        ],
        batch_size=1,
        max_concurrent_tasks=5
    )
    
    print(f"✅ Documents added successfully! Doc IDs: {doc_ids}")
    print("   - Documents are chunked and stored")
    print("   - Entities and hyperedges are extracted")
    print("   - Embeddings are generated and stored in Neo4j")
    print("   - Graph structure is created in Neo4j")
    
    # Clean up resources
    rag.close()

if __name__ == "__main__":
    asyncio.run(main())

