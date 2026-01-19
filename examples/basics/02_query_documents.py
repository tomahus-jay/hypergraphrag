"""Example: Query search in Hypergraph RAG"""
import asyncio
from hypergraphrag import HyperGraphRAG

async def main():
    # Initialize client (assumes data is already inserted)
    rag = HyperGraphRAG(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Example queries (HotpotQA related)
    queries = [
        "What other titles did the magazine Science Fantasy appear under?",
        "The Oberoi family is part of a hotel company that has a head office in what city?",
        "Musician and satirist Allie Goertz wrote a song about the \"The Simpsons\" character Milhouse, who Matt Groening named after who?",
        "What nationality was the co-founder of the Baron Hotel in Aleppo?"
    ]
    
    print("🔍 Querying Hypergraph RAG...\n")
    
    for i, query_text in enumerate(queries, 1):
        print(f"Query {i}: {query_text}")
        print("-" * 60)
        
        # Perform search
        results = await rag.query(
            query_text=query_text,
            top_n=5
        )
        
        # Display results
        print(f"📊 Results:")
        print(f"   - Hyperedges found: {len(results.hyperedges)}")
        
        # Show hyperedges
        if results.hyperedges:
            print(f"\n🔗 Top {min(5, len(results.hyperedges))} Hyperedges:")
            for j, hyperedge in enumerate(results.hyperedges[:5], 1):
                print(f"\n   {j}. Content: {hyperedge.content}")
                print(f"      Entities: {', '.join(hyperedge.entity_names)}")
                if hyperedge.chunk:
                    print(f"      Source Chunk:")
                    print(f"         - ID: {hyperedge.chunk.id}")
                    # Truncate content for display if too long
                    content_preview = hyperedge.chunk.content[:100] + "..." if len(hyperedge.chunk.content) > 100 else hyperedge.chunk.content
                    print(f"         - Content: {content_preview}")
                    print(f"         - Metadata: {hyperedge.chunk.metadata}")
        
        print("\n" + "=" * 60 + "\n")
    
    # Clean up resources
    rag.close()
    print("✅ Query search completed!")

if __name__ == "__main__":
    asyncio.run(main())

