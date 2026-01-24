"""Basic usage example for Hypergraph RAG"""
from hypergraphrag import HyperGraphRAG

# Initialize client
rag = HyperGraphRAG(
    chunk_size=500,
    chunk_overlap=50
)

# Example documents: "Project Chimera" - A fictional tech/political thriller scenario
documents = [
    # Document 1: Corporate secrecy, investment, and controversial technology
    """
    Global tech giant 'Argos Corp' has been secretly developing a revolutionary AI project code-named 'Chimera'. 
    Whistleblower reports indicate that CEO Marcus Thorne signed an undisclosed agreement with the 'Silverstone Investment Group' 
    to secure funding for Chimera, bypassing the board of directors. 
    The project aims to integrate autonomous surveillance systems into smart city infrastructures, raising major privacy concerns.
    """,
    
    # Document 2: Geopolitics, legislation, and civil unrest
    """
    The Eastern European nation of 'Kravonia' recently passed the controversial Cybersecurity Act of 2024, 
    granting broad surveillance powers to the state. 
    Coincidentally, Argos Corp announced plans to build a massive data center in Kravonia's capital, Volgrad. 
    The civil rights organization 'Liberty Net' has launched protests in Volgrad, claiming the new law is designed 
    to facilitate Argos Corp's Chimera project in exchange for government kickbacks.
    """,
    
    # Document 3: Financial tracing and corruption investigation
    """
    Financial investigators have traced a series of suspicious transactions from 'Silverstone Investment Group' 
    through shell companies in the 'Isle of Mist' tax haven. 
    The funds appear to have ended up in accounts linked to high-ranking officials in the Kravonia government. 
    Interpol is currently assessing whether to launch a formal investigation into potential bribery 
    involving Marcus Thorne and Kravonia's Prime Minister, Viktor Krum.
    """
]

# Insert data (async)
import asyncio

async def main():
    # Reset database for clean test
    rag.reset_database()
    
    # Insert data
    print(f"📝 Inserting documents...")
    doc_ids = await rag.add(
        documents=documents,
        metadata=[
            {"source": "report_1", "category": "corporate_leak"},
            {"source": "news_article", "category": "geopolitics"},
            {"source": "investigation_memo", "category": "financial_crime"}
        ]
    )
    
    # Query search using local search (entity-centric graph traversal)
    # This query requires connecting Argos Corp -> Chimera -> Kravonia -> Silverstone -> Corruption
    query = "What is the relationship between Argos Corp's Chimera project and the Kravonia government, and what controversy is involved?"
    
    print("\n🔍 Searching...")
    results = await rag.query(
        query_text=query,
        top_n=10
    )
    
    # Print results
    print(f"\n📊 Search Results:")
    print(f"- Query: {results.query}")
    print(f"- Hyperedges found: {len(results.hyperedges)}")
    
    if results.hyperedges:
        print(f"\n🔗 Hyperedge Information:")
        for i, hyperedge in enumerate(results.hyperedges[:5], 1):
            print(f"\n{i}. Hyperedge ID: {hyperedge.hyperedge_id}")
            print(f"   Content: {hyperedge.content}")
            print(f"   Entities: {', '.join(hyperedge.entity_names)}")
            if hyperedge.chunk:
                print(f"      Source Chunk:")
                print(f"         - ID: {hyperedge.chunk.id}")
                print(f"         - Content: {hyperedge.chunk.content}")
                print(f"         - Metadata: {hyperedge.chunk.metadata}")
    
    # Clean up resources
    rag.close()
    print("\n✅ Done!")

if __name__ == "__main__":
    asyncio.run(main())
