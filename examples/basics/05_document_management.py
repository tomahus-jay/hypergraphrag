"""Example: Document Management (Isolation and Deletion)"""
import asyncio
from hypergraphrag import HyperGraphRAG

async def main():
    rag = HyperGraphRAG(chunk_size=500, chunk_overlap=50)
    
    # Optional: Reset database for clean state
    rag.reset_database()

    # --- Scenario 1: Add Document A (Tech Context) ---
    documents_a = ["Apple released the new iPhone with advanced AI features."]
    doc_id_a = "doc_tech_01"
    
    print(f"\n1️⃣ Adding Document A (Tech): '{documents_a[0]}'")
    await rag.add(documents=documents_a, doc_ids=[doc_id_a])
    print(f"   ✅ Document A ({doc_id_a}) added.")

    # --- Scenario 2: Add Document B (Fruit Context) ---
    # Strategy 1 (Isolation) allows 'Apple' to exist as separate entities in different documents
    # because entities are now keyed by (name, doc_id) internally or linked to specific docs.
    documents_b = ["Apple is a sweet and crunchy fruit rich in fiber."]
    doc_id_b = "doc_fruit_01"
    
    print(f"\n2️⃣ Adding Document B (Fruit): '{documents_b[0]}'")
    await rag.add(documents=documents_b, doc_ids=[doc_id_b])
    print(f"   ✅ Document B ({doc_id_b}) added.")
    print("   ℹ️  Note: 'Apple' entity in Document A and Document B are managed separately.")

    # --- Scenario 3: Verify Search finds both ---
    query = "Apple"
    print(f"\n3️⃣ Verifying data by searching for '{query}'...")
    results = await rag.query(query, top_n=10)
    
    print(f"   Found {len(results.hyperedges)} hyperedges related to '{query}':")
    for he in results.hyperedges:
        print(f"   - Content: {he.content}")
        if he.chunk and he.chunk.metadata:
             # The chunk metadata contains doc_id if we added it, 
             # but check if we can see which document it came from.
             print(f"     Source Doc Index/Meta: {he.chunk.metadata}") 

    # --- Scenario 4: Delete Document A ---
    print(f"\n4️⃣ Deleting Document A ({doc_id_a})...")
    rag.delete(doc_id_a)
    print(f"   ✅ Document A deleted successfully.")
    print("   ℹ️  Document B data should remain intact.")

    # --- Verify: Search to see what remains ---
    print(f"\n5️⃣ Verifying data by searching for '{query}' again...")
    results = await rag.query(query, top_n=10)
    
    if results.hyperedges:
        print(f"   Found {len(results.hyperedges)} hyperedges related to '{query}':")
        for he in results.hyperedges:
            print(f"   - Content: {he.content}")
    else:
        print("   No results found (unexpected if Document B remains).")

    rag.close()

if __name__ == "__main__":
    asyncio.run(main())
