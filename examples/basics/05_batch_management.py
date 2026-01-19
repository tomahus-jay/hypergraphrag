"""Example: Batch Management (Isolation and Rollback)"""
import asyncio
from hypergraphrag import HyperGraphRAG
from hypergraphrag.models import IngestionConfig

async def main():
    rag = HyperGraphRAG(chunk_size=500, chunk_overlap=50)
    rag.reset_database()
    print("Database reset successfully.")
    # --- Scenario 1: Add Batch A (Tech Context) ---
    documents_a = ["Apple released the new iPhone with advanced AI features."]
    config_a = IngestionConfig(batch_id="batch_tech_01", tags=["tech", "apple"])
    
    print(f"\n1️⃣ Adding Batch A (Tech): '{documents_a[0]}'")
    await rag.add(documents=documents_a, config=config_a)
    print(f"   ✅ Batch A ({config_a.batch_id}) added.")

    # --- Scenario 2: Add Batch B (Fruit Context) ---
    # Strategy 1 (Isolation) allows 'Apple' to exist as separate entities in different batches
    documents_b = ["Apple is a sweet and crunchy fruit rich in fiber."]
    config_b = IngestionConfig(batch_id="batch_fruit_01", tags=["food", "apple"])
    
    print(f"\n2️⃣ Adding Batch B (Fruit): '{documents_b[0]}'")
    await rag.add(documents=documents_b, config=config_b)
    print(f"   ✅ Batch B ({config_b.batch_id}) added.")
    print("   ℹ️  Note: 'Apple' entity in Batch A and Batch B are completely isolated.")

    # --- Scenario 3: Delete Batch A (Rollback) ---
    # print(f"\n3️⃣ Deleting Batch A ({config_a.batch_id})...")
    # rag.delete(config_a.batch_id)
    # print(f"   ✅ Batch A deleted successfully.")
    # print("   ℹ️  Batch B data remains intact.")

    # --- Verify: Search to see what remains ---
    query = "Apple"
    print(f"\n4️⃣ Verifying data by searching for '{query}'...")
    results = await rag.query(query, top_n=5)
    
    if results.hyperedges:
        print(f"   Found {len(results.hyperedges)} hyperedges related to '{query}':")
        for he in results.hyperedges:
            print(f"   - Content: {he.content}")
            if he.metadata:
                print(f"     Metadata: {he.metadata}") 
    else:
        print("   No results found (unexpected if Batch B remains).")

    rag.close()

if __name__ == "__main__":
    asyncio.run(main())
