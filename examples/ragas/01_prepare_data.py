import asyncio
import os
import warnings
import json
import pickle
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Filter multiprocessing resource tracker warning
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from datasets import load_dataset
from hypergraphrag import HyperGraphRAG

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
SAMPLE_SIZE = 1000
DATASET_PATH = "hotpot_qa_dataset.pkl"

async def main():
    print("🚀 Starting HotpotQA Data Preparation...")
    
    # 1. Load HotpotQA Dataset
    print("📦 Loading HotpotQA dataset (distractor)...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # Select a subset
    subset = dataset.select(range(SAMPLE_SIZE))
    print(f"📊 Evaluated samples: {len(subset)}")
    
    # 2. Bulk Insert Data (Simulate realistic retrieval environment)
    print("preparing documents for bulk insert...")
    all_documents = []
    all_metadatas = []
    seen_titles = set()

    # Collect ALL contexts from the subset
    for sample in subset:
        context = sample["context"] # {'title': [], 'sentences': []}
        for title, sentences in zip(context["title"], context["sentences"]):
            if title not in seen_titles:
                doc_content = f"{title}\n" + " ".join(sentences)
                all_documents.append(doc_content)
                all_metadatas.append({"title": title, "source": "hotpot_qa"})
                seen_titles.add(title)
    
    print(f"📝 Bulk Inserting {len(all_documents)} unique documents...")
    
    # Save the dataset subset for evaluation step
    with open(DATASET_PATH, "wb") as f:
        pickle.dump(subset, f)
    print(f"💾 Dataset subset saved to {DATASET_PATH}")

    # Initialize RAG Client (Will use .env settings for model)
    rag = HyperGraphRAG(
        chunk_size=768,
        chunk_overlap=50,
        llm_request_timeout=300.0  # Increase timeout to 5 minutes to avoid timeouts
    )
    
    try:
        # Reset DB once
        rag.reset_database()
        
        # Bulk Insert
        # Lower concurrency to avoid overloading the LLM server/API
        await rag.add(
            documents=all_documents, 
            metadata=all_metadatas,
            batch_size=5,
            max_concurrent_tasks=10
        )
        print("✅ Data insertion completed successfully.")

    finally:
        rag.close()

if __name__ == "__main__":
    asyncio.run(main())
