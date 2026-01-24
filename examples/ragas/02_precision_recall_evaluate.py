import asyncio
import os
import time
import warnings
import pickle
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Filter multiprocessing resource tracker warning
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from ragas import evaluate
from ragas.metrics import context_recall, context_precision
from ragas.run_config import RunConfig
from hypergraphrag import HyperGraphRAG
from datasets import Dataset

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TOP_N = 10
DATASET_PATH = "hotpot_qa_dataset.pkl"

async def main():
    print("🚀 Starting Ragas Evaluation...")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset file not found: {DATASET_PATH}")
        print("Please run 01_prepare_data.py first.")
        return

    # Load dataset
    print(f"📦 Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "rb") as f:
        subset = pickle.load(f)
    print(f"📊 Evaluated samples: {len(subset)}")

    # Initialize RAG Client (Will use .env settings for model)
    rag = HyperGraphRAG(
        chunk_size=768,
        chunk_overlap=50,
        llm_request_timeout=300.0
    )
    
    try:
        # 3. Query & Collect Results
        print("🔍 Querying and collecting results...")
        ragas_data = {
            "question": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for i, sample in enumerate(subset):
            question = sample["question"]
            supporting_facts = sample["supporting_facts"]
            context = sample["context"]
            
            # Prepare Ground Truth Sentences
            ground_truth_sentences = []
            context_map = dict(zip(context["title"], context["sentences"]))
            
            for title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"]):
                if title in context_map and sent_id < len(context_map[title]):
                    ground_truth_sentences.append(context_map[title][sent_id])
            
            if not ground_truth_sentences:
                continue
            
            # Retrieval
            result = await rag.query(
                query_text=question,
                top_n=TOP_N
            )

            retrieved_chunks = []
            for he in result.hyperedges:
                if he.chunk:
                    retrieved_chunks.append(he.chunk.content)
            
            # Take top K unique contents
            retrieved_contexts = list(dict.fromkeys(retrieved_chunks))
            ragas_data["question"].append(question)
            ragas_data["contexts"].append(retrieved_contexts)
            ragas_data["ground_truth"].append(ground_truth_sentences)
            
            if (i+1) % 5 == 0:
                print(f"   Processed {i+1}/{len(subset)} queries...")

    finally:
        rag.close()

    return ragas_data

if __name__ == "__main__":
    # 1. Collect Data (Async)
    ragas_data = asyncio.run(main())
    
    if not ragas_data or not ragas_data["question"]:
        print("❌ No data collected. Exiting.")
        exit(0)

    # 2. Run Ragas Evaluation (Sync context to avoid asyncio loop conflicts)
    print("\n🤖 Running Ragas Evaluation...")
    
    # Load configuration again for Sync context
    evaluator_llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL")
    )
            
    # Create Dataset
    import pandas as pd
    from datasets import Dataset
    
    df = pd.DataFrame(ragas_data)
    df["ground_truth"] = df["ground_truth"].apply(lambda x: "\n".join(x))
    ragas_dataset = Dataset.from_pandas(df)
    
    # Run evaluation
    results = evaluate(
        ragas_dataset,
        metrics=[
            context_recall,
            context_precision,
        ],
        llm=evaluator_llm,
        run_config=RunConfig(
            timeout=600
        )
    )
    
    print("\n" + "="*50)
    print("🏆 Ragas Evaluation Results")
    print("="*50)
    print(results)
    
    # Detailed view
    print("\n📄 Detailed Results per Sample:")
    res_df = results.to_pandas()
    
    # Ragas might rename 'question' to 'user_input' or just return metrics
    question_col = "question"
    if "question" not in res_df.columns and "user_input" in res_df.columns:
        question_col = "user_input"
        
    cols_to_show = [question_col, 'context_recall', 'context_precision']
    # Filter only existing columns just in case
    cols_to_show = [c for c in cols_to_show if c in res_df.columns]
    
    print(res_df[cols_to_show])

    # Save results to CSV
    output_file = "ragas_evaluation_results.csv"
    res_df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to {output_file}")
