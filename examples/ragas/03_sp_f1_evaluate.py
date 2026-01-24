import asyncio
import os
import pickle
import warnings
import pandas as pd
from typing import List, Dict, Set, Tuple, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from hypergraphrag import HyperGraphRAG

# Load environment variables
load_dotenv()

# Filter warnings
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
TOP_N = 10
DATASET_PATH = "hotpot_qa_dataset.pkl"
OUTPUT_PATH = "sp_f1_results_with_selection.csv"

# LLM for Sentence Selection
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o") # Use a smart model for reasoning
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

async def select_relevant_sentences(
    llm: ChatOpenAI, 
    question: str, 
    context_sentences: List[str]
) -> List[str]:
    """
    Ask LLM to select relevant sentences from the context.
    Returns a list of sentences that are deemed relevant.
    """
    if not context_sentences:
        return []
        
    # Simple numbering for the prompt
    numbered_context = "\n".join([f"{i+1}. {sent}" for i, sent in enumerate(context_sentences)])
    
    prompt = ChatPromptTemplate.from_template("""
Given a QUESTION and a list of CANDIDATE SENTENCES, identify the sentence indices that provide the necessary facts.

QUESTION: {question}

CANDIDATE SENTENCES:
{context}

INSTRUCTIONS:
1. Analyze the question to determine the key facts needed.
2. Look for sentences that contain these facts.
3. Also include sentences that provide essential context (e.g., resolving pronouns like "he" to a name).
4. IMPORTANT: If a sentence is partially relevant, INCLUDE IT. High recall is preferred.
5. OUTPUT FORMAT: Just the comma-separated numbers (e.g., "1, 3, 5"). No text, no explanation.
""")
    
    # Create a chain
    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({
            "question": question,
            "context": numbered_context
        })
        content = response.content.strip()
        
        # Simple parsing for comma-separated numbers
        selected_indices = []
        # Remove any brackets or text that might have leaked
        clean_content = "".join([c if c.isdigit() or c == "," else " " for c in content])
        
        for part in clean_content.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(context_sentences):
                    selected_indices.append(idx)
        
        # Deduplicate
        selected_indices = list(set(selected_indices))
        return [context_sentences[i] for i in selected_indices]
        
    except Exception as e:
        print(f"⚠️ LLM Selection Error: {e}")
        return [] # Return empty if failed (safe fallback to avoid noise)

async def main():
    print("🚀 Starting SP-F1 Evaluation with Sentence Selection...")
    
    # 1. Load Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset file not found: {DATASET_PATH}")
        print("Please run 01_prepare_data.py first.")
        return

    print(f"📦 Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "rb") as f:
        subset = pickle.load(f)
    print(f"📊 Evaluated samples: {len(subset)}")

    # 2. Build Global Context Map (Title -> Sentences)
    print("🗺️ Building global context map...")
    global_context_map: Dict[str, List[str]] = {}
    for sample in subset:
        context = sample["context"]
        for title, sentences in zip(context["title"], context["sentences"]):
            if title not in global_context_map:
                global_context_map[title] = sentences
    print(f"   Indexed {len(global_context_map)} unique documents.")

    # 3. Initialize Clients
    rag = HyperGraphRAG(
        chunk_size=768,
        chunk_overlap=50,
        llm_request_timeout=300.0
    )
    
    selector_llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.0,
        reasoning_effort="low"
    )

    results = []

    # Limit concurrency for LLM calls to avoid rate limits
    sem = asyncio.Semaphore(5)

    try:
        print("🔍 Querying and evaluating SP-F1 (This may take a while)...")
        
        async def process_sample(i, sample):
            async with sem:
                question = sample["question"]
                
                # --- Build Gold SP Set {(title, sent_id)} ---
                gold_sp: Set[Tuple[str, int]] = set()
                supporting_facts = sample["supporting_facts"]
                for title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"]):
                    gold_sp.add((title, sent_id))
                
                # --- Retrieval ---
                query_result = await rag.query(
                    query_text=question,
                    top_n=TOP_N
                )

                # --- Collect All Candidate Sentences from Chunks ---
                # Structure: candidates = [ {"title": t, "sent_idx": i, "text": "..."} ]
                candidates = []
                seen_sentences = set()

                for he in query_result.hyperedges:
                    if not he.chunk:
                        continue
                    
                    chunk_title = getattr(he.chunk.metadata, "title", None)
                    chunk_content = he.chunk.content
                    
                    if chunk_title in global_context_map:
                        doc_sentences = global_context_map[chunk_title]
                        for sent_idx, sent in enumerate(doc_sentences):
                            # Strict string inclusion check
                            if sent.strip() in chunk_content:
                                key = (chunk_title, sent_idx)
                                if key not in seen_sentences:
                                    candidates.append({
                                        "title": chunk_title,
                                        "sent_idx": sent_idx,
                                        "text": sent
                                    })
                                    seen_sentences.add(key)
                
                if not candidates:
                    return {
                        "question": question,
                        "precision": 0.0, "recall": 0.0, "sp_f1": 0.0,
                        "gold_count": len(gold_sp), "pred_count": 0, "tp": 0
                    }

                # --- LLM Sentence Selection ---
                candidate_texts = [c["text"] for c in candidates]
                selected_texts = await select_relevant_sentences(selector_llm, question, candidate_texts)
                
                # --- Build Predicted SP Set ---
                predicted_sp: Set[Tuple[str, int]] = set()
                
                # Map selected texts back to (title, id)
                # Note: This simple text matching assumes sentences are unique enough within the candidate set.
                # Since we deduped candidates, it should be fine.
                for text in selected_texts:
                    for cand in candidates:
                        if cand["text"] == text:
                            predicted_sp.add((cand["title"], cand["sent_idx"]))
                            break

                # --- Calculate Metrics ---
                tp = len(predicted_sp.intersection(gold_sp))
                fp = len(predicted_sp - gold_sp)
                fn = len(gold_sp - predicted_sp)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = calculate_f1(precision, recall)
                
                return {
                    "question": question,
                    "precision": precision,
                    "recall": recall,
                    "sp_f1": f1,
                    "gold_count": len(gold_sp),
                    "pred_count": len(predicted_sp),
                    "tp": tp
                }

        # Process in batches to show progress
        tasks = [process_sample(i, sample) for i, sample in enumerate(subset)]
        
        for i, future in enumerate(asyncio.as_completed(tasks)):
            res = await future
            results.append(res)
            
            if (i + 1) % 10 == 0:
                avg_f1 = sum(r["sp_f1"] for r in results) / len(results)
                print(f"   Processed {i+1}/{len(subset)} queries... Current Avg SP-F1: {avg_f1:.4f}")

    finally:
        rag.close()

    # 4. Summary and Save
    if not results:
        print("❌ No results collected.")
        return

    df = pd.DataFrame(results)
    
    avg_precision = df["precision"].mean()
    avg_recall = df["recall"].mean()
    avg_f1 = df["sp_f1"].mean()
    
    print("\n" + "="*50)
    print("🏆 SP-F1 Evaluation Results (with Sentence Selection)")
    print("="*50)
    print(f"Samples Evaluated: {len(df)}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall:    {avg_recall:.4f}")
    print(f"Average SP-F1:     {avg_f1:.4f}")
    print("="*50)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n💾 Detailed results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    asyncio.run(main())
