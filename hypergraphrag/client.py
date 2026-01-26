"""Hypergraph RAG main client class"""
from typing import List, Dict, Any, Optional, Set, Tuple, AsyncIterator
import hashlib
import asyncio
import time
import json
import math
import uuid
from .neo4j_manager import Neo4jManager
from .embedding import EmbeddingGenerator
from .text_processor import TextProcessor
from .llm_extractor import LLMExtractor
from .models import QueryResult, Hyperedge
from .logger import setup_logger

logger = setup_logger("hypergraphrag.client")

# Optional tqdm for progress bar
try:
    from tqdm.asyncio import tqdm
except ImportError:
    tqdm = None

# Constants
DEFAULT_CHUNK_LIMIT = 100
DEFAULT_ENTITY_SEARCH_MULTIPLIER = 3
DEFAULT_MAX_CONCURRENT_TASKS = 10
NEO4J_QUERY_CONCURRENCY = 5
PATH_SCORE_SIMILARITY_WEIGHT = 0.7
PATH_SCORE_PARENT_WEIGHT = 0.3


class HyperGraphRAG:
    """Hypergraph RAG main client"""
    
    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_request_timeout: float = 120.0,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Initialize Hypergraph RAG client"""
        # Initialize Embedding first to get dimension
        self.embedding = EmbeddingGenerator(
            model_name=embedding_model,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
            dimension=embedding_dimension,
            timeout=llm_request_timeout
        )

        # Initialize Neo4j Manager with vector dimension
        self.neo4j = Neo4jManager(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            vector_dimension=self.embedding.dimension
        )
        
        self.text_processor = TextProcessor()
        self.llm_extractor = LLMExtractor(
            model=llm_model,
            api_key=llm_api_key,
            base_url=llm_base_url,
            timeout=llm_request_timeout
        )
        
        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Semaphore for query concurrency
        self.query_semaphore = asyncio.Semaphore(NEO4J_QUERY_CONCURRENCY)
    
    # =========================================================================
    # Public API Methods
    # =========================================================================

    def check_health(self) -> Dict[str, Any]:
        """
        Check database connection and configuration health.
        Returns detailed status including connection state and dimension compatibility.
        """
        status = self.neo4j.check_connection()
        current_dim = self.embedding.dimension
        is_healthy = status["connected"]
        messages = []
        
        if not status["connected"]:
            messages.append("Failed to connect to Neo4j.")
            if status["errors"]:
                messages.append(f"Error: {status['errors'][0]}")
            return {"healthy": False, "messages": messages, "details": status}

        if status["errors"]:
            is_healthy = False
            for err in status["errors"]:
                messages.append(f"Error during health check: {err}")

        for idx_name, idx_dim in status["indexes"].items():
            if idx_dim != current_dim:
                is_healthy = False
                messages.append(
                    f"Dimension mismatch for index '{idx_name}': "
                    f"DB has {idx_dim}, Client configured for {current_dim}."
                )
        
        if not messages:
            messages.append("Connection successful and configuration matches.")
            
        return {
            "healthy": is_healthy,
            "messages": messages,
            "details": status
        }

    async def add(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[str]] = None,
        batch_size: int = 10,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS,
        show_progress: bool = True
    ) -> List[str]:
        """
        Insert document data and create hypergraph structure with async batch processing.
        Returns the doc_ids.
        """
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
        elif len(doc_ids) != len(documents):
            raise ValueError("doc_ids length must match documents length")
            
        logger.info(f"Started ingestion for {len(documents)} documents")
        
        stream = self.add_stream(
            documents, 
            metadata, 
            doc_ids,
            batch_size, 
            max_concurrent_tasks
        )
        
        pbar = None
        async for update in stream:
            status = update.get("status")
            
            if status == "chunking_complete":
                logger.debug(f"Chunking complete: {update['total_chunks']} chunks created.")
                if show_progress and tqdm:
                    total_chunks = update['total_chunks']
                    total_batches = math.ceil(total_chunks / batch_size)
                    pbar = tqdm(total=total_batches, desc="Processing", unit="batch")
                
            elif status == "processing":
                total_batches = update["total_batches"]
                completed = update["completed_batches"]
                
                if pbar is None and show_progress and tqdm:
                    pbar = tqdm(total=total_batches, desc="Processing", unit="batch")
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        "Ent": update["total_stats"]["entities"],
                        "Edge": update["total_stats"]["hyperedges"]
                    })
                else:
                    logger.info(
                        f"Progress: {completed}/{total_batches} ({update['progress']:.1f}%) "
                        f"| Ent: {update['total_stats']['entities']}, Edge: {update['total_stats']['hyperedges']}"
                    )
            
            elif status == "complete":
                if pbar:
                    pbar.close()
                    pbar = None
                    
                stats = update["total_stats"]
                logger.info(
                    f"Ingestion complete. Created {stats['chunks']} chunks, "
                    f"{stats['entities']} entities, and {stats['hyperedges']} hyperedges."
                )
        
        if pbar:
            pbar.close()
            
        return doc_ids

    async def add_stream(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[str]] = None,
        batch_size: int = 10,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Insert document data and yield progress updates using an iterator pattern.
        """
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
        elif len(doc_ids) != len(documents):
            raise ValueError("doc_ids length must match documents length")
        
        for doc_id in doc_ids:
            self.neo4j.create_document_node(doc_id)
            
        all_chunks = self._chunk_documents(documents, metadata, doc_ids)
        yield {"status": "chunking_complete", "total_chunks": len(all_chunks)}
        
        logger.debug("Starting extraction and storage pipeline...")
        
        chunk_batches = [
            all_chunks[i:i + batch_size]
            for i in range(0, len(all_chunks), batch_size)
        ]
        
        total_batches = len(chunk_batches)
        
        db_semaphore = asyncio.Semaphore(1)
        task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        tasks = [
            self._process_batch_with_semaphore(
                i, batch, db_semaphore, task_semaphore
            ) 
            for i, batch in enumerate(chunk_batches)
        ]
        
        completed_batches = 0
        total_stats = {"chunks": 0, "entities": 0, "hyperedges": 0}
        
        for future in asyncio.as_completed(tasks):
            result = await future
            completed_batches += 1
            
            total_stats["entities"] += result["entities"]
            total_stats["hyperedges"] += result["hyperedges"]
            total_stats["chunks"] += result["chunks"]
            
            yield {
                "status": "processing",
                "completed_batches": completed_batches,
                "total_batches": total_batches,
                "progress": completed_batches / total_batches * 100,
                "batch_result": result,
                "total_stats": total_stats
            }
            
        yield {
            "status": "complete",
            "total_stats": total_stats
        }

    async def query(
        self,
        query_text: str,
        top_n: int = 10
    ) -> QueryResult:
        """
        Main query interface for Hypergraph RAG.
        Combines Local Search (Entity-centric) and Global Search (Direct Hyperedge).
        """
        local_n = top_n // 2
        global_n = top_n - local_n
        
        local_task = self._local_search(query_text, top_n=local_n)
        global_task = self._global_search(query_text, top_n=global_n)
        
        results = await asyncio.gather(local_task, global_task)
        local_result, global_result = results
        
        final_hyperedges = []
        seen_ids = set()
        
        for he in local_result.hyperedges:
            if he.hyperedge_id not in seen_ids:
                final_hyperedges.append(he)
                seen_ids.add(he.hyperedge_id)
        
        for he in global_result.hyperedges:
            if he.hyperedge_id not in seen_ids:
                final_hyperedges.append(he)
                seen_ids.add(he.hyperedge_id)
        
        return QueryResult(
            query=query_text,
            hyperedges=final_hyperedges
        )

    def delete(self, doc_id: str):
        """Delete a document and all its associated data (Rollback)."""
        logger.info(f"Deleting document {doc_id}...")
        self.neo4j.delete_document(doc_id)
        logger.info(f"Document {doc_id} deleted.")

    def reset_database(self):
        """Reset Neo4j graph (including vectors)"""
        logger.info("Resetting database...")
        self.neo4j.reset_db()
        logger.debug("Database reset successfully.")

    def close(self):
        """Clean up resources"""
        self.neo4j.close()

    # =========================================================================
    # Internal Orchestration Methods
    # =========================================================================

    async def _process_batch_with_semaphore(
        self,
        batch_idx: int,
        chunk_batch: List[Dict[str, Any]],
        db_semaphore: asyncio.Semaphore,
        task_semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Wrapper to process batch with semaphore"""
        async with task_semaphore:
            return await self._process_chunk_batch_and_store(batch_idx, chunk_batch, db_semaphore)

    async def _process_chunk_batch_and_store(
        self,
        batch_idx: int,
        chunk_batch: List[Dict[str, Any]],
        db_semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Process a batch of chunks: Extract -> Embed -> Store"""
        
        # Extract entities and hyperedges (handles doc_id internally)
        batch_entities, batch_hyperedges = await self._extract_from_chunks(chunk_batch)
        
        # Generate embeddings (Hyperedges instead of Chunks)
        hyperedge_embeddings, entity_embeddings = await self._generate_batch_embeddings(
            batch_hyperedges, batch_entities
        )
        
        # Store immediately (with lock to prevent deadlocks)
        async with db_semaphore:
            await self._store_batch(
                batch_entities, batch_hyperedges,
                chunk_batch, hyperedge_embeddings,
                entity_embeddings
            )
        
        return {
            "batch_idx": batch_idx,
            "chunks": len(chunk_batch),
            "entities": len(batch_entities),
            "hyperedges": len(batch_hyperedges)
        }

    async def _local_search(
        self,
        query_text: str,
        top_n: int = 10
    ) -> QueryResult:
        """
        Local search: Find best hyperedges connected to relevant entities using Hybrid Search.
        """
        logger.debug(f"Starting local search with top_n={top_n}")
        
        # Step 0: Identify relevant documents via BM25
        try:
            top_docs = await asyncio.to_thread(
                self.neo4j.get_top_documents_by_bm25,
                query_text,
                top_k=3 
            )
            logger.debug(f"Identified relevant documents via BM25: {top_docs}")
        except Exception as e:
            logger.warning(f"BM25 Document search failed, falling back to pure vector search: {e}")
            top_docs = []

        # Step 1: Hybrid Entity Search (Global Vector + Document Bias)
        query_embedding = self.embedding.generate_embedding(query_text)
        
        # Search for enough entities to likely yield results
        # We use a fixed reasonable number instead of multiplier logic
        # 10-20 entities is a good baseline for finding connected hyperedges
        k_entities = max(top_n * 2, 10)
        
        global_k = int(k_entities * 0.7)
        local_k = k_entities - global_k
        
        # Ensure minimum counts
        if global_k < 3: global_k = 3
        if top_docs and local_k < 3: local_k = 3

        top_entities = await self._search_initial_entities_hybrid(
            query_embedding, 
            global_k, 
            local_k, 
            top_docs
        )
        
        entities_found = [e["name"] for e in top_entities]
        
        # Step 2: Get BEST hyperedges for each entity
        # [Modified] Over-fetch to ensure diversity after dedup
        top_hyperedges_candidates = await self._find_best_hyperedges(
            entities_found, 
            query_embedding, 
            top_n=top_n * 3,     # Over-fetching
            limit_per_entity=10  # Increased limit
        )
        
        # Deduplicate by Chunk ID
        unique_hyperedges = []
        seen_chunk_ids = set()
        
        for he in top_hyperedges_candidates:
            # Fallback to hyperedge_id if chunk_id is missing
            dedup_key = he.get("chunk_id") or he["hyperedge_id"]
            
            if dedup_key not in seen_chunk_ids:
                seen_chunk_ids.add(dedup_key)
                unique_hyperedges.append(he)
            
            if len(unique_hyperedges) >= top_n:
                break
                
        top_hyperedges = unique_hyperedges
        logger.debug(f"Found {len(top_hyperedges)} unique best hyperedges (after chunk dedup)")

        # Step 3: Fetch chunks for these hyperedges (for content/metadata)
        chunks_map = await self._fetch_hyperedge_chunks(top_hyperedges)
        
        # Step 4: Construct Hyperedge objects
        final_hyperedges = self._construct_query_result(top_hyperedges, chunks_map)
            
        return QueryResult(
            query=query_text,
            hyperedges=final_hyperedges
        )

    async def _global_search(
        self,
        query_text: str,
        top_n: int = 10
    ) -> QueryResult:
        """
        Global search: Direct vector search on Hyperedge nodes.
        Finds hyperedges semantically similar to the query, regardless of entities.
        """
        logger.debug(f"Starting global search with top_n={top_n}")
        query_embedding = self.embedding.generate_embedding(query_text)
        
        results = await asyncio.to_thread(
            self.neo4j.search_vectors,
            query_vector=query_embedding,
            top_k=top_n,
            target_node="Hyperedge" # Search Hyperedge nodes directly
        )
        
        top_hyperedges = []
        for res in results:
            
            hyperedge_id = res.get("id")
            if not hyperedge_id: continue
            
            top_hyperedges.append({
                "hyperedge_id": hyperedge_id,
                "content": res.get("content"),
                "chunk_id": res.get("chunk_id"),
                "metadata": res.get("metadata"),
                "score": res.get("score"),
            })

        chunks_map = await self._fetch_hyperedge_chunks(top_hyperedges)

        async def fetch_entities(he_id):
            return await asyncio.to_thread(self.neo4j.get_entities_by_hyperedge, he_id)

        entity_tasks = [fetch_entities(he["hyperedge_id"]) for he in top_hyperedges]
        entity_results = await asyncio.gather(*entity_tasks)

        for i, he in enumerate(top_hyperedges):
            he["entity_names"] = [e["name"] for e in entity_results[i]]

        final_hyperedges = self._construct_query_result(top_hyperedges, chunks_map)
        
        return QueryResult(
            query=query_text,
            hyperedges=final_hyperedges
        )

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    def _chunk_documents(
        self, 
        documents: List[str], 
        metadata: Optional[List[Dict[str, Any]]],
        doc_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Split documents into chunks and attach metadata/doc_ids"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        if len(metadata) != len(documents):
            raise ValueError("documents and metadata must have the same length")
        
        all_chunks = []
        logger.debug("Chunking documents...")
        
        for doc_idx, (doc, doc_metadata, doc_id) in enumerate(zip(documents, metadata, doc_ids)):
            chunks = self.text_processor.chunk_text(
                doc,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            for chunk in chunks:
                chunk["metadata"] = {
                    **doc_metadata,
                    "document_index": doc_idx,
                    "chunk_index": chunk["chunk_index"],
                    "start_idx": chunk["start_idx"],
                    "end_idx": chunk["end_idx"]
                }
                chunk["doc_id"] = doc_id
            all_chunks.extend(chunks)
            
        logger.debug(f"Created {len(all_chunks)} chunks")
        return all_chunks

    async def _extract_from_chunks(
        self,
        chunk_batch: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and hyperedges from a batch of chunks.
        """
        # Parallel extraction
        extraction_tasks = [
            asyncio.to_thread(self.llm_extractor.extract_data, chunk["content"])
            for chunk in chunk_batch
        ]
        extraction_results = await asyncio.gather(*extraction_tasks)
        
        all_potential_entities = {}
        batch_hyperedges = []
        
        for idx, (entities, hyperedges) in enumerate(extraction_results):
            chunk = chunk_batch[idx]
            doc_id = chunk["doc_id"]
            
            # Collect potential entities
            for entity in entities:
                key = (entity.name, doc_id)
                all_potential_entities[key] = {
                    "name": entity.name,
                    "description": entity.description,
                    "doc_id": doc_id
                }
            
            # Process hyperedges
            for hyperedge in hyperedges:
                entity_names_sorted = sorted(hyperedge.entity_names)
                hyperedge_id = hashlib.md5(
                    f"{hyperedge.content}_{'_'.join(entity_names_sorted)}".encode()
                ).hexdigest()
                
                batch_hyperedges.append({
                    "hyperedge_id": hyperedge_id,
                    "entity_names": hyperedge.entity_names,
                    "content": hyperedge.content,
                    "chunk_id": chunk["id"],
                    "metadata": {**chunk.get("metadata", {}), **(hyperedge.metadata or {})},
                    "doc_id": doc_id
                })

        # Resolve entities (ensure referential integrity within doc scope)
        batch_entities = self._resolve_entities_for_hyperedges(
            batch_hyperedges, all_potential_entities
        )
        
        return batch_entities, batch_hyperedges

    def _resolve_entities_for_hyperedges(
        self,
        hyperedges: List[Dict[str, Any]],
        available_entities: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Filter and consolidate entities that are actually used in hyperedges."""
        resolved_entities = {}
        
        for h in hyperedges:
            doc_id = h["doc_id"]
            for name in h["entity_names"]:
                key = (name, doc_id)
                if key in available_entities:
                    resolved_entities[key] = available_entities[key]
                else:
                    # Implicit entity extraction from relationship
                    resolved_entities[key] = {
                        "name": name,
                        "description": "Extracted from hyperedge relationship",
                        "doc_id": doc_id
                    }
        return resolved_entities

    async def _generate_batch_embeddings(
        self,
        batch_hyperedges: List[Dict[str, Any]],
        batch_entities: Dict[Tuple[str, str], Dict[str, Any]]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate embeddings for hyperedges and entities in a batch"""
        # 1. Hyperedge embeddings
        hyperedge_contents = [h["content"] for h in batch_hyperedges]
        hyperedge_embeddings = []
        if hyperedge_contents:
            hyperedge_embeddings = await asyncio.to_thread(
                self.embedding.generate_embeddings,
                hyperedge_contents
            )
        
        # 2. Entity embeddings
        entity_texts = []
        for entity_info in batch_entities.values():
            name = entity_info["name"]
            desc = entity_info.get("description")
            text = f"{name}: {desc}" if desc else name
            entity_texts.append(text)

        entity_embeddings = []
        if entity_texts:
            entity_embeddings = await asyncio.to_thread(
                self.embedding.generate_embeddings,
                entity_texts
            )
        
        return hyperedge_embeddings, entity_embeddings

    async def _store_batch(
        self,
        batch_entities: Dict[Tuple[str, str], Dict[str, Any]],
        batch_hyperedges: List[Dict[str, Any]],
        chunk_batch: List[Dict[str, Any]],
        hyperedge_embeddings: List[List[float]],
        entity_embeddings: List[List[float]]
    ) -> None:
        """Store extracted data to Neo4j"""
        # 1. Entities
        if batch_entities:
            await self._store_entities(batch_entities, entity_embeddings)
        
        # 2. Chunks
        if chunk_batch:
            await self._store_chunks(chunk_batch)

        # 3. Hyperedges
        if batch_hyperedges:
            await self._store_hyperedges(batch_hyperedges, hyperedge_embeddings)

    async def _store_entities(
        self, 
        batch_entities: Dict[Tuple[str, str], Dict[str, Any]], 
        entity_embeddings: List[List[float]]
    ) -> None:
        """Helper to store entities"""
        entities_to_store = []
        # Values iteration order is preserved in Python 3.7+
        entity_values = list(batch_entities.values())
        
        if len(entity_values) == len(entity_embeddings):
            for i, info in enumerate(entity_values):
                info_with_vector = info.copy()
                info_with_vector["vector"] = entity_embeddings[i]
                entities_to_store.append(info_with_vector)
        else:
            logger.warning("Entity embedding count mismatch. Storing entities without vectors.")
            entities_to_store = list(batch_entities.values())
        
        await asyncio.to_thread(
            self.neo4j.batch_create_entities, 
            entities_to_store
        )

    async def _store_chunks(self, chunk_batch: List[Dict[str, Any]]) -> None:
        """Helper to store chunks"""
        chunks_to_store = [
            {
                "chunk_id": chunk["id"],
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {}),
                "doc_id": chunk["doc_id"]
            }
            for chunk in chunk_batch
        ]
        await asyncio.to_thread(
            self.neo4j.batch_create_chunks,
            chunks_to_store
        )

    async def _store_hyperedges(
        self, 
        batch_hyperedges: List[Dict[str, Any]], 
        hyperedge_embeddings: List[List[float]]
    ) -> None:
        """Helper to store hyperedges"""
        hyperedges_to_store = []
        for h, embedding in zip(batch_hyperedges, hyperedge_embeddings):
                h_with_vector = h.copy()
                h_with_vector["vector"] = embedding
                hyperedges_to_store.append(h_with_vector)

        await asyncio.to_thread(
            self.neo4j.batch_create_hyperedges,
            hyperedges_to_store
        )

    async def _search_initial_entities_hybrid(
        self,
        query_embedding: List[float],
        k_global: int,
        k_local: int,
        doc_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Search for initial entities using Hybrid approach"""
        entity_results = await asyncio.to_thread(
            self.neo4j.search_vectors_with_document_bias,
            query_vector=query_embedding,
            top_k_global=k_global,
            top_k_local=k_local,
            focus_doc_ids=doc_ids
        )
        
        return [
            {
                "name": ent["name"],
                "description": ent.get("description"),
                "similarity": ent["score"]
            }
            for ent in entity_results
        ]

    async def _find_best_hyperedges(
        self, 
        entity_names: List[str], 
        query_embedding: List[float],
        top_n: int,
        limit_per_entity: int = 1
    ) -> List[Dict[str, Any]]:
        """Find best hyperedges for each entity"""
        
        async def get_best_hyperedge(entity_name):
            async with self.query_semaphore:
                return await asyncio.to_thread(
                    self.neo4j.get_best_hyperedges_with_entities, 
                    entity_name,
                    query_embedding,
                    limit=limit_per_entity
                )
        
        tasks = [get_best_hyperedge(name) for name in entity_names]
        results = await asyncio.gather(*tasks)
        
        # Flatten and deduplicate
        all_hyperedges = []
        seen_ids = set()
        
        for res_list in results:
            for he in res_list:
                if he["hyperedge_id"] not in seen_ids:
                    all_hyperedges.append(he)
                    seen_ids.add(he["hyperedge_id"])
        
        # Sort by score
        all_hyperedges.sort(key=lambda x: x["score"], reverse=True)
        return all_hyperedges[:top_n]

    async def _fetch_hyperedge_chunks(
        self, 
        hyperedges: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch chunk data for hyperedges"""
        chunk_ids = set()
        for he in hyperedges:
            if he.get("chunk_id"):
                chunk_ids.add(he["chunk_id"])
        
        chunks_map = {}
        if chunk_ids:
            chunks_data = await asyncio.to_thread(
                self.neo4j.get_chunks_by_ids,
                list(chunk_ids)
            )
            for c in chunks_data:
                chunks_map[c["id"]] = c
        return chunks_map

    def _construct_query_result(
        self, 
        hyperedges: List[Dict[str, Any]], 
        chunks_map: Dict[str, Dict[str, Any]]
    ) -> List[Hyperedge]:
        """Construct Hyperedge objects with chunk data"""
        final_hyperedges = []
        for he in hyperedges:
            entity_names = he.get("entity_names", [])
            
            chunk_obj = None
            c_id = he.get("chunk_id")
            if c_id and c_id in chunks_map:
                c_data = chunks_map[c_id]
                c_meta = c_data.get("metadata", {})
                chunk_obj = {
                    "id": c_data["id"],
                    "content": c_data["content"],
                    "metadata": c_meta,
                    "entities": [] 
                }
            
            # Parse metadata
            meta = he.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    pass

            final_hyperedges.append(Hyperedge(
                hyperedge_id=he["hyperedge_id"],
                entity_names=entity_names,
                content=he["content"],
                chunk_id=c_id,
                chunk=chunk_obj,
                metadata=meta
            ))
        return final_hyperedges
