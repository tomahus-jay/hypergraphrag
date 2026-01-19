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
from .models import QueryResult, Hyperedge, IngestionConfig
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
        """
        Initialize Hypergraph RAG client
        """
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
        
        # Add logic to compare dimensions
        current_dim = self.embedding.dimension
        is_healthy = status["connected"]
        messages = []
        
        if not status["connected"]:
            messages.append("Failed to connect to Neo4j.")
            if status["errors"]:
                messages.append(f"Error: {status['errors'][0]}")
            return {"healthy": False, "messages": messages, "details": status}

        # Check for errors during index retrieval (even if connected)
        if status["errors"]:
            is_healthy = False
            for err in status["errors"]:
                messages.append(f"Error during health check: {err}")

        # Check dimensions
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
        config: Optional[IngestionConfig] = None,
        batch_size: int = 10,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS,
        show_progress: bool = True
    ) -> str:
        """
        Insert document data and create hypergraph structure with async batch processing.
        Returns the batch_id.
        """
        if config is None:
            config = IngestionConfig(batch_id=str(uuid.uuid4()), tags=[])
            
        # Create Batch Node
        self.neo4j.create_batch_node(config.batch_id, config.tags)
        logger.info(f"Started ingestion for batch: {config.batch_id}")
        
        stream = self.add_stream(
            documents, 
            metadata, 
            config,
            batch_size, 
            max_concurrent_tasks
        )
        
        pbar = None
            
        async for update in stream:
            status = update.get("status")
            
            if status == "chunking_complete":
                # Chunking done, we might want to log this but pbar is for batches
                logger.debug(f"Chunking complete: {update['total_chunks']} chunks created.")
                
                # Initialize pbar immediately after chunking
                if show_progress and tqdm:
                    total_chunks = update['total_chunks']
                    total_batches = math.ceil(total_chunks / batch_size)
                    pbar = tqdm(total=total_batches, desc=f"Batch {config.batch_id[:8]}", unit="batch")
                
            elif status == "processing":
                total_batches = update["total_batches"]
                completed = update["completed_batches"]
                
                # Initialize pbar if not exists (fallback)
                if pbar is None and show_progress and tqdm:
                    pbar = tqdm(total=total_batches, desc=f"Batch {config.batch_id[:8]}", unit="batch")
                
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix({
                        "Ent": update["total_stats"]["entities"],
                        "Edge": update["total_stats"]["hyperedges"]
                    })
                else:
                    # Fallback logging if no tqdm
                    logger.info(
                        f"Progress: {completed}/{total_batches} ({update['progress']:.1f}%) "
                        f"| Ent: {update['total_stats']['entities']}, Edge: {update['total_stats']['hyperedges']}"
                    )
            
            elif status == "complete":
                # Close pbar BEFORE logging completion
                if pbar:
                    pbar.close()
                    pbar = None
                    
                stats = update["total_stats"]
                logger.info(
                    f"Batch {config.batch_id} complete. Created {stats['chunks']} chunks, "
                    f"{stats['entities']} entities, and {stats['hyperedges']} hyperedges."
                )
        
        if pbar:
            pbar.close()
            
        return config.batch_id

    def delete(self, batch_id: str):
        """
        Delete a batch and all its associated data (Rollback).
        """
        logger.info(f"Deleting batch {batch_id}...")
        self.neo4j.delete_batch(batch_id)
        logger.info(f"Batch {batch_id} deleted.")

    async def add_stream(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        config: Optional[IngestionConfig] = None,
        batch_size: int = 10,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Insert document data and yield progress updates using an iterator pattern.
        """
        if config is None:
            config = IngestionConfig(batch_id=str(uuid.uuid4()), tags=[])
            # Ensure batch node exists if calling add_stream directly
            self.neo4j.create_batch_node(config.batch_id, config.tags)
            
        all_chunks = self._chunk_documents(documents, metadata)
        yield {"status": "chunking_complete", "total_chunks": len(all_chunks)}
        
        # Step 2: Extract entities and hyperedges AND Store in parallel batches (async pipeline)
        logger.debug("Starting extraction and storage pipeline...")
        
        chunk_batches = [
            all_chunks[i:i + batch_size]
            for i in range(0, len(all_chunks), batch_size)
        ]
        
        total_batches = len(chunk_batches)
        
        # Create semaphores
        db_semaphore = asyncio.Semaphore(1)
        task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        tasks = [
            self._process_batch_with_semaphore(
                i, batch, db_semaphore, task_semaphore, config.batch_id
            ) 
            for i, batch in enumerate(chunk_batches)
        ]
        
        completed_batches = 0
        total_stats = {"chunks": 0, "entities": 0, "hyperedges": 0}
        
        for future in asyncio.as_completed(tasks):
            result = await future
            completed_batches += 1
            
            # Aggregate stats
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
        """
        # max_hops is currently ignored in simplified local search
        return await self._local_search(query_text, top_n=top_n)

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
        task_semaphore: asyncio.Semaphore,
        batch_id: str
    ) -> Dict[str, Any]:
        """Wrapper to process batch with semaphore"""
        async with task_semaphore:
            return await self._process_chunk_batch_and_store(batch_idx, chunk_batch, db_semaphore, batch_id)

    async def _process_chunk_batch_and_store(
        self,
        batch_idx: int,
        chunk_batch: List[Dict[str, Any]],
        db_semaphore: asyncio.Semaphore,
        batch_id: str
    ) -> Dict[str, Any]:
        """Process a batch of chunks: Extract -> Embed -> Store"""
        
        # Extract entities and hyperedges
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
                entity_embeddings,
                batch_id
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
        Local search: Find best hyperedges connected to relevant entities using Vector Search.
        """
        logger.debug(f"Starting local search with top_n={top_n}")
        
        # Step 1: Get initial entities from Neo4j (Vector Search)
        query_embedding = self.embedding.generate_embedding(query_text)
        top_entities = await self._search_initial_entities(query_embedding, top_n)
        
        entities_found = [e["name"] for e in top_entities]
        
        # Step 2: Get BEST hyperedge for each entity
        top_hyperedges = await self._find_best_hyperedges(entities_found, query_embedding, top_n)
        logger.debug(f"Found {len(top_hyperedges)} unique best hyperedges")

        # Step 3: Fetch chunks for these hyperedges (for content/metadata)
        chunks_map = await self._fetch_hyperedge_chunks(top_hyperedges)
        
        # Step 4: Construct Hyperedge objects
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
        metadata: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Helper to chunk all documents"""
        if metadata is None:
            metadata = [{}] * len(documents)
        
        if len(metadata) != len(documents):
            raise ValueError("documents and metadata must have the same length")
        
        all_chunks = []
        logger.debug("Chunking documents...")
        
        for doc_idx, (doc, doc_metadata) in enumerate(zip(documents, metadata)):
            chunks = self.text_processor.chunk_text(
                doc,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk["metadata"] = {
                    **doc_metadata,
                    "document_index": doc_idx,
                    "chunk_index": chunk["chunk_index"],
                    "start_idx": chunk["start_idx"],
                    "end_idx": chunk["end_idx"]
                }
            all_chunks.extend(chunks)
            
        logger.debug(f"Created {len(all_chunks)} chunks")
        return all_chunks

    async def _extract_from_chunks(
        self,
        chunk_batch: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and hyperedges from a batch of chunks.
        Filters out entities that are not part of any hyperedge to prevent orphan nodes.
        """
        all_potential_entities = {} # All entities extracted by LLM
        batch_hyperedges = []
        
        # Process chunks concurrently using gather
        extraction_tasks = []
        for chunk in chunk_batch:
            extraction_tasks.append(
                asyncio.to_thread(
                    self.llm_extractor.extract_data,
                    chunk["content"]
                )
            )
        
        # Run extractions in parallel
        extraction_results = await asyncio.gather(*extraction_tasks)
        
        # Process results
        for idx, (entities, hyperedges) in enumerate(extraction_results):
            chunk = chunk_batch[idx]
            
            # 1. Collect all potential entities from this chunk
            for entity in entities:
                all_potential_entities[entity.name] = {
                    "name": entity.name,
                    "description": entity.description
                }
            
            # 2. Process hyperedges
            for hyperedge in hyperedges:
                entity_names_sorted = sorted(hyperedge.entity_names)
                # Create ID based on content and connected entities
                hyperedge_id = hashlib.md5(
                    f"{hyperedge.content}_{'_'.join(entity_names_sorted)}".encode()
                ).hexdigest()
                
                batch_hyperedges.append({
                    "hyperedge_id": hyperedge_id,
                    "entity_names": hyperedge.entity_names,
                    "content": hyperedge.content,
                    "chunk_id": chunk["id"],
                    "metadata": chunk.get("metadata", {})
                })

        # 3. Filter entities: Only keep those used in hyperedges
        used_entity_names = set()
        for h in batch_hyperedges:
            used_entity_names.update(h["entity_names"])
            
        batch_entities = {}
        for name in used_entity_names:
            if name in all_potential_entities:
                # Use extracted info (description)
                batch_entities[name] = all_potential_entities[name]
            else:
                # Implicit entity found in hyperedge but not in entity list
                batch_entities[name] = {
                    "name": name,
                    "description": "Extracted from hyperedge relationship"
                }
        
        return batch_entities, batch_hyperedges

    async def _generate_batch_embeddings(
        self,
        batch_hyperedges: List[Dict[str, Any]],
        batch_entities: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Generate embeddings for hyperedges and entities in a batch
        """
        hyperedge_contents = [h["content"] for h in batch_hyperedges]
        if hyperedge_contents:
            hyperedge_embeddings = await asyncio.to_thread(
                self.embedding.generate_embeddings,
                hyperedge_contents
            )
        else:
             hyperedge_embeddings = []
        
        # Prepare entity texts for embedding (inline logic, removed _prepare_entity_embeddings)
        entity_texts = []
        # entity_data_list removal: We rely on batch_entities ordering
        
        for entity_name, entity_info in batch_entities.items():
            # description is optional
            desc = entity_info.get("description")
            if desc:
                text = f"{entity_name}: {desc}"
            else:
                text = entity_name
                
            entity_texts.append(text)

        if entity_texts:
            entity_embeddings = await asyncio.to_thread(
                self.embedding.generate_embeddings,
                entity_texts
            )
        else:
            entity_embeddings = []
        
        return hyperedge_embeddings, entity_embeddings

    async def _store_batch(
        self,
        batch_entities: Dict[str, Dict[str, Any]],
        batch_hyperedges: List[Dict[str, Any]],
        chunk_batch: List[Dict[str, Any]],
        hyperedge_embeddings: List[List[float]],
        entity_embeddings: List[List[float]],
        batch_id: str
    ) -> None:
        """Store extracted data to Neo4j (including vectors for entities and hyperedges)"""
        import time
        start_time = time.time()
        
        # 1. Store entities with vectors
        t0 = time.time()
        if batch_entities:
            await self._store_entities(batch_entities, entity_embeddings, batch_id)
        t1 = time.time()
        logger.debug(f"[Profile] Store Entities ({len(batch_entities)}): {t1-t0:.4f}s")
        
        # 2. Store Chunks (NO vectors)
        t0 = time.time()
        if chunk_batch:
            await self._store_chunks(chunk_batch, batch_id)
        t1 = time.time()
        logger.debug(f"[Profile] Store Chunks ({len(chunk_batch)}): {t1-t0:.4f}s")

        # 3. Store Hyperedges with vectors
        t0 = time.time()
        if batch_hyperedges:
            await self._store_hyperedges(batch_hyperedges, hyperedge_embeddings, batch_id)
        t1 = time.time()
        logger.debug(f"[Profile] Store Hyperedges ({len(batch_hyperedges)}): {t1-t0:.4f}s")
        
        total_time = time.time() - start_time
        logger.debug(f"[Profile] Total Batch Store Time: {total_time:.4f}s")

    async def _store_entities(
        self, 
        batch_entities: Dict[str, Dict[str, Any]], 
        entity_embeddings: List[List[float]],
        batch_id: str
    ) -> None:
        """Helper to store entities"""
        entities_to_store = []
        entity_keys = list(batch_entities.keys())
        
        if len(entity_keys) == len(entity_embeddings):
            for i, key in enumerate(entity_keys):
                info = batch_entities[key]
                info_with_vector = info.copy()
                info_with_vector["vector"] = entity_embeddings[i]
                entities_to_store.append(info_with_vector)
        else:
            # Fallback: Store without vectors if count mismatch (should not happen)
            logger.warning("Entity embedding count mismatch. Storing entities without vectors.")
            entities_to_store = list(batch_entities.values())
        
        await asyncio.to_thread(
            self.neo4j.batch_create_entities, 
            entities_to_store,
            batch_id
        )

    async def _store_chunks(self, chunk_batch: List[Dict[str, Any]], batch_id: str) -> None:
        """Helper to store chunks"""
        chunks_to_store = [
            {
                "chunk_id": chunk["id"],
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {})
            }
            for chunk in chunk_batch
        ]
        await asyncio.to_thread(
            self.neo4j.batch_create_chunks,
            chunks_to_store,
            batch_id
        )

    async def _store_hyperedges(
        self, 
        batch_hyperedges: List[Dict[str, Any]], 
        hyperedge_embeddings: List[List[float]],
        batch_id: str
    ) -> None:
        """Helper to store hyperedges"""
        hyperedges_to_store = []
        for h, embedding in zip(batch_hyperedges, hyperedge_embeddings):
                h_with_vector = h.copy()
                h_with_vector["vector"] = embedding
                hyperedges_to_store.append(h_with_vector)

        await asyncio.to_thread(
            self.neo4j.batch_create_hyperedges,
            hyperedges_to_store,
            batch_id
        )

    async def _search_initial_entities(
        self,
        query_embedding: List[float],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """Search for initial entities in Neo4j (Vector Search)"""
        logger.debug("Searching for similar entities in Neo4j...")
        entity_results = await asyncio.to_thread(
            self.neo4j.search_vectors,
            query_embedding,
            top_n,
            target_node="Entity"
        )
        
        logger.debug(f"Found {len(entity_results)} entities from Neo4j search")
        for i, ent in enumerate(entity_results[:5], 1):
            logger.debug(f"  {i}. {ent['name']} (similarity: {ent['score']:.4f})")
        
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
        top_n: int
    ) -> List[Dict[str, Any]]:
        """Find best hyperedge for each entity"""
        
        async def get_best_hyperedge(entity_name):
            async with self.query_semaphore:
                return await asyncio.to_thread(
                    self.neo4j.get_best_hyperedges_with_entities, 
                    entity_name,
                    query_embedding,
                    limit=1
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
