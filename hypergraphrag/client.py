"""Hypergraph RAG main client class"""
from typing import List, Dict, Any, Optional, Set, Tuple, AsyncIterator
import hashlib
import asyncio
import time
import json
import math
import numpy as np
from numpy.linalg import norm
from .neo4j_manager import Neo4jManager
from .embedding import EmbeddingGenerator
from .text_processor import TextProcessor
from .llm_extractor import LLMExtractor
from .models import QueryResult, ChunkSearchResult, HyperedgeInfo
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
            dimension=embedding_dimension
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

    async def insert_data_stream(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 10,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Insert document data and yield progress updates using an iterator pattern.
        """
        if metadata is None:
            metadata = [{}] * len(documents)
        
        if len(metadata) != len(documents):
            raise ValueError("documents and metadata must have the same length")
        
        all_chunks = []
        
        # Step 1: Text chunking (fast, can be done sequentially)
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
        yield {"status": "chunking_complete", "total_chunks": len(all_chunks)}
        
        # Step 2: Extract entities and hyperedges AND Store in parallel batches (async pipeline)
        logger.debug("Starting extraction and storage pipeline...")
        
        # Create semaphore to limit concurrent DB writes to prevent deadlocks
        db_semaphore = asyncio.Semaphore(1)
        
        async def process_chunk_batch_and_store(
            batch_idx: int,
            chunk_batch: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """Process a batch of chunks: Extract -> Embed -> Store (Immediately)"""
            
            # Extract entities and hyperedges
            batch_entities, batch_hyperedges, batch_links = await self._extract_from_chunks(chunk_batch)
            
            # Generate embeddings
            chunk_embeddings, entity_embeddings, entity_data_list = await self._generate_batch_embeddings(
                chunk_batch, batch_entities
            )
            
            # Store immediately (with lock to prevent deadlocks)
            async with db_semaphore:
                await self._store_batch(
                    batch_entities, batch_hyperedges, batch_links,
                    chunk_batch, chunk_embeddings,
                    entity_data_list, entity_embeddings
                )
            
            return {
                "batch_idx": batch_idx,
                "chunks": len(chunk_batch),
                "entities": len(batch_entities),
                "hyperedges": len(batch_hyperedges)
            }

        
        # Process chunks in parallel batches using asyncio
        chunk_batches = [
            all_chunks[i:i + batch_size]
            for i in range(0, len(all_chunks), batch_size)
        ]
        
        total_batches = len(chunk_batches)
        
        # Create semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        async def process_with_semaphore(idx, batch):
            async with semaphore:
                return await process_chunk_batch_and_store(idx, batch)
        
        # Process all batches concurrently and yield results as they complete
        tasks = [process_with_semaphore(i, batch) for i, batch in enumerate(chunk_batches)]
        
        completed_batches = 0
        total_entities = 0
        total_hyperedges = 0
        total_chunks = 0
        
        for future in asyncio.as_completed(tasks):
            result = await future
            completed_batches += 1
            
            # Aggregate stats
            total_entities += result["entities"]
            total_hyperedges += result["hyperedges"]
            total_chunks += result["chunks"]
            
            yield {
                "status": "processing",
                "completed_batches": completed_batches,
                "total_batches": total_batches,
                "progress": completed_batches / total_batches * 100,
                "batch_result": result,
                "total_stats": {
                    "chunks": total_chunks,
                    "entities": total_entities,
                    "hyperedges": total_hyperedges
                }
            }
            
        yield {
            "status": "complete",
            "total_stats": {
                "chunks": total_chunks,
                "entities": total_entities,
                "hyperedges": total_hyperedges
            }
        }

    async def insert_data(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 10,
        max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS,
        show_progress: bool = True
    ) -> None:
        """
        Insert document data and create hypergraph structure with async batch processing.
        Displays a progress bar if tqdm is available and show_progress is True.
        """
        stream = self.insert_data_stream(
            documents, 
            metadata, 
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
                    pbar = tqdm(total=total_batches, desc="Processing Batches", unit="batch")
                
            elif status == "processing":
                total_batches = update["total_batches"]
                completed = update["completed_batches"]
                
                # Initialize pbar if not exists (fallback)
                if pbar is None and show_progress and tqdm:
                    pbar = tqdm(total=total_batches, desc="Processing Batches", unit="batch")
                
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
                    f"Successfully created {stats['chunks']} chunks, "
                    f"{stats['entities']} entities, and {stats['hyperedges']} hyperedges (Total)"
                )
        
        if pbar:
            pbar.close()

    async def query_local(
        self,
        query_text: str,
        top_n: int = 10,
        max_hops: int = 2
    ) -> QueryResult:
        """
        Local search: Entity-centric graph traversal with Path Scoring and Re-ranking.
        """
        logger.debug(f"Starting local search with max_hops={max_hops}, top_n={top_n}")
        
        # Step 1: Get initial entities from Neo4j (Vector Search)
        query_embedding = self.embedding.generate_embedding(query_text)
        top_entities = await self._search_initial_entities(query_embedding, top_n)
        
        # Step 2: Initialize data structures
        all_entities_found = set([e["name"] for e in top_entities])
        visited_entities = set([e["name"] for e in top_entities])
        # Initialize entity scores with similarity scores
        entity_scores: Dict[str, float] = {e["name"]: e["similarity"] for e in top_entities}
        
        visited_hyperedges: Set[str] = set()
        all_hyperedges: Dict[str, Dict[str, Any]] = {}
        
        # Step 3: 1-hop: Get hyperedges for top entities
        limit_per_entity = min(top_n * 5, 100)
        hop1_hyperedges = await self._get_hyperedges_for_entities(
            [e["name"] for e in top_entities],
            entity_scores,
            visited_hyperedges,
            all_hyperedges,
            top_n=top_n,
            limit_per_entity=limit_per_entity
        )
        logger.debug(f"Hop 1: Found {len(hop1_hyperedges)} hyperedges")
        
        # Step 4: Subsequent hops (if max_hops > 1)
        if max_hops > 1:
            await self._process_subsequent_hops(
                hop1_hyperedges,
                query_embedding,
                top_n,
                max_hops,
                visited_entities,
                entity_scores,
                visited_hyperedges,
                all_entities_found,
                all_hyperedges
            )
        
        # Step 5: Retrieve chunks and build result
        result_chunks = await self._retrieve_chunks_for_entities(
            all_entities_found,
            query_embedding,
            limit=top_n
        )
        hyperedge_info = self._convert_hyperedges_to_info(all_hyperedges)
        
        logger.debug(
            f"Local search completed: {len(result_chunks)} chunks, "
            f"{len(hyperedge_info)} hyperedges, {len(all_entities_found)} entities"
        )
        
        return QueryResult(
            query=query_text,
            top_chunks=result_chunks,
            hyperedges=hyperedge_info,
            expanded_chunks=[],
            total_chunks_found=len(result_chunks),
            total_hyperedges_found=len(hyperedge_info),
            total_expanded_chunks=0,
            entities_found=list(all_entities_found)
        )
    
    async def _extract_from_chunks(
        self,
        chunk_batch: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and hyperedges from a batch of chunks
        """
        batch_entities = {}
        batch_hyperedges = []
        batch_links = []
        
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
            
            entity_names = []
            for entity in entities:
                entity_name = entity.name
                entity_type = entity.type.value if hasattr(entity.type, "value") else str(entity.type)
                batch_entities[entity_name] = {
                    "name": entity_name,
                    "type": entity_type,
                    "description": entity.description
                }
                entity_names.append(entity_name)
            
            for hyperedge in hyperedges:
                entity_names_sorted = sorted(hyperedge.entity_names)
                # Create ID based on content and connected entities
                hyperedge_id = hashlib.md5(
                    f"{hyperedge.content}_{'_'.join(entity_names_sorted)}".encode()
                ).hexdigest()
                
                # Add any new entities found in hyperedges (implicitly created)
                for ent_name in hyperedge.entity_names:
                    if ent_name not in batch_entities:
                        batch_entities[ent_name] = {
                            "name": ent_name,
                            "type": "CONCEPT",
                            "description": "Extracted from hyperedge relationship"
                        }
                    if ent_name not in entity_names:
                        entity_names.append(ent_name)
                
                batch_hyperedges.append({
                    "hyperedge_id": hyperedge_id,
                    "entity_names": hyperedge.entity_names,
                    "content": hyperedge.content,
                    "metadata": {"chunk_id": chunk["id"], **chunk["metadata"]}
                })
            
            # Store chunk-entity links
            if entity_names:
                batch_links.append({
                    "chunk_id": chunk["id"],
                    "entity_names": entity_names
                })
        
        return batch_entities, batch_hyperedges, batch_links
    
    async def _generate_batch_embeddings(
        self,
        chunk_batch: List[Dict[str, Any]],
        batch_entities: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[List[float]], List[List[float]], List[Dict[str, Any]]]:
        """
        Generate embeddings for chunks and entities in a batch
        """
        chunk_contents = [chunk["content"] for chunk in chunk_batch]
        chunk_embeddings = await asyncio.to_thread(
            self.embedding.generate_embeddings,
            chunk_contents
        )
        
        entity_texts, entity_data_list = self._prepare_entity_embeddings(batch_entities)
        if entity_texts:
            entity_embeddings = await asyncio.to_thread(
                self.embedding.generate_embeddings,
                entity_texts
            )
        else:
            entity_embeddings = []
        
        return chunk_embeddings, entity_embeddings, entity_data_list
    
    async def _store_batch(
        self,
        batch_entities: Dict[str, Dict[str, Any]],
        batch_hyperedges: List[Dict[str, Any]],
        batch_links: List[Dict[str, Any]],
        chunk_batch: List[Dict[str, Any]],
        chunk_embeddings: List[List[float]],
        entity_data_list: List[Dict[str, Any]],
        entity_embeddings: List[List[float]]
    ) -> None:
        """Store extracted data to Neo4j (including vectors)"""
        import time
        start_time = time.time()
        
        # 1. Store entities with vectors
        t0 = time.time()
        # Merge vector data into batch_entities dict for Neo4j processing
        if batch_entities:
            # Create a map of entity_name -> embedding for easy lookup
            entity_embedding_map = {
                data["entity_name"]: embedding
                for data, embedding in zip(entity_data_list, entity_embeddings)
            }
            
            entities_to_store = []
            for name, info in batch_entities.items():
                info_with_vector = info.copy()
                if name in entity_embedding_map:
                    info_with_vector["vector"] = entity_embedding_map[name]
                entities_to_store.append(info_with_vector)
            
            await asyncio.to_thread(
                self.neo4j.batch_create_entities, 
                entities_to_store
            )
        t1 = time.time()
        logger.debug(f"[Profile] Store Entities ({len(batch_entities)}): {t1-t0:.4f}s")

        # 2. Store Chunks with vectors
        t0 = time.time()
        if chunk_batch:
            chunks_to_store = [
                {
                    "chunk_id": chunk["id"],
                    "content": chunk["content"],
                    "metadata": chunk["metadata"],
                    "vector": embedding
                }
                for chunk, embedding in zip(chunk_batch, chunk_embeddings)
            ]
            await asyncio.to_thread(
                self.neo4j.batch_create_chunks,
                chunks_to_store
            )
        t1 = time.time()
        logger.debug(f"[Profile] Store Chunks ({len(chunk_batch)}): {t1-t0:.4f}s")

        # 3. Store relationships
        t0 = time.time()
        if batch_links:
            await asyncio.to_thread(
                self.neo4j.batch_link_chunks_to_entities, 
                batch_links
            )
        t1 = time.time()
        logger.debug(f"[Profile] Store Links ({len(batch_links)}): {t1-t0:.4f}s")
        
        # 4. Store Hyperedges
        t0 = time.time()
        if batch_hyperedges:
            await asyncio.to_thread(
                self.neo4j.batch_create_hyperedges, 
                batch_hyperedges
            )
        t1 = time.time()
        logger.debug(f"[Profile] Store Hyperedges ({len(batch_hyperedges)}): {t1-t0:.4f}s")

        total_time = time.time() - start_time
        logger.debug(f"[Profile] Total Batch Store Time: {total_time:.4f}s")
    
    def _prepare_entity_embeddings(
        self,
        all_entities: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Prepare entity texts for embedding generation
        """
        entity_texts = []
        entity_data_list = []
        
        for entity_name, entity_info in all_entities.items():
            entity_text = self._create_entity_text(entity_name, entity_info.get("description"))
            entity_texts.append(entity_text)
            entity_data_list.append({
                "entity_name": entity_name,
                "entity_type": entity_info.get("type"),
                "description": entity_info.get("description")
            })
        
        return entity_texts, entity_data_list
    
    def _create_entity_text(self, entity_name: str, description: Optional[str] = None) -> str:
        """Create entity text for embedding (name + description)"""
        if description:
            return f"{entity_name}: {description}"
        return entity_name
    
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
                "type": ent.get("type"),
                "description": ent.get("description"),
                "similarity": ent["score"]
            }
            for ent in entity_results
        ]
    
    async def _get_hyperedges_for_entities(
        self,
        entity_names: List[str],
        entity_scores: Dict[str, float],
        visited_hyperedges: Set[str],
        all_hyperedges: Dict[str, Dict[str, Any]],
        top_n: int = 10,
        limit_per_entity: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get hyperedges for given entities (async) with Beam Search Pruning
        """
        hop_hyperedges: Dict[str, Dict[str, Any]] = {}
        
        # Process entities concurrently
        async def get_hyperedges_for_entity(entity_name: str):
            async with self.query_semaphore:
                hyperedges = await asyncio.to_thread(
                    self.neo4j.get_hyperedges_by_entity,
                    entity_name,
                    limit=limit_per_entity
                )
                return entity_name, hyperedges
        
        # Get hyperedges for all entities concurrently
        tasks = [get_hyperedges_for_entity(name) for name in entity_names]
        results = await asyncio.gather(*tasks)
        
        # Collect all unique hyperedges and their best source score
        all_hyperedge_data = {}  # hyperedge_id -> (hyperedge_data, max_source_score)
        
        for entity_name, hyperedges in results:
            source_score = entity_scores.get(entity_name, 0.0)
            
            for hyperedge in hyperedges:
                hyperedge_id = hyperedge["hyperedge_id"]
                if hyperedge_id not in visited_hyperedges:
                    if hyperedge_id not in all_hyperedge_data:
                        all_hyperedge_data[hyperedge_id] = (hyperedge, source_score)
                    else:
                        # Update max source score if we found a better path to this hyperedge
                        current_data, current_max = all_hyperedge_data[hyperedge_id]
                        if source_score > current_max:
                            all_hyperedge_data[hyperedge_id] = (current_data, source_score)
        
        # Beam Search: Sort by score and take top_n
        sorted_candidates = sorted(
            all_hyperedge_data.items(),
            key=lambda item: item[1][1],  # source_score
            reverse=True
        )
        top_candidates = sorted_candidates[:top_n]
        final_hyperedge_data = dict(top_candidates)
        
        # Get entities for selected hyperedges concurrently
        async def get_entities_for_hyperedge(hyperedge_id: str):
            async with self.query_semaphore:
                entities = await asyncio.to_thread(
                    self.neo4j.get_entities_by_hyperedge,
                    hyperedge_id
                )
                return hyperedge_id, entities
        
        # Process hyperedges and get their entities
        tasks = []
        for hyperedge_id, (hyperedge, source_score) in final_hyperedge_data.items():
            if hyperedge_id in visited_hyperedges:
                continue
            
            visited_hyperedges.add(hyperedge_id)
            tasks.append(get_entities_for_hyperedge(hyperedge_id))
            
        if tasks:
            results = await asyncio.gather(*tasks)
            
            for hyperedge_id, entities_in_hyperedge in results:
                # Retrieve original data
                hyperedge, source_score = final_hyperedge_data[hyperedge_id]
                
                entity_names_in_hyperedge = [e["name"] for e in entities_in_hyperedge]
                
                # Store hyperedge info with source score
                hyperedge_info = {
                    "hyperedge_id": hyperedge_id,
                    "entities": entity_names_in_hyperedge,
                    "content": hyperedge.get("content", ""),
                    "source_score": source_score  # Added for Path Scoring
                }
                hop_hyperedges[hyperedge_id] = hyperedge_info
                all_hyperedges[hyperedge_id] = hyperedge_info
        
        return hop_hyperedges
    
    async def _process_subsequent_hops(
        self,
        previous_hop_hyperedges: Dict[str, Dict[str, Any]],
        query_embedding: List[float],
        top_n: int,
        max_hops: int,
        visited_entities: Set[str],
        entity_scores: Dict[str, float],
        visited_hyperedges: Set[str],
        all_entities_found: Set[str],
        all_hyperedges: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Process subsequent hops in graph traversal
        """
        current_hop_hyperedges = previous_hop_hyperedges
        
        for hop in range(2, max_hops + 1):
            logger.debug(f"Hop {hop}: Exploring entities from previous hop's hyperedges...")
            
            # Get candidate entities and their parent scores from current hop's hyperedges
            candidate_entities, candidate_parent_scores = self._get_candidate_entities(
                current_hop_hyperedges,
                visited_entities
            )
            
            if not candidate_entities:
                logger.debug(f"No new entities to explore at hop {hop}")
                break
            
            logger.debug(f"Found {len(candidate_entities)} candidate entities (not visited)")
            
            # Select top entities based on Path Scoring
            selected_entities, new_scores = await self._select_top_entities_by_similarity(
                candidate_entities,
                candidate_parent_scores,
                query_embedding,
                top_n
            )
            
            if not selected_entities:
                logger.debug(f"No entities selected for hop {hop}")
                break
            
            logger.debug(f"Selected top {len(selected_entities)} entities for hop {hop}")
            
            # Update entity scores with new path scores
            entity_scores.update(new_scores)
            
            # Mark as visited and update sets
            visited_entities.update(selected_entities)
            all_entities_found.update(selected_entities)
            
            # Get hyperedges for selected entities
            limit_per_entity = min(top_n * 5, 100)
            current_hop_hyperedges = await self._get_hyperedges_for_entities(
                selected_entities,
                entity_scores,
                visited_hyperedges,
                all_hyperedges,
                top_n=top_n,
                limit_per_entity=limit_per_entity
            )
            
            logger.debug(f"Hop {hop}: Found {len(current_hop_hyperedges)} new hyperedges")
    
    def _get_candidate_entities(
        self,
        hyperedges: Dict[str, Dict[str, Any]],
        visited_entities: Set[str]
    ) -> Tuple[Set[str], Dict[str, float]]:
        """
        Get candidate entities from hyperedges, excluding visited ones
        """
        candidate_entities = set()
        candidate_parent_scores = {}
        
        for hyperedge_info in hyperedges.values():
            source_score = hyperedge_info.get("source_score", 0.0)
            
            for entity in hyperedge_info["entities"]:
                if entity not in visited_entities:
                    candidate_entities.add(entity)
                    # If entity is reached by multiple hyperedges, take the max parent score
                    if entity not in candidate_parent_scores or source_score > candidate_parent_scores[entity]:
                        candidate_parent_scores[entity] = source_score
                        
        return candidate_entities, candidate_parent_scores
    
    async def _select_top_entities_by_similarity(
        self,
        candidate_entities: Set[str],
        candidate_parent_scores: Dict[str, float],
        query_embedding: List[float],
        top_n: int
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        Select top N entities from candidates based on Path Scoring (async)
        Using Neo4j Vector Search instead of Qdrant
        """
        if not candidate_entities:
            return [], {}
        
        # Search in Neo4j and filter to candidates (async)
        search_limit = max(top_n * DEFAULT_ENTITY_SEARCH_MULTIPLIER, len(candidate_entities))
        entity_search_results = await asyncio.to_thread(
            self.neo4j.search_vectors,
            query_embedding,
            search_limit,
            target_node="Entity"
        )
        
        filtered_results = []
        for res in entity_search_results:
            name = res["name"] # Neo4j uses 'name'
            if name in candidate_entities:
                similarity = res["score"]
                parent_score = candidate_parent_scores.get(name, 0.0)
                
                # Calculate Path Score
                final_score = (
                    similarity * PATH_SCORE_SIMILARITY_WEIGHT + 
                    parent_score * PATH_SCORE_PARENT_WEIGHT
                )
                
                res["final_score"] = final_score
                res["entity_name"] = name # Map for compatibility
                filtered_results.append(res)
        
        # Sort by final score
        filtered_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Select top N
        top_results = filtered_results[:top_n]
        
        selected_entities = [r["entity_name"] for r in top_results]
        new_scores = {r["entity_name"]: r["final_score"] for r in top_results}
        
        return selected_entities, new_scores
    
    def _compute_cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec_a or not vec_b:
            return 0.0
        
        a = np.array(vec_a)
        b = np.array(vec_b)
        
        # Handle zero vectors
        norm_a = norm(a)
        norm_b = norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return float(np.dot(a, b) / (norm_a * norm_b))

    async def _retrieve_chunks_for_entities(
        self,
        entity_names: Set[str],
        query_embedding: List[float],
        limit: int = 10
    ) -> List[ChunkSearchResult]:
        """
        Retrieve, Score, and Rank chunks for given entities.
        Revised to use Neo4j for Chunk retrieval and scoring.
        """
        logger.debug("Retrieving and ranking chunks...")
        
        # Strategy:
        # 1. Get all Chunk IDs connected to found entities (Graph Traversal)
        # 2. Retrieve vectors for these chunks (Neo4j Search/Lookup)
        #    Note: Since Neo4j doesn't support "get vectors for these specific IDs" easily efficiently in one go
        #    without index lookup or extensive matching, and we need to score them.
        
        # Alternative Strategy (Simpler):
        # 1. Perform Vector Search on Chunks globally to find top relevant chunks
        # 2. Boost or filter by whether they are connected to the found entities?
        
        # Current Strategy Implementation (Mimicking previous logic):
        # 1. Find chunks connected to entities
        # 2. Calculate similarity manually (or use Neo4j to calc distance if we could query by ID)
        
        # Step 1: Get Chunks connected to entities
        # We can do this with a single Cypher query for efficiency, but let's stick to the manager pattern
        all_chunk_ids = set()
        
        async def get_chunks_for_entity(entity_name: str):
            async with self.query_semaphore:
                chunk_ids = await asyncio.to_thread(
                    self.neo4j.get_chunks_by_entity,
                    entity_name
                )
                return chunk_ids
        
        tasks = [get_chunks_for_entity(name) for name in entity_names]
        results = await asyncio.gather(*tasks)
        
        for chunk_ids in results:
            all_chunk_ids.update(chunk_ids)
        
        if not all_chunk_ids:
            logger.debug("No chunks found linked to entities.")
            # Fallback: Perform global vector search on chunks if no graph connections found?
            # For now, return empty as per original logic
            return []
            
        logger.debug(f"Found {len(all_chunk_ids)} candidate chunks.")
        
        # Optimization: Instead of fetching all vectors to python and calculating cosine similarity,
        # We can ask Neo4j to calculate similarity for these specific chunks.
        # However, retrieving vectors is also fine for reasonable batch sizes.
        
        # Let's fetch the chunks data (including vectors)
        # We'll need a method in Neo4jManager to "get chunks by ids"
        # Since we didn't add that specifically, let's add a helper logic here using cypher
        
        def fetch_chunk_data(c_ids):
             with self.neo4j.driver.session() as session:
                result = session.run("""
                    UNWIND $ids AS id
                    MATCH (c:Chunk {id: id})
                    RETURN c.id as chunk_id, c.content as content, c.embedding as vector, c.metadata as metadata
                """, ids=list(c_ids))
                return [dict(record) for record in result]

        chunk_data_list = await asyncio.to_thread(fetch_chunk_data, all_chunk_ids)
        
        ranked_chunks = []
        
        for data in chunk_data_list:
            chunk_vector = data.get("vector")
            if not chunk_vector:
                continue
                
            content = data.get("content", "")
            chunk_id = data.get("chunk_id")
            
            # Deserialize metadata
            metadata = data.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    pass
            
            relevance_score = self._compute_cosine_similarity(query_embedding, chunk_vector)
            entities_in_chunk = [] # Could populate this if needed
            
            ranked_chunks.append(ChunkSearchResult(
                chunk_id=chunk_id,
                content=content,
                score=relevance_score,
                entities=entities_in_chunk,
                metadata=metadata
            ))
        
        ranked_chunks.sort(key=lambda x: x.score, reverse=True)
        final_results = ranked_chunks[:limit]
        
        logger.debug(f"Returned top {len(final_results)} chunks after re-ranking.")
        return final_results
    
    def _convert_hyperedges_to_info(
        self,
        all_hyperedges: Dict[str, Dict[str, Any]]
    ) -> List[HyperedgeInfo]:
        """Convert hyperedge dictionaries to HyperedgeInfo objects"""
        return [
            HyperedgeInfo(
                hyperedge_id=he["hyperedge_id"],
                entities=he["entities"],
                content=he["content"]
            )
            for he in all_hyperedges.values()
        ]
    
    async def query(
        self,
        query_text: str,
        top_n: int = 10,
        max_hops: int = 2
    ) -> QueryResult:
        """
        Main query interface for Hypergraph RAG.
        """
        return await self.query_local(query_text, top_n=top_n, max_hops=max_hops)
    
    def reset_database(self):
        """Reset Neo4j graph (including vectors)"""
        logger.info("Resetting database...")
        self.neo4j.reset_db()
        logger.debug("Database reset successfully.")

    def close(self):
        """Clean up resources"""
        self.neo4j.close()
