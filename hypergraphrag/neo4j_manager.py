"""Neo4j graph database management class"""
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import os
import json
from dotenv import load_dotenv

load_dotenv()


class Neo4jManager:
    """Neo4j graph database connection and management"""
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        vector_dimension: int = 384
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.vector_dimension = vector_dimension
        
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize graph schema with indexes"""
        with self.driver.session() as session:
            # Create indexes for entities
            for index_query in [
                "CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)",
                "CREATE INDEX ON :Entity(name)"
            ]:
                try:
                    session.run(index_query)
                    break
                except Exception:
                    continue
            
            # Create indexes for hyperedges
            for index_query in [
                "CREATE INDEX hyperedge_id_index IF NOT EXISTS FOR (h:Hyperedge) ON (h.id)",
                "CREATE INDEX ON :Hyperedge(id)"
            ]:
                try:
                    session.run(index_query)
                    break
                except Exception:
                    continue
            
            # Create index for Chunks (Vector Store Node)
            for index_query in [
                "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
                "CREATE INDEX ON :Chunk(id)"
            ]:
                try:
                    session.run(index_query)
                    break
                except Exception:
                    continue

            # Create index for Document (replacing Batch)
            for index_query in [
                "CREATE INDEX document_id_index IF NOT EXISTS FOR (d:Document) ON (d.id)",
                "CREATE INDEX ON :Document(id)"
            ]:
                try:
                    session.run(index_query)
                    break
                except Exception:
                    continue

            # Create Vector Index for Hyperedge
            try:
                session.run("""
                    CREATE VECTOR INDEX hyperedge_embedding_index IF NOT EXISTS
                    FOR (h:Hyperedge) ON (h.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                """, dim=self.vector_dimension)
            except Exception:
                pass

            # Create Vector Index for Entity
            try:
                session.run("""
                    CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
                    FOR (e:Entity) ON (e.embedding)
                    OPTIONS {indexConfig: {
                        `vector.dimensions`: $dim,
                        `vector.similarity_function`: 'cosine'
                    }}
                """, dim=self.vector_dimension)
            except Exception:
                pass

            # Create Chunk Full-Text Index for BM25 Search
            try:
                session.run("""
                    CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
                    FOR (c:Chunk) ON EACH [c.content]
                """)
            except Exception:
                pass
    
    def check_connection(self) -> Dict[str, Any]:
        """Check Neo4j connection and vector index dimensions"""
        status = {
            "connected": False,
            "indexes": {},
            "errors": []
        }
        
        try:
            with self.driver.session() as session:
                # 1. Basic Connection Check
                session.run("RETURN 1")
                status["connected"] = True

                # 2. Check Vector Index Dimensions
                # Note: SHOW INDEXES output format can vary by Neo4j version
                result = session.run("SHOW INDEXES YIELD name, type, options")
                for record in result:
                    if record["type"] == "VECTOR":
                        name = record["name"]
                        options = record["options"]
                        if options and "indexConfig" in options:
                            config = options["indexConfig"]
                            # Retrieve dimensions safely
                            dim = config.get("vector.dimensions")
                            if dim is not None:
                                status["indexes"][name] = int(dim)
        except Exception as e:
            status["errors"].append(str(e))
            
        return status
    
    def create_or_update_entity(
        self,
        entity_name: str,
        description: Optional[str] = None,
        vector: Optional[List[float]] = None
    ):
        """Create or update entity node"""
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.created_at = datetime(), e.chunk_ids = []
                SET e.description = COALESCE($description, e.description),
                    e.embedding = $vector,
                    e.updated_at = datetime()
            """, name=entity_name, description=description, vector=vector)
    
    def create_hyperedge(
        self,
        hyperedge_id: str,
        entity_names: List[str],
        content: str,
        chunk_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create hyperedge connecting multiple entities"""
        with self.driver.session() as session:
            # Create hyperedge node
            metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
            session.run("""
                MERGE (h:Hyperedge {id: $hyperedge_id})
                ON CREATE SET h.created_at = datetime()
                SET h.content = $content,
                    h.chunk_id = $chunk_id,
                    h.metadata = $metadata,
                    h.updated_at = datetime()
            """, 
                hyperedge_id=hyperedge_id,
                content=content,
                chunk_id=chunk_id,
                metadata=metadata_json
            )
            
            # Connect entities to hyperedge
            for entity_name in entity_names:
                session.run("""
                    MATCH (e:Entity {name: $entity_name})
                    MATCH (h:Hyperedge {id: $hyperedge_id})
                    MERGE (e)-[:PARTICIPATES_IN]->(h)
                """, entity_name=entity_name, hyperedge_id=hyperedge_id)
    
    def batch_create_entities(self, entities: List[Dict[str, Any]]):
        """Batch create or update entities with embeddings (Isolated per Document)"""
        if not entities:
            return
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $entities AS entity
                MATCH (d:Document {id: entity.doc_id})
                MERGE (e:Entity {name: entity.name, doc_id: entity.doc_id})
                MERGE (e)-[:FROM_DOCUMENT]->(d)
                ON CREATE SET 
                    e.created_at = datetime()
                SET e.description = COALESCE(entity.description, e.description),
                    e.embedding = entity.vector,
                    e.updated_at = datetime()
            """, entities=[
                {
                    "name": e["name"],
                    "description": e.get("description"),
                    "vector": e.get("vector"),
                    "doc_id": e["doc_id"]
                }
                for e in entities
            ])
    
    def batch_create_chunks(self, chunks: List[Dict[str, Any]]):
        """Batch create chunks (nodes only, no vector index)"""
        if not chunks:
            return

        with self.driver.session() as session:
            session.run("""
                UNWIND $chunks AS chunk
                MATCH (d:Document {id: chunk.doc_id})
                MERGE (c:Chunk {id: chunk.chunk_id})
                MERGE (c)-[:FROM_DOCUMENT]->(d)
                ON CREATE SET 
                    c.created_at = datetime()
                SET c.content = chunk.content,
                    c.metadata = chunk.metadata,
                    c.updated_at = datetime()
            """, chunks=[
                {
                    "chunk_id": c["chunk_id"],
                    "content": c["content"],
                    "metadata": json.dumps(c.get("metadata", {}), ensure_ascii=False) if isinstance(c.get("metadata"), dict) else c.get("metadata", "{}"),
                    "doc_id": c["doc_id"]
                }
                for c in chunks
            ])
    
    def batch_create_hyperedges(self, hyperedges: List[Dict[str, Any]]):
        """Batch create hyperedges"""
        if not hyperedges:
            return
        
        with self.driver.session() as session:
            # Create hyperedge nodes
            session.run("""
                UNWIND $hyperedges AS h
                MATCH (d:Document {id: h.doc_id})
                MERGE (hyperedge:Hyperedge {id: h.hyperedge_id, doc_id: h.doc_id})
                MERGE (hyperedge)-[:FROM_DOCUMENT]->(d)
                ON CREATE SET hyperedge.created_at = datetime()
                SET hyperedge.content = h.content,
                    hyperedge.chunk_id = h.chunk_id,
                    hyperedge.metadata = h.metadata,
                    hyperedge.embedding = h.vector,
                    hyperedge.updated_at = datetime()
            """, hyperedges=[
                {
                    "hyperedge_id": h["hyperedge_id"],
                    "content": h["content"],
                    "chunk_id": h.get("chunk_id"),
                    "metadata": json.dumps(h.get("metadata", {}), ensure_ascii=False),
                    "vector": h.get("vector"),
                    "doc_id": h["doc_id"]
                }
                for h in hyperedges
            ])
            
            # Connect entities to hyperedges
            # Entities must also match the doc_id to ensure we are connecting to the correct isolated entity
            for hyperedge in hyperedges:
                entity_names = hyperedge["entity_names"]
                hyperedge_id = hyperedge["hyperedge_id"]
                doc_id = hyperedge["doc_id"]
                
                session.run("""
                    UNWIND $entity_names AS entity_name
                    MATCH (e:Entity {name: entity_name, doc_id: $doc_id})
                    MATCH (h:Hyperedge {id: $hyperedge_id, doc_id: $doc_id})
                    MERGE (e)-[:PARTICIPATES_IN]->(h)
                """, doc_id=doc_id, entity_names=entity_names, hyperedge_id=hyperedge_id)
    
    def get_entities_by_hyperedge(self, hyperedge_id: str) -> List[Dict[str, Any]]:
        """Get all entities connected to a hyperedge"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)-[:PARTICIPATES_IN]->(h:Hyperedge {id: $hyperedge_id})
                RETURN e.name as name, e.description as description
            """, hyperedge_id=hyperedge_id)
            
            return [dict(record) for record in result]
    
    def get_hyperedges_by_entity(self, entity_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all hyperedges that an entity participates in (with limit)"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $entity_name})-[:PARTICIPATES_IN]->(h:Hyperedge)
                MATCH (other:Entity)-[:PARTICIPATES_IN]->(h)
                RETURN h.id as hyperedge_id, h.content as content, h.chunk_id as chunk_id, h.metadata as metadata, collect(other.name) as entity_names
                LIMIT $limit
            """, entity_name=entity_name, limit=limit)
            
            return [dict(record) for record in result]

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get chunks by their IDs"""
        if not chunk_ids:
            return []
            
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $chunk_ids AS chunk_id
                MATCH (c:Chunk {id: chunk_id})
                RETURN c.id as id, c.content as content, c.metadata as metadata, c.embedding as vector
            """, chunk_ids=chunk_ids)
            
            results = []
            for record in result:
                data = dict(record)
                # Deserialize metadata if it's a string
                if "metadata" in data and isinstance(data["metadata"], str):
                    try:
                        data["metadata"] = json.loads(data["metadata"])
                    except:
                        pass
                results.append(data)
            return results
    
    def get_chunks_by_entity(self, entity_name: str) -> List[str]:
        """Get chunk IDs that mention an entity (from Entity properties)"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $entity_name})
                RETURN e.chunk_ids as chunk_ids
            """, entity_name=entity_name)
            
            record = result.single()
            if record and record["chunk_ids"]:
                return record["chunk_ids"]
            return []
    
    def find_related_entities(
        self,
        entity_name: str,
        max_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """Find related entities through hyperedge connections"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (start:Entity {name: $entity_name})-[:PARTICIPATES_IN*1..$max_depth]-(end:Entity)
                WHERE start <> end
                RETURN DISTINCT end.name as name, 
                       end.description as description, length(path) as distance
                ORDER BY distance
                LIMIT 50
            """, entity_name=entity_name, max_depth=max_depth)
            
            return [dict(record) for record in result]
    
    def get_entities_by_chunk(self, chunk_id: str) -> List[str]:
        """Get entity names mentioned in a chunk (Scan entities)"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE $chunk_id IN e.chunk_ids
                RETURN e.name as entity_name
            """, chunk_id=chunk_id)
            
            return [record["entity_name"] for record in result]
    
    def search_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 10, 
        target_node: str = "Chunk"
    ) -> List[Dict[str, Any]]:
        """Search nodes by vector similarity"""
        index_name = "chunk_embedding_index"
        if target_node == "Entity":
            index_name = "entity_embedding_index"
        elif target_node == "Hyperedge":
            index_name = "hyperedge_embedding_index"
            
        with self.driver.session() as session:
            result = session.run(f"""
                CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
                YIELD node, score
                RETURN node, score
            """, index_name=index_name, k=top_k, query_vector=query_vector)
            
            results = []
            for record in result:
                node = record["node"]
                data = dict(node)
                # Deserialize metadata if it's a string
                if "metadata" in data and isinstance(data["metadata"], str):
                    try:
                        data["metadata"] = json.loads(data["metadata"])
                    except:
                        pass
                
                results.append({
                    **data,
                    "score": record["score"]
                })
            return results

    def get_top_documents_by_bm25(self, query_text: str, top_k: int = 3) -> List[str]:
        """
        BM25 search for Chunks to identify relevant Documents.
        """
        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.fulltext.queryNodes("chunk_fulltext", $query, {limit: 50}) 
                YIELD node as chunk, score
                
                MATCH (chunk)-[:FROM_DOCUMENT]->(d:Document)
                
                WITH d, sum(score) as doc_score
                ORDER BY doc_score DESC
                LIMIT $k
                
                RETURN d.id as doc_id
            """, query=query_text, k=top_k)
            
            return [record["doc_id"] for record in result]

    def search_vectors_with_document_bias(
        self, 
        query_vector: List[float], 
        top_k_global: int = 5,
        top_k_local: int = 5,
        focus_doc_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: Global Vector Search + Local (Document-biased) Vector Search
        """
        if not focus_doc_ids:
            return self.search_vectors(query_vector, top_k_global + top_k_local, "Entity")

        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('entity_embedding_index', $k_global, $query_vector)
                YIELD node, score
                RETURN node, score, "global" as source
                
                UNION
                
                CALL db.index.vector.queryNodes('entity_embedding_index', 100, $query_vector)
                YIELD node, score
                MATCH (node)
                WHERE node.doc_id IN $doc_ids
                RETURN node, score, "focused" as source
                LIMIT $k_local
            """, 
            k_global=top_k_global, 
            k_local=top_k_local, 
            query_vector=query_vector, 
            doc_ids=focus_doc_ids)
            
            results = []
            seen_names = set()
            
            raw_records = list(result)
            raw_records.sort(key=lambda x: x["score"], reverse=True)
            
            for record in raw_records:
                node = record["node"]
                name = node["name"]
                
                if name not in seen_names:
                    seen_names.add(name)
                    data = dict(node)
                    results.append({
                        **data,
                        "score": record["score"],
                        "source_strategy": record["source"]
                    })
            
            return results[:(top_k_global + top_k_local)]

    def get_best_hyperedges_with_entities(
        self,
        entity_name: str,
        query_vector: List[float],
        limit: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Find best hyperedges connected to an entity using vector similarity.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $name})-[:PARTICIPATES_IN]->(h:Hyperedge)
                WHERE h.embedding IS NOT NULL
                WITH h, vector.similarity.cosine(h.embedding, $query_vector) AS score
                ORDER BY score DESC
                LIMIT $limit
                
                MATCH (connected_end:Entity)-[:PARTICIPATES_IN]->(h)
                RETURN 
                    h.id as hyperedge_id, 
                    h.content as content, 
                    h.chunk_id as chunk_id, 
                    h.metadata as metadata, 
                    score,
                    collect(connected_end.name) as entity_names
            """, name=entity_name, query_vector=query_vector, limit=limit)
            
            return [dict(record) for record in result]

    def create_document_node(self, doc_id: str, tags: List[str] = None):
        """Create a Document node to track data lineage"""
        with self.driver.session() as session:
            session.run("""
                MERGE (d:Document {id: $doc_id})
                ON CREATE SET d.created_at = datetime(), d.tags = $tags
                ON MATCH SET d.updated_at = datetime(), d.tags = $tags
            """, doc_id=doc_id, tags=tags or [])

    def delete_document(self, doc_id: str):
        """
        Delete a document and all its associated data (Strategy 1: Complete Isolation).
        Since entities, chunks, and hyperedges are isolated per document, 
        we can simply delete everything connected to the Document node.
        """
        with self.driver.session() as session:
            # Delete Document node and all connected nodes (Chunk, Hyperedge, Entity)
            session.run("""
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (n)-[:FROM_DOCUMENT]->(d)
                DETACH DELETE d, n
            """, doc_id=doc_id)

    def reset_db(self):
        """Delete all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            # Drop indexes to ensure clean state
            try:
                session.run("DROP INDEX entity_name_index IF EXISTS")
                session.run("DROP INDEX hyperedge_id_index IF EXISTS")
                session.run("DROP INDEX chunk_id_index IF EXISTS")
                session.run("DROP INDEX chunk_embedding_index IF EXISTS")
                session.run("DROP INDEX entity_embedding_index IF EXISTS")
                session.run("DROP INDEX chunk_fulltext IF EXISTS")
                session.run("DROP INDEX batch_id_index IF EXISTS")
                session.run("DROP INDEX document_id_index IF EXISTS")
            except Exception:
                pass
            
            # Re-initialize schema
            self._initialize_schema()

    def close(self):
        """Close connection"""
        self.driver.close()
