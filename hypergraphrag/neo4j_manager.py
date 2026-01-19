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

            # Create index for Batch
            for index_query in [
                "CREATE INDEX batch_id_index IF NOT EXISTS FOR (b:Batch) ON (b.id)",
                "CREATE INDEX ON :Batch(id)"
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
    
    def batch_create_entities(self, entities: List[Dict[str, Any]], batch_id: str):
        """Batch create or update entities with embeddings (Isolated per batch)"""
        if not entities:
            return
        
        with self.driver.session() as session:
            session.run("""
                UNWIND $entities AS entity
                MATCH (b:Batch {id: $batch_id})
                MERGE (e:Entity {name: entity.name, batch_id: $batch_id})
                MERGE (e)-[:CREATED_IN]->(b)
                ON CREATE SET 
                    e.created_at = datetime()
                SET e.description = COALESCE(entity.description, e.description),
                    e.embedding = entity.vector,
                    e.updated_at = datetime()
            """, batch_id=batch_id, entities=[
                {
                    "name": e["name"],
                    "description": e.get("description"),
                    "vector": e.get("vector")
                }
                for e in entities
            ])
    
    def batch_create_chunks(self, chunks: List[Dict[str, Any]], batch_id: str):
        """Batch create chunks (nodes only, no vector index)"""
        if not chunks:
            return

        with self.driver.session() as session:
            session.run("""
                UNWIND $chunks AS chunk
                MATCH (b:Batch {id: $batch_id})
                MERGE (c:Chunk {id: chunk.chunk_id})
                MERGE (c)-[:CREATED_IN]->(b)
                ON CREATE SET 
                    c.created_at = datetime()
                SET c.content = chunk.content,
                    c.metadata = chunk.metadata,
                    c.updated_at = datetime()
            """, batch_id=batch_id, chunks=[
                {
                    "chunk_id": c["chunk_id"],
                    "content": c["content"],
                    "metadata": json.dumps(c.get("metadata", {}), ensure_ascii=False) if isinstance(c.get("metadata"), dict) else c.get("metadata", "{}")
                }
                for c in chunks
            ])
    
    def batch_create_hyperedges(self, hyperedges: List[Dict[str, Any]], batch_id: str):
        """Batch create hyperedges"""
        if not hyperedges:
            return
        
        with self.driver.session() as session:
            # Create hyperedge nodes
            session.run("""
                UNWIND $hyperedges AS h
                MATCH (b:Batch {id: $batch_id})
                MERGE (hyperedge:Hyperedge {id: h.hyperedge_id, batch_id: $batch_id})
                MERGE (hyperedge)-[:CREATED_IN]->(b)
                ON CREATE SET hyperedge.created_at = datetime()
                SET hyperedge.content = h.content,
                    hyperedge.chunk_id = h.chunk_id,
                    hyperedge.metadata = h.metadata,
                    hyperedge.embedding = h.vector,
                    hyperedge.updated_at = datetime()
            """, batch_id=batch_id, hyperedges=[
                {
                    "hyperedge_id": h["hyperedge_id"],
                    "content": h["content"],
                    "chunk_id": h.get("chunk_id"),
                    "metadata": json.dumps(h.get("metadata", {}), ensure_ascii=False),
                    "vector": h.get("vector")
                }
                for h in hyperedges
            ])
            
            # Connect entities to hyperedges
            for hyperedge in hyperedges:
                entity_names = hyperedge["entity_names"]
                hyperedge_id = hyperedge["hyperedge_id"]
                
                session.run("""
                    UNWIND $entity_names AS entity_name
                    MATCH (e:Entity {name: entity_name, batch_id: $batch_id})
                    MATCH (h:Hyperedge {id: $hyperedge_id, batch_id: $batch_id})
                    MERGE (e)-[:PARTICIPATES_IN]->(h)
                """, batch_id=batch_id, entity_names=entity_names, hyperedge_id=hyperedge_id)
    
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

    def create_batch_node(self, batch_id: str, tags: List[str] = None):
        """Create a Batch node to track data lineage"""
        with self.driver.session() as session:
            session.run("""
                MERGE (b:Batch {id: $batch_id})
                ON CREATE SET b.created_at = datetime(), b.tags = $tags
                ON MATCH SET b.updated_at = datetime(), b.tags = $tags
            """, batch_id=batch_id, tags=tags or [])

    def delete_batch(self, batch_id: str):
        """
        Delete a batch and all its associated data (Strategy 1: Complete Isolation).
        Since entities, chunks, and hyperedges are isolated per batch, 
        we can simply delete everything connected to the Batch node.
        """
        with self.driver.session() as session:
            # Delete Batch node and all connected nodes (Chunk, Hyperedge, Entity)
            # Strategy 1 ensures these nodes are exclusive to this batch.
            session.run("""
                MATCH (b:Batch {id: $batch_id})
                OPTIONAL MATCH (n)-[:CREATED_IN]->(b)
                DETACH DELETE b, n
            """, batch_id=batch_id)

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
            except Exception:
                pass
            
            # Re-initialize schema
            self._initialize_schema()

    def close(self):
        """Close connection"""
        self.driver.close()
