"""Pydantic models for type safety and validation"""
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class Entity(BaseModel):
    """Entity model"""
    name: str = Field(..., description="Canonical name of the entity")
    description: Optional[str] = Field(None, description="Detailed description of the entity")
    
    class Config:
        use_enum_values = True
    
    def __hash__(self):
        """Make entity hashable for set operations"""
        return hash(self.name)


class Hyperedge(BaseModel):
    """Hyperedge model - connects multiple entities with shared knowledge"""
    entity_names: List[str] = Field(..., description="List of entity names connected by this hyperedge")
    content: str = Field(..., description="Knowledge content shared by these entities")
    hyperedge_id: Optional[str] = Field(None, description="Unique hyperedge identifier")
    chunk_id: Optional[str] = Field(None, description="Source chunk identifier")
    chunk: Optional["Chunk"] = Field(None, description="Source chunk associated with this hyperedge (populated during query)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    class Config:
        use_enum_values = True
    
    def __hash__(self):
        """Make hyperedge hashable for set operations"""
        return hash((tuple(sorted(self.entity_names)), self.content))


class ExtractedHyperedge(BaseModel):
    """
    Strict Hyperedge model for LLM extraction (Structured Outputs).
    Excludes fields like 'metadata' (Dict[str, Any]) which are not compatible with strict schemas.
    """
    entity_names: List[str] = Field(..., description="List of entity names connected by this hyperedge")
    content: str = Field(..., description="Knowledge content shared by these entities")
    attributes: List[str] = Field(default_factory=list, description="Contextual attributes like time, location, or manner")
    
    class Config:
        use_enum_values = True


class ChunkMetadata(BaseModel):
    """Chunk metadata model"""
    document_index: Optional[int] = Field(None, description="Index of the document this chunk belongs to")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk within the document")
    start_idx: Optional[int] = Field(None, description="Start index in the original document")
    end_idx: Optional[int] = Field(None, description="End index in the original document")
    source: Optional[str] = Field(None, description="Source of the document")
    category: Optional[str] = Field(None, description="Category of the document")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    
    class Config:
        extra = "allow"  # Allow additional fields


class Chunk(BaseModel):
    """Chunk model"""
    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk content")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")
    entities: Optional[List[str]] = Field(None, description="List of entity names mentioned in this chunk")
    
    class Config:
        extra = "allow"


class QueryResult(BaseModel):
    """Query search result model"""
    query: str = Field(..., description="Original query text")
    hyperedges: List[Hyperedge] = Field(default_factory=list, description="Related hyperedges")


class GraphExtractionResult(BaseModel):
    """Result of graph extraction containing entities and hyperedges"""
    entities: List[Entity] = Field(..., description="List of extracted entities")
    hyperedges: List[ExtractedHyperedge] = Field(..., description="List of extracted hyperedges")
