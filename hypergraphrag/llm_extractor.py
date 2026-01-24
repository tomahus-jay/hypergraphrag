"""LLM-based entity and hyperedge extraction"""
from typing import List, Optional, Tuple
import os
from dotenv import load_dotenv
from .prompts import (
    GRAPH_EXTRACTION_SYSTEM_PROMPT,
    GRAPH_EXTRACTION_USER_PROMPT_TEMPLATE
)
from .models import Entity, Hyperedge, GraphExtractionResult
from .logger import setup_logger

load_dotenv()

logger = setup_logger("hypergraphrag.llm_extractor")


class LLMExtractor:
    """Extract entities and hyperedges from text using LLM"""
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        **kwargs
    ):
        """
        Initialize LLM extractor
        
        Args:
            model: LLM model name (optional, defaults to LLM_MODEL env var)
            api_key: API key for LLM service (optional for self-hosted)
            base_url: Base URL for LLM service (for self-hosted or custom endpoints)
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters
        """
        # Prioritize env var if model is not explicitly provided, or fall back to default
        self.model = model or os.getenv("LLM_MODEL")
        
        # Get defaults from environment if not provided
        api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or "not-needed",  # Some self-hosted servers don't need API key
                base_url=base_url,
                timeout=timeout,
                **kwargs
            )
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

    def extract_data(self, text: str) -> Tuple[List[Entity], List[Hyperedge]]:
        """
        Extract entities and hyperedges from text using LLM in a single pass
        Uses OpenAI's Structured Outputs (response_format=json_schema) for strict schema adherence.
        
        Args:
            text: Text to extract from
        
        Returns:
            Tuple of (List[Entity], List[Hyperedge])
        """
        prompt = GRAPH_EXTRACTION_USER_PROMPT_TEMPLATE.format(text=text)
        messages = [
            {"role": "system", "content": GRAPH_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        try:
            # Use OpenAI Structured Outputs (beta.chat.completions.parse)
            # This enforces the schema defined in GraphExtractionResult Pydantic model
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=GraphExtractionResult,
            )
            
            message = completion.choices[0].message
            
            # Check for refusal
            if message.refusal:
                logger.warning(f"LLM Refusal: {message.refusal}")
                return [], []
            
            result = message.parsed
            
            # Check for parsing failure (result can be None if content is not valid JSON despite no refusal)
            if result is None:
                logger.error("Failed to parse LLM response (result is None)")
                return [], []
            
            # Since result is already a Pydantic object, we don't need manual JSON parsing or type conversion
            entities_data = result.entities
            hyperedges_data = result.hyperedges
            
            # Post-processing: Remove duplicates and normalize
            seen = set()
            unique_entities = []
            entity_name_map = {}  # lower -> canonical name
            
            for entity in entities_data:
                name_lower = entity.name.lower()
                if name_lower and name_lower not in seen:
                    seen.add(name_lower)
                    unique_entities.append(entity)
                    entity_name_map[name_lower] = entity.name
            
            # Post-processing: Normalize hyperedges and handle implicit entities
            normalized_hyperedges = []
            for hyperedge in hyperedges_data:
                # Validate and normalize entity names within hyperedge
                validated_names = []
                for name in hyperedge.entity_names:
                    name_clean = name.strip()
                    name_lower = name_clean.lower()
                    
                    if name_lower in entity_name_map:
                        validated_names.append(entity_name_map[name_lower])
                    else:
                        # Entity in hyperedge but not in extracted entities list
                        # Add it as a new entity implicitly
                        validated_names.append(name_clean)
                        
                        # Add to unique entities if not seen
                        if name_lower not in seen:
                            seen.add(name_lower)
                            entity_name_map[name_lower] = name_clean
                            # Create implicit entity object
                            new_entity = Entity(
                                name=name_clean,
                                description="Extracted from hyperedge relationship"
                            )
                            unique_entities.append(new_entity)
                
                # Only keep unique names to avoid self-loops
                validated_names = list(set(validated_names))
                
                if len(validated_names) >= 2:
                    # Convert ExtractedHyperedge to full Hyperedge model
                    new_hyperedge = Hyperedge(
                        entity_names=validated_names,
                        content=hyperedge.content,
                        metadata={"attributes": getattr(hyperedge, "attributes", [])}
                    )
                    normalized_hyperedges.append(new_hyperedge)
            
            return unique_entities, normalized_hyperedges
            
        except Exception as e:
            logger.error(f"Error extracting graph data with Structured Outputs: {e}", exc_info=True)
            return [], []
