"""Embedding generation utility"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
import os
import logging
from dotenv import load_dotenv

# Optional imports handled inside classes
try:
    import numpy as np
except ImportError:
    np = None

load_dotenv()
logger = logging.getLogger("hypergraphrag.embedding")


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends"""
    
    @abstractmethod
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding vector dimension"""
        pass


class SentenceTransformersBackend(EmbeddingBackend):
    """Local Sentence Transformers backend"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required. "
                "Install with: pip install sentence-transformers"
            )
    
    def generate_embedding(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbeddingsBackend(EmbeddingBackend):
    """OpenAI Embeddings API backend"""
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        dimension: Optional[int] = None
    ):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or "not-needed",
                base_url=base_url,
                timeout=timeout
            )
            self.model = model
            
            # Default dimensions for OpenAI models
            self._dimension_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            
            if dimension:
                self._dimension = dimension
            else:
                self._dimension = self._dimension_map.get(model)
                if not self._dimension:
                    self._dimension = self._detect_dimension()
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

    def _detect_dimension(self) -> int:
        try:
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
        except Exception:
            return 1536
    
    def generate_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding API call failed: {e}\n{e.response.text}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            # OpenAI API has a limit on input tokens/count, but for simplicity we assume batch is handled by caller
            # or is small enough.
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding API call failed: {e}\n{e.response.text}")
    
    @property
    def dimension(self) -> int:
        return self._dimension


class HTTPEmbeddingsBackend(EmbeddingBackend):
    """HTTP API backend for self-hosted embedding servers"""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        dimension: Optional[int] = None
    ):
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests package is required. Install with: pip install requests")
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.headers = headers or {}
        self._dimension = dimension
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = "application/json"
        
        if not self._dimension:
            self._dimension = self._detect_dimension()
    
    def _detect_dimension(self) -> int:
        try:
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
        except Exception:
            return 384
    
    def generate_embedding(self, text: str) -> List[float]:
        # Try OpenAI-compatible endpoint first
        url = f"{self.base_url}/embeddings"
        payload = {
            "input": text,
            "model": self.model
        }
        
        try:
            response = self.requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0].get("embedding", [])
            elif "embedding" in result:
                return result["embedding"]
            elif isinstance(result, list):
                return result
            else:
                return self._try_alternative_endpoint(text)
                
        except self.requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request to embedding server failed: {e}\n{e.response.text}")
    
    def _try_alternative_endpoint(self, text: str) -> List[float]:
        for endpoint in ["/embed", "/encode", "/v1/embeddings"]:
            try:
                url = f"{self.base_url}{endpoint}"
                payload = {"text": text} if endpoint in ["/embed", "/encode"] else {"input": text}
                
                response = self.requests.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                
                if "embedding" in result:
                    return result["embedding"]
                elif isinstance(result, list):
                    return result
            except Exception:
                continue
        raise ValueError("Could not parse embedding response from server")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        payload = {
            "input": texts,
            "model": self.model
        }
        
        try:
            response = self.requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "data" in result:
                return [item.get("embedding", []) for item in result["data"]]
            elif isinstance(result, list):
                return result
            else:
                return [self.generate_embedding(text) for text in texts]
                
        except self.requests.exceptions.RequestException as e:
            raise RuntimeError(f"HTTP request to embedding server failed: {e}\n{e.response.text}")
    
    @property
    def dimension(self) -> int:
        if not self._dimension:
            self._dimension = self._detect_dimension()
        return self._dimension


class EmbeddingGenerator:
    """Text embedding generation with support for multiple backends"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimension: Optional[int] = None,
        backend: Optional[EmbeddingBackend] = None,
        **kwargs
    ):
        """
        Initialize embedding generator
        
        Args:
            model_name: Model name or path
            api_key: API key if required
            base_url: Base URL for API-based backends
            dimension: Embedding dimension (for HTTP backend)
            backend: Custom embedding backend instance (overrides settings)
            **kwargs: Additional provider-specific parameters
        """
        if backend:
            self.backend = backend
        else:
            # Get defaults from environment if not provided
            model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_API_KEY")
            base_url = base_url or os.getenv("EMBEDDING_BASE_URL")
            dimension = dimension or (int(os.getenv("EMBEDDING_DIMENSION")) if os.getenv("EMBEDDING_DIMENSION") else None)
            
            self.backend = self._create_backend(
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                dimension=dimension,
                **kwargs
            )
    
    def _create_backend(
        self,
        model_name: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        dimension: Optional[int],
        **kwargs
    ) -> EmbeddingBackend:
        
        # Determine provider based on parameters
        if base_url:
            # If base_url is provided, use HTTP backend (could be local or remote)
            return HTTPEmbeddingsBackend(
                base_url=base_url,
                model=model_name,
                api_key=api_key,
                timeout=kwargs.get("timeout", 60.0),
                headers=kwargs.get("headers"),
                dimension=dimension
            )
        elif api_key:
            # If API key is provided but no base_url, assume OpenAI
            model_name = model_name or "text-embedding-3-small"
            return OpenAIEmbeddingsBackend(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                timeout=kwargs.get("timeout", 60.0),
                dimension=dimension
            )
        else:
            # Default to local Sentence Transformers if no API details provided
            model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            return SentenceTransformersBackend(model_name=model_name)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        return self.backend.generate_embedding(text)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch"""
        return self.backend.generate_embeddings(texts)
    
    @property
    def dimension(self) -> int:
        """Return embedding vector dimension"""
        return self.backend.dimension
