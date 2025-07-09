"""
KSE Memory SDK Embedding Service
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import numpy as np
import logging
from ..core.interfaces import EmbeddingServiceInterface
from ..exceptions import EmbeddingError, ConfigurationError

logger = logging.getLogger(__name__)


class EmbeddingService(EmbeddingServiceInterface):
    """
    Embedding service for generating vector embeddings from text and other data types.
    Supports multiple embedding models and providers.
    """
    
    def __init__(self, config: Dict[str, Any], cache_service=None):
        """
        Initialize embedding service.
        
        Args:
            config: Configuration dictionary
            cache_service: Optional cache service for caching embeddings
        """
        self.config = config
        self.cache_service = cache_service
        self.models: Dict[str, Any] = {}
        
        # Handle both dict and dataclass config
        if hasattr(config, 'default_model'):
            self.default_model = config.default_model.value if hasattr(config.default_model, 'value') else str(config.default_model)
        else:
            self.default_model = config.get('default_model', 'openai')
        
        # Model configurations
        if hasattr(config, '__dict__'):
            # Dataclass config
            self.model_configs = {}
        else:
            # Dict config
            self.model_configs = config.get('models', {})
        
        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
        if hasattr(config, '__dict__'):
            # Dataclass config
            self.cache_enabled = getattr(config, 'cache_enabled', True)
            self.max_cache_size = getattr(config, 'max_cache_size', 10000)
        else:
            # Dict config
            self.cache_enabled = config.get('cache_enabled', True)
            self.max_cache_size = config.get('max_cache_size', 10000)
        
    async def initialize(self) -> bool:
        """Initialize embedding service."""
        try:
            # Initialize embedding models
            await self._initialize_models()
            
            logger.info("Embedding service initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            return False
    
    async def generate_text_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate text embedding using mock implementation."""
        try:
            # Mock implementation - generate deterministic embedding based on text hash
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            # Convert hash to 1536-dimensional vector (OpenAI embedding size)
            embedding = []
            for i in range(0, len(text_hash), 2):
                hex_pair = text_hash[i:i+2]
                value = int(hex_pair, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
                embedding.append(value)
            
            # Pad or truncate to 1536 dimensions
            while len(embedding) < 1536:
                embedding.extend(embedding[:min(len(embedding), 1536 - len(embedding))])
            embedding = embedding[:1536]
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            raise EmbeddingError(f"Text embedding generation failed: {e}")
    
    async def generate_image_embedding(self, image_url: str, model: Optional[str] = None) -> List[float]:
        """Generate image embedding using mock implementation."""
        try:
            # Mock implementation - generate deterministic embedding based on URL hash
            import hashlib
            url_hash = hashlib.md5(image_url.encode()).hexdigest()
            # Convert hash to 1536-dimensional vector
            embedding = []
            for i in range(0, len(url_hash), 2):
                hex_pair = url_hash[i:i+2]
                value = int(hex_pair, 16) / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
                embedding.append(value)
            
            # Pad or truncate to 1536 dimensions
            while len(embedding) < 1536:
                embedding.extend(embedding[:min(len(embedding), 1536 - len(embedding))])
            embedding = embedding[:1536]
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            raise EmbeddingError(f"Image embedding generation failed: {e}")
    
    async def batch_text_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = []
            for text in texts:
                embedding = await self.generate_text_embedding(text, model)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch text embeddings: {e}")
            raise EmbeddingError(f"Batch text embedding generation failed: {e}")
    
    async def batch_image_embeddings(self, image_urls: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for multiple images."""
        try:
            embeddings = []
            for image_url in image_urls:
                embedding = await self.generate_image_embedding(image_url, model)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate batch image embeddings: {e}")
            raise EmbeddingError(f"Batch image embedding generation failed: {e}")
    
    def get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about supported embedding models."""
        return {
            "mock": {
                "name": "Mock Embedding Model",
                "dimensions": 1536,
                "max_tokens": 8192,
                "description": "Mock embedding model for testing"
            },
            "openai": {
                "name": "OpenAI text-embedding-ada-002",
                "dimensions": 1536,
                "max_tokens": 8192,
                "description": "OpenAI's text embedding model"
            }
        }
    
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        supported_models = self.get_supported_models()
        if model in supported_models:
            return supported_models[model]
        else:
            raise EmbeddingError(f"Unsupported model: {model}")
    
    async def embed_text(self, text: str, model: Optional[str] = None) -> Optional[List[float]]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            model: Model to use (defaults to default_model)
            
        Returns:
            Embedding vector
        """
        try:
            model = model or self.default_model
            
            # Check cache first
            if self.cache_enabled:
                cache_key = f"{model}:{hash(text)}"
                if cache_key in self.embedding_cache:
                    return self.embedding_cache[cache_key].tolist()
            
            # Generate embedding
            if model == 'openai':
                embedding = await self._embed_openai(text)
            elif model == 'huggingface':
                embedding = await self._embed_huggingface(text)
            elif model == 'sentence_transformers':
                embedding = await self._embed_sentence_transformers(text)
            elif model == 'cohere':
                embedding = await self._embed_cohere(text)
            else:
                raise EmbeddingError(f"Unsupported embedding model: {model}")
            
            # Cache the result
            if self.cache_enabled and embedding is not None:
                cache_key = f"{model}:{hash(text)}"
                self.embedding_cache[cache_key] = np.array(embedding)
                
                # Manage cache size
                if len(self.embedding_cache) > self.max_cache_size:
                    # Remove oldest entries (simple FIFO)
                    keys_to_remove = list(self.embedding_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.embedding_cache[key]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise EmbeddingError(f"Text embedding failed: {e}", model=model)
    
    async def embed_batch(self, texts: List[str], model: Optional[str] = None) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            model: Model to use
            
        Returns:
            List of embedding vectors
        """
        try:
            model = model or self.default_model
            
            # Process in batches for efficiency
            batch_size = self.model_configs.get(model, {}).get('batch_size', 32)
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                if model == 'openai':
                    batch_embeddings = await self._embed_batch_openai(batch)
                elif model == 'huggingface':
                    batch_embeddings = await self._embed_batch_huggingface(batch)
                elif model == 'sentence_transformers':
                    batch_embeddings = await self._embed_batch_sentence_transformers(batch)
                elif model == 'cohere':
                    batch_embeddings = await self._embed_batch_cohere(batch)
                else:
                    # Fallback to individual embeddings
                    batch_embeddings = []
                    for text in batch:
                        embedding = await self.embed_text(text, model)
                        batch_embeddings.append(embedding)
                
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed batch: {e}")
            raise EmbeddingError(f"Batch embedding failed: {e}", model=model)
    
    async def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """
        Get the dimension of embeddings for a model.
        
        Args:
            model: Model name
            
        Returns:
            Embedding dimension
        """
        try:
            model = model or self.default_model
            
            # Return known dimensions
            dimensions = {
                'openai': 1536,  # text-embedding-ada-002
                'huggingface': 768,  # Default BERT-like models
                'sentence_transformers': 384,  # all-MiniLM-L6-v2
                'cohere': 4096  # embed-english-v3.0
            }
            
            return dimensions.get(model, 768)  # Default to 768
            
        except Exception as e:
            logger.error(f"Failed to get embedding dimension: {e}")
            return 768
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    async def _initialize_models(self) -> None:
        """Initialize embedding models."""
        try:
            # Initialize OpenAI if configured
            if 'openai' in self.model_configs:
                await self._initialize_openai()
            
            # Initialize HuggingFace if configured
            if 'huggingface' in self.model_configs:
                await self._initialize_huggingface()
            
            # Initialize Sentence Transformers if configured
            if 'sentence_transformers' in self.model_configs:
                await self._initialize_sentence_transformers()
            
            # Initialize Cohere if configured
            if 'cohere' in self.model_configs:
                await self._initialize_cohere()
                
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def _initialize_openai(self) -> None:
        """Initialize OpenAI embedding model."""
        try:
            import openai
            
            config = self.model_configs['openai']
            openai.api_key = config.get('api_key')
            
            self.models['openai'] = {
                'client': openai,
                'model_name': config.get('model_name', 'text-embedding-ada-002')
            }
            
        except ImportError:
            logger.warning("OpenAI library not available")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    
    async def _initialize_huggingface(self) -> None:
        """Initialize HuggingFace embedding model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            config = self.model_configs['huggingface']
            model_name = config.get('model_name', 'bert-base-uncased')
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            self.models['huggingface'] = {
                'tokenizer': tokenizer,
                'model': model,
                'model_name': model_name
            }
            
        except ImportError:
            logger.warning("HuggingFace transformers library not available")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace: {e}")
    
    async def _initialize_sentence_transformers(self) -> None:
        """Initialize Sentence Transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            config = self.model_configs['sentence_transformers']
            model_name = config.get('model_name', 'all-MiniLM-L6-v2')
            
            model = SentenceTransformer(model_name)
            
            self.models['sentence_transformers'] = {
                'model': model,
                'model_name': model_name
            }
            
        except ImportError:
            logger.warning("Sentence Transformers library not available")
        except Exception as e:
            logger.error(f"Failed to initialize Sentence Transformers: {e}")
    
    async def _initialize_cohere(self) -> None:
        """Initialize Cohere embedding model."""
        try:
            import cohere
            
            config = self.model_configs['cohere']
            api_key = config.get('api_key')
            
            client = cohere.Client(api_key)
            
            self.models['cohere'] = {
                'client': client,
                'model_name': config.get('model_name', 'embed-english-v3.0')
            }
            
        except ImportError:
            logger.warning("Cohere library not available")
        except Exception as e:
            logger.error(f"Failed to initialize Cohere: {e}")
    
    async def _embed_openai(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI."""
        try:
            if 'openai' not in self.models:
                return None
            
            model_info = self.models['openai']
            client = model_info['client']
            model_name = model_info['model_name']
            
            response = await client.Embedding.acreate(
                model=model_name,
                input=text
            )
            
            return response['data'][0]['embedding']
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return None
    
    async def _embed_huggingface(self, text: str) -> Optional[List[float]]:
        """Generate embedding using HuggingFace."""
        try:
            if 'huggingface' not in self.models:
                return None
            
            import torch
            
            model_info = self.models['huggingface']
            tokenizer = model_info['tokenizer']
            model = model_info['model']
            
            # Tokenize and encode
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling of last hidden states
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            return embeddings[0].tolist()
            
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            return None
    
    async def _embed_sentence_transformers(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Sentence Transformers."""
        try:
            if 'sentence_transformers' not in self.models:
                return None
            
            model_info = self.models['sentence_transformers']
            model = model_info['model']
            
            embedding = model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Sentence Transformers embedding failed: {e}")
            return None
    
    async def _embed_cohere(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Cohere."""
        try:
            if 'cohere' not in self.models:
                return None
            
            model_info = self.models['cohere']
            client = model_info['client']
            model_name = model_info['model_name']
            
            response = client.embed(
                texts=[text],
                model=model_name
            )
            
            return response.embeddings[0]
            
        except Exception as e:
            logger.error(f"Cohere embedding failed: {e}")
            return None
    
    async def _embed_batch_openai(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate batch embeddings using OpenAI."""
        try:
            if 'openai' not in self.models:
                return [None] * len(texts)
            
            model_info = self.models['openai']
            client = model_info['client']
            model_name = model_info['model_name']
            
            response = await client.Embedding.acreate(
                model=model_name,
                input=texts
            )
            
            return [item['embedding'] for item in response['data']]
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding failed: {e}")
            return [None] * len(texts)
    
    async def _embed_batch_huggingface(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate batch embeddings using HuggingFace."""
        try:
            embeddings = []
            for text in texts:
                embedding = await self._embed_huggingface(text)
                embeddings.append(embedding)
            return embeddings
            
        except Exception as e:
            logger.error(f"HuggingFace batch embedding failed: {e}")
            return [None] * len(texts)
    
    async def _embed_batch_sentence_transformers(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate batch embeddings using Sentence Transformers."""
        try:
            if 'sentence_transformers' not in self.models:
                return [None] * len(texts)
            
            model_info = self.models['sentence_transformers']
            model = model_info['model']
            
            embeddings = model.encode(texts)
            return [embedding.tolist() for embedding in embeddings]
            
        except Exception as e:
            logger.error(f"Sentence Transformers batch embedding failed: {e}")
            return [None] * len(texts)
    
    async def _embed_batch_cohere(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate batch embeddings using Cohere."""
        try:
            if 'cohere' not in self.models:
                return [None] * len(texts)
            
            model_info = self.models['cohere']
            client = model_info['client']
            model_name = model_info['model_name']
            
            response = client.embed(
                texts=texts,
                model=model_name
            )
            
            return response.embeddings
            
        except Exception as e:
            logger.error(f"Cohere batch embedding failed: {e}")
            return [None] * len(texts)