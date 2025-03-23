"""Service for AI model operations."""
import logging
import json
import re
from enum import Enum
from typing import List, Dict, Optional
from langchain_openai import AzureChatOpenAI
from langchain_cohere import CohereEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from dataclasses import dataclass
from langchain_azure_ai.embeddings import AzureAIEmbeddingsModel
from azure.ai.inference import EmbeddingsClient

from azure.core.credentials import AzureKeyCredential

from src.config.settings import AZURE_OPENAI_SETTINGS, VECTOR_STORE_SETTINGS

logger = logging.getLogger(__name__)

class LLMType(Enum):
    """Types of LLM models available."""
    INTELLIGENT = "intelligent"
    CHEAP = "cheap"

class AIModelsService:
    """Service for handling AI model operations."""
    
    def __init__(self):
        """Initialize the AI models service."""
        self._intelligent_llm = None
        self._cheap_llm = None
        self._embedding_model = None
    
    def _initialize_llm(self, intelligent_or_cheap: LLMType) -> AzureChatOpenAI:
        """Initialize and return an instance of AzureChatOpenAI LLM."""
        try:
            if intelligent_or_cheap == LLMType.INTELLIGENT:
                return AzureChatOpenAI(
                    deployment_name=AZURE_OPENAI_SETTINGS["intelligent_deployment"],
                    model=AZURE_OPENAI_SETTINGS["intelligent_model"],
                    api_key=AZURE_OPENAI_SETTINGS["api_key"],
                    azure_endpoint=AZURE_OPENAI_SETTINGS["api_endpoint"],
                    api_version=AZURE_OPENAI_SETTINGS["api_version"],
                    temperature=1,
                )
            elif intelligent_or_cheap == LLMType.CHEAP:
                return AzureChatOpenAI(
                    deployment_name=AZURE_OPENAI_SETTINGS["cheap_deployment"],
                    model=AZURE_OPENAI_SETTINGS["cheap_model"],
                    api_key=AZURE_OPENAI_SETTINGS["api_key"],
                    azure_endpoint=AZURE_OPENAI_SETTINGS["api_endpoint"],
                    api_version=AZURE_OPENAI_SETTINGS["api_version"],
                    temperature=0.9,
                )
            else:
                raise ValueError(f"Invalid LLM type: {intelligent_or_cheap}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def get_llm(self, intelligent_or_cheap: LLMType) -> AzureChatOpenAI:
        """Get the initialized LLM instance. If not initialized, initialize it first."""
        if intelligent_or_cheap == LLMType.INTELLIGENT:
            if self._intelligent_llm is None:
                self._intelligent_llm = self._initialize_llm(LLMType.INTELLIGENT)
            return self._intelligent_llm
        elif intelligent_or_cheap == LLMType.CHEAP:
            if self._cheap_llm is None:
                self._cheap_llm = self._initialize_llm(LLMType.CHEAP)
            return self._cheap_llm
        else:
            raise ValueError(f"Invalid LLM type: {intelligent_or_cheap}")

    
    
    def get_embedding_model(self):
        
        """Get the embedding model instance."""
        if self._embedding_model is None:
            try:

                # For Serverless API or Managed Compute endpoints

                self._embedding_model = EmbeddingsClient(

                    endpoint= VECTOR_STORE_SETTINGS["cohere_api_endpoint"],

                    credential=AzureKeyCredential(VECTOR_STORE_SETTINGS["cohere_api_key"]),
                    
                    model= VECTOR_STORE_SETTINGS["cohere_api_embedding_model"]

                )
                
                self._embedding_model = CohereEmbeddings(
                    model=VECTOR_STORE_SETTINGS["cohere_api_embedding_model"],
                    cohere_api_key=VECTOR_STORE_SETTINGS["cohere_api_key"],
                    base_url="https://rober-m87k7tom-eastus2.services.ai.azure.com/"
                )
                
                self._embedding_model = AzureAIEmbeddingsModel(
                    model_name=VECTOR_STORE_SETTINGS["cohere_api_embedding_model"],
                    credential=VECTOR_STORE_SETTINGS["cohere_api_key"],
                    endpoint= VECTOR_STORE_SETTINGS["cohere_api_endpoint"]
                )
                
                
                logger.info("Successfully initialized Cohere embeddings model")
            except Exception as e:
                logger.error(f"Failed to initialize Cohere embeddings model: {e}")
                raise
        return self._embedding_model
    
    def call_llm(self, prompt: str, llm_type: LLMType) -> str:
        """
        Call the LLM with a prompt and return the response content.
        
        Args:
            prompt: The prompt to send to the LLM
            llm_type: Type of LLM to use (INTELLIGENT or CHEAP)
            
        Returns:
            str: The response content from the LLM
        """
        try:
            logger.debug(f"Calling LLM ({llm_type.value}) with prompt:\n{prompt}")
            llm = self.get_llm(llm_type)
            response = llm.invoke(prompt)
            content = response.content.strip()
            
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response:\n{content}")
            
            # Remove JSON code block markers if present
            if "```" in content:
                logger.debug("Removing code block markers from response")
                content = content.replace("```json", "").replace("```markdown", "").replace("```plaintext", "").replace("```", "").strip()
                logger.debug(f"Response after removing markers:\n{content}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise
  
def test_emb_model():
    """Test the embedding model."""
    ai_models = AIModelsService()
    emb_model = ai_models.get_embedding_model()
    print(emb_model)
    
    hello_world_emb = emb_model.embed_documents(["Hello, world!"])
    print(hello_world_emb)

if __name__ == "__main__":
    test_emb_model()