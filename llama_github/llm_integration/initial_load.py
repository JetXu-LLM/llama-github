# initial_load.py
from threading import Lock
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch
from llama_github.config.config import config
from llama_github.logger import logger
from typing import Optional, Any

class LLMManager:
    _instance_lock = Lock()
    _instance = None
    llm = None
    embedding_model = None
    rerank_model = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # First check (unlocked)
            with cls._instance_lock:  # Acquire lock
                if cls._instance is None:  # Second check (locked)
                    cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 huggingface_token: Optional[str] = None,
                 open_source_models_hg_dir: Optional[str] = None,
                 embedding_model: Optional[str] =config.get("default_embedding"),
                 rerank_model: Optional[str] =config.get("default_reranker"),
                 llm: Any = None):
        with self._instance_lock:   # Prevent re-initialization
            if self._initialized:
                return
            self._initialized = True
        
        # Initialize for OpenAI GPT-4
        if llm is not None:
            self.llm = llm
            self.model_type = "Custom_langchain_llm"
        elif openai_api_key is not None and openai_api_key != "" and self.llm is None:
            logger.info("Initializing OpenAI API...")
            self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o")
            self.model_type = "OpenAI"
        # Initialize for Open Source Models
        elif open_source_models_hg_dir is not None and open_source_models_hg_dir != "" and self.llm is None:
            logger.info(f"Initializing {open_source_models_hg_dir}...")
            #load hugingface models
            self.model_type = "Hubgingface"
        elif self.llm is None:
            #default model is phi3_mini_128k
            self.model_type = "Hubgingface"

        #initial model_kwargs
        device="cpu"
        if torch.cuda.is_available():
            device="cuda"
        elif torch.backends.mps.is_available():
            device="mps"
        
        #initial embedding_model
        if self.embedding_model is None:
            model_kwargs = {'device': device, 'trust_remote_code':True}
            if (huggingface_token is not None and huggingface_token != ""):
                model_kwargs['token'] = huggingface_token
            encode_kwargs = {'normalize_embeddings': True}
            logger.info(f"Initializing {embedding_model}...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        #initial rerank_model
        if self.rerank_model is None:
            model_kwargs = {'device': device, 'trust_remote_code':True}
            logger.info(f"Initializing {rerank_model}...")
            self.rerank_model = HuggingFaceCrossEncoder(
                model_name=rerank_model,
                model_kwargs=model_kwargs,
            )   

    def get_llm(self):
        return self.llm
    
    def get_embedding_model(self):
        return self.embedding_model
    
    def get_rerank_model(self):
        return self.rerank_model