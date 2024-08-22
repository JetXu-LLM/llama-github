# initial_load.py
import torch
from typing import Optional, Any
from threading import Lock
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_mistralai.chat_models import ChatMistralAI

from llama_github.config.config import config
from llama_github.logger import logger

from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class LLMManager:
    _instance_lock = Lock()
    _instance = None
    llm = None
    # embedding_model = None
    rerank_model = None
    _initialized = False
    llm_simple = None
    tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # First check (unlocked)
            with cls._instance_lock:  # Acquire lock
                if cls._instance is None:  # Second check (locked)
                    cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 mistral_api_key: Optional[str] = None,
                 huggingface_token: Optional[str] = None,
                 open_source_models_hg_dir: Optional[str] = None,
                 embedding_model: Optional[str] = config.get(
                     "default_embedding"),
                 rerank_model: Optional[str] = config.get("default_reranker"),
                 llm: Any = None):
        with self._instance_lock:   # Prevent re-initialization
            if self._initialized:
                return
            self._initialized = True

        # Initialize for OpenAI GPT-4
        if llm is not None:
            self.llm = llm
            self.model_type = "Custom_langchain_llm"
        elif mistral_api_key is not None and mistral_api_key != "" and self.llm is None:
            logger.info("Initializing Mistral API...")
            self.llm = ChatMistralAI(mistral_api_key=mistral_api_key, model="mistral-large-latest")
            self.llm_simple = ChatMistralAI(mistral_api_key=mistral_api_key, model="open-mistral-nemo")
            self.model_type = "OpenAI"
        elif openai_api_key is not None and openai_api_key != "" and self.llm is None:
            logger.info("Initializing OpenAI API...")
            self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo")
            self.llm_simple = ChatOpenAI(
                api_key=openai_api_key, model="gpt-4o-mini")
            self.model_type = "OpenAI"
        # Initialize for Open Source Models
        elif open_source_models_hg_dir is not None and open_source_models_hg_dir != "" and self.llm is None:
            logger.info(f"Initializing {open_source_models_hg_dir}...")
            # load hugingface models
            self.model_type = "Hubgingface"
        elif self.llm is None:
            # default model is phi3_mini_128k
            self.model_type = "Hubgingface"

        # initial model_kwargs
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        # initial embedding_model
        if self.tokenizer is None:
            logger.info(f"Initializing {embedding_model}...")
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            self.embedding_model = AutoModel.from_pretrained(
                embedding_model, trust_remote_code=True).to(self.device)

        # initial rerank_model
        if self.rerank_model is None:
            logger.info(f"Initializing {rerank_model}...")
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                rerank_model, num_labels=1, trust_remote_code=True
            ).to(self.device)

    def get_llm(self):
        return self.llm

    def get_llm_simple(self):
        return self.llm_simple

    def get_tokenizer(self):
        return self.tokenizer

    def get_rerank_model(self):
        return self.rerank_model

    def get_embedding_model(self):
        return self.embedding_model
