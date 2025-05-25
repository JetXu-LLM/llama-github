# initial_load.py
from typing import Optional, Any
from threading import Lock

from llama_github.config.config import config
from llama_github.logger import logger

class LLMManager:
    """
    Singleton class for managing Language Models and related components.
    This class handles initialization and access to various models including LLMs,
    embedding models, and reranking models.
    """
    _instance_lock = Lock()
    _instance = None
    llm = None
    rerank_model = None
    _initialized = False
    llm_simple = None
    tokenizer = None
    embedding_model = None

    def __new__(cls, *args, **kwargs):
        """
        Ensure only one instance of LLMManager is created (Singleton pattern).
        """
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
                 llm: Any = None,
                 simple_mode: bool = False):
        """
        Initialize the LLMManager with specified models and API keys.

        Args:
            openai_api_key (Optional[str]): API key for OpenAI.
            mistral_api_key (Optional[str]): API key for Mistral AI.
            huggingface_token (Optional[str]): Token for Hugging Face.
            open_source_models_hg_dir (Optional[str]): Directory for open-source models.
            embedding_model (Optional[str]): Name or path of the embedding model.
            rerank_model (Optional[str]): Name or path of the reranking model.
            llm (Any): Custom LLM instance if provided.
            simple_mode (bool): If True, skip initialization of embedding and reranking models.
        """
        with self._instance_lock:   # Prevent re-initialization
            if self._initialized:
                return
            self._initialized = True

        self.simple_mode = simple_mode

        # Initialize LLM based on provided API keys or custom LLM
        if llm is not None:
            self.llm = llm
            self.model_type = "Custom_langchain_llm"
        elif mistral_api_key is not None and mistral_api_key != "" and self.llm is None:
            logger.info("Initializing Codestral API...")
            from langchain_mistralai.chat_models import ChatMistralAI
            self.llm = ChatMistralAI(mistral_api_key=mistral_api_key, model="mistral-medium-latest", temperature=0.3)
            self.llm_simple = ChatMistralAI(
                mistral_api_key=mistral_api_key,
                model="devstral-small-latest",
                temperature=0.2
            )
            self.model_type = "OpenAI"
        elif openai_api_key is not None and openai_api_key != "" and self.llm is None:
            from langchain_openai import ChatOpenAI
            logger.info("Initializing OpenAI API...")
            self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo")
            self.llm_simple = ChatOpenAI(
                api_key=openai_api_key, model="gpt-4o-mini")
            self.model_type = "OpenAI"
        # Initialize for Open Source Models
        elif open_source_models_hg_dir is not None and open_source_models_hg_dir != "" and self.llm is None:
            logger.info(f"Initializing {open_source_models_hg_dir}...")
            # load huggingface models
            self.model_type = "Hubgingface"
        elif self.llm is None:
            # default model is phi3_mini_128k
            self.model_type = "Hubgingface"
            
        if not self.simple_mode:
            import sys
            import platform
            import subprocess

            def get_device():
                if sys.platform.startswith('darwin'):  # macOS
                    # Check for Apple Silicon (M1/M2)
                    if platform.machine() == 'arm64':
                        return 'mps'
                elif sys.platform.startswith('linux') or sys.platform.startswith('win'):
                    # Check for NVIDIA GPU
                    try:
                        subprocess.run(['nvidia-smi'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return 'cuda'
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        pass
                
                # Default to CPU
                return 'cpu'

            # Usage
            self.device = get_device()

            from transformers import AutoModel
            from transformers import AutoModelForSequenceClassification
            from transformers import AutoTokenizer
            # Initialize embedding model
            if self.tokenizer is None:
                logger.info(f"Initializing {embedding_model}...")
                self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
                self.embedding_model = AutoModel.from_pretrained(
                    embedding_model, trust_remote_code=True).to(self.device)

            # Initialize reranking model
            if self.rerank_model is None:
                logger.info(f"Initializing {rerank_model}...")
                self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                    rerank_model, num_labels=1, trust_remote_code=True
                ).to(self.device)
        else:
            logger.info("Simple mode enabled. Skipping embedding and rerank model initialization.")

    def get_llm(self):
        """
        Get the main Language Model.

        Returns:
            The initialized Language Model.
        """
        return self.llm

    def get_llm_simple(self):
        """
        Get the simplified Language Model.

        Returns:
            The initialized simplified Language Model.
        """
        return self.llm_simple

    def get_tokenizer(self):
        """
        Get the tokenizer for the embedding model.

        Returns:
            The initialized tokenizer.
        """
        return self.tokenizer

    def get_rerank_model(self):
        """
        Get the reranking model.

        Returns:
            The initialized reranking model.
        """
        return self.rerank_model

    def get_embedding_model(self):
        """
        Get the embedding model.

        Returns:
            The initialized embedding model.
        """
        return self.embedding_model
