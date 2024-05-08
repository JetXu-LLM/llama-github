# initial_load.py
from threading import Lock
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

class LLMManager:
    _instance_lock = Lock()
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:  # First check (unlocked)
            with cls._instance_lock:  # Acquire lock
                if cls._instance is None:  # Second check (locked)
                    cls._instance = super(LLMManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, openai_api_key, huggingface_token, open_source_models_hg_dir, 
                 embedding_model="jinaai/jina-embeddings-v2-base-code", rerank_model="jinaai/jina-reranker-v1-turbo-en", llm=None):
        # Initialize for OpenAI GPT-4
        if llm is not None:
            self.llm = llm
            self.model_type = "Custom_langchain_llm"
        elif openai_api_key is not None and openai_api_key != "" and self.llm is None:
            self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4-turbo")
            self.model_type = "OpenAI"
        # Initialize for Open Source Models
        elif open_source_models_hg_dir is not None and open_source_models_hg_dir != "" and self.llm is None:
            #load hugingface models
            self.model_type = "Hubgingface"
        else:
            #default model is phi3_mini_128k
            self.model_type = "Hubgingface"

        #initial embeddings model
        device="cpu"
        if torch.cuda.is_available():
            device="cuda"
        elif torch.backends.mps.is_available():
            device="mps"
        
        model_kwargs = {'device': device, 'trust_remote_code':True}
        if (huggingface_token is not None and huggingface_token != ""):
            model_kwargs['token'] = huggingface_token
        encode_kwargs = {'normalize_embeddings': True}
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        self.rerank_model = HuggingFaceEmbeddings(
            model_name=rerank_model,
            model_kwargs=model_kwargs,
        )

    def get_llm(self):
        return self.llm
    
    def get_embedding_model(self):
        return self.embedding_model
    
    def get_rerank_model(self):
        return self.rerank_model