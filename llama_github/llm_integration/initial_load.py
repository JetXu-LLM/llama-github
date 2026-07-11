from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Optional

from llama_github.config.config import config
from llama_github.logger import logger


class LLMManager:
    """
    Lazily manages chat models, embeddings, rerankers, and tokenizers.

    The manager is instance-scoped so different GithubRAG instances can safely
    carry different credentials or model choices in the same process.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        open_source_models_hg_dir: Optional[str] = None,
        embedding_model: Optional[str] = config.get("default_embedding"),
        rerank_model: Optional[str] = config.get("default_reranker"),
        llm: Any = None,
        simple_mode: bool = False,
        embedding_revision: Optional[str] = None,
        rerank_revision: Optional[str] = None,
    ):
        """
        Initialize an instance-scoped model manager.

        The manager loads chat models lazily and only loads embedding / reranker
        models on demand when professional-mode ranking requires them.
        """
        self.openai_api_key = openai_api_key
        self.mistral_api_key = mistral_api_key
        self.huggingface_token = huggingface_token
        self.open_source_models_hg_dir = open_source_models_hg_dir
        self.embedding_model_name = embedding_model or config.get("default_embedding")
        self.rerank_model_name = rerank_model or config.get("default_reranker")
        self.embedding_revision = embedding_revision
        if self.embedding_revision is None and (
            self.embedding_model_name == config.get("default_embedding")
        ):
            self.embedding_revision = config.get("default_embedding_revision")
        self.rerank_revision = rerank_revision
        if self.rerank_revision is None and (
            self.rerank_model_name == config.get("default_reranker")
        ):
            self.rerank_revision = config.get("default_reranker_revision")
        self.simple_mode = simple_mode

        self.llm = llm
        self.llm_simple = llm
        self.tokenizer = None
        self.embedding_model = None
        self.rerank_model = None
        self._provider_warning_emitted = False
        self._revision_warnings_emitted = set()

        if llm is not None:
            self.model_type = "Custom"
        elif mistral_api_key:
            self.model_type = "Mistral"
        elif openai_api_key:
            self.model_type = "OpenAI"
        elif open_source_models_hg_dir:
            self.model_type = "ReservedOpenSource"
        else:
            self.model_type = "Unavailable"

        if self.simple_mode:
            logger.info(
                "Simple mode enabled. Embedding and rerank models will load only if explicitly requested."
            )

    def _warn_unsupported_open_source_mode(self) -> None:
        """Emit the open-source placeholder warning at most once per manager instance."""
        if self.model_type == "ReservedOpenSource" and not self._provider_warning_emitted:
            logger.warning(
                "open_source_models_hg_dir is reserved for future native provider support. "
                "Use a custom compatible chat model via the llm argument for now."
            )
            self._provider_warning_emitted = True

    def _load_openai_models(self) -> None:
        """Instantiate the default OpenAI chat models."""
        if self.llm is not None:
            return

        from langchain_openai import ChatOpenAI

        logger.info("Initializing OpenAI chat models...")
        self.llm = ChatOpenAI(
            api_key=self.openai_api_key,
            model=config.get("default_openai_model"),
        )
        self.llm_simple = ChatOpenAI(
            api_key=self.openai_api_key,
            model=config.get("default_openai_simple_model"),
        )

    def _load_mistral_models(self) -> None:
        """Instantiate the default Mistral chat models."""
        if self.llm is not None:
            return

        from langchain_mistralai.chat_models import ChatMistralAI

        logger.info("Initializing Mistral chat models...")
        self.llm = ChatMistralAI(
            mistral_api_key=self.mistral_api_key,
            model=config.get("default_mistral_model"),
            temperature=0.2,
        )
        self.llm_simple = ChatMistralAI(
            mistral_api_key=self.mistral_api_key,
            model=config.get("default_mistral_simple_model"),
            temperature=0.1,
        )

    def _ensure_chat_models_loaded(self) -> None:
        """Load chat models lazily based on the configured provider choice."""
        if self.llm is not None:
            return

        if self.model_type == "Mistral":
            self._load_mistral_models()
        elif self.model_type == "OpenAI":
            self._load_openai_models()
        elif self.model_type == "ReservedOpenSource":
            self._warn_unsupported_open_source_mode()

    def _get_device(self) -> str:
        """Pick the preferred inference device for local embedding and reranker models."""
        if sys.platform.startswith("darwin") and platform.machine() == "arm64":
            return "mps"

        if sys.platform.startswith(("linux", "win")):
            try:
                subprocess.run(
                    ["nvidia-smi"],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return "cuda"
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        return "cpu"

    def _ensure_tokenizer_loaded(self) -> None:
        """Load the tokenizer used for chunk sizing and professional-mode query token counting."""
        if self.tokenizer is not None:
            return

        from transformers import AutoTokenizer

        logger.info("Initializing tokenizer with configured embedding model")
        kwargs = self._model_revision_kwargs("embedding", self.embedding_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_name,
            **kwargs,
        )

    def _ensure_embedding_model_loaded(self) -> None:
        """Load the embedding model lazily when semantic ranking is needed."""
        if self.embedding_model is not None:
            return

        from transformers import AutoModel

        self._ensure_tokenizer_loaded()
        logger.info("Initializing configured embedding model")
        kwargs = self._model_revision_kwargs("embedding", self.embedding_revision)
        self.embedding_model = AutoModel.from_pretrained(
            self.embedding_model_name,
            trust_remote_code=True,
            **kwargs,
        ).to(self._get_device())

    def _ensure_rerank_model_loaded(self) -> None:
        """Load the reranker model lazily when professional-mode ranking is needed."""
        if self.rerank_model is not None:
            return

        from transformers import AutoModelForSequenceClassification

        logger.info("Initializing configured reranker")
        kwargs = self._model_revision_kwargs("reranker", self.rerank_revision)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            self.rerank_model_name,
            num_labels=1,
            trust_remote_code=True,
            **kwargs,
        ).to(self._get_device())

    def _model_revision_kwargs(
        self,
        model_kind: str,
        revision: Optional[str],
    ) -> dict:
        """Pin built-in models and warn once when callers choose an unpinned model."""
        if revision:
            return {"revision": revision}
        if model_kind not in self._revision_warnings_emitted:
            logger.warning(
                "Custom %s model has no immutable revision; remote code may change",
                model_kind,
            )
            self._revision_warnings_emitted.add(model_kind)
        return {}

    def get_llm(self):
        """Return the main chat model, loading it if necessary."""
        self._ensure_chat_models_loaded()
        return self.llm

    def get_llm_simple(self):
        """Return the lightweight chat model, loading it if necessary."""
        self._ensure_chat_models_loaded()
        return self.llm_simple

    def get_tokenizer(self):
        """Return the tokenizer used in professional mode, or `None` in simple mode."""
        if self.simple_mode:
            return None
        self._ensure_tokenizer_loaded()
        return self.tokenizer

    def get_rerank_model(self):
        """Return the reranker model in professional mode, or `None` in simple mode."""
        if self.simple_mode:
            return None
        self._ensure_rerank_model_loaded()
        return self.rerank_model

    def get_embedding_model(self):
        """Return the embedding model in professional mode, or `None` in simple mode."""
        if self.simple_mode:
            return None
        self._ensure_embedding_model_loaded()
        return self.embedding_model
