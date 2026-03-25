import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from llama_github.llm_integration.initial_load import LLMManager


class TestLLMManager:
    def test_init_openai_lazy_load(self, monkeypatch):
        fake_chat_model = MagicMock(name="chat-openai")
        chat_openai_cls = MagicMock(return_value=fake_chat_model)
        monkeypatch.setitem(
            sys.modules,
            "langchain_openai",
            SimpleNamespace(ChatOpenAI=chat_openai_cls),
        )

        manager = LLMManager(openai_api_key="sk-test", simple_mode=True)
        assert manager.model_type == "OpenAI"
        assert manager.llm is None

        llm = manager.get_llm()
        assert llm is fake_chat_model
        assert manager.get_llm_simple() is fake_chat_model
        assert chat_openai_cls.call_count == 2

    def test_init_huggingface_full_mode(self, monkeypatch):
        fake_tokenizer = MagicMock()
        fake_embedding = MagicMock()
        fake_embedding.to.return_value = fake_embedding
        fake_reranker = MagicMock()
        fake_reranker.to.return_value = fake_reranker

        auto_tokenizer = MagicMock()
        auto_tokenizer.from_pretrained.return_value = fake_tokenizer
        auto_model = MagicMock()
        auto_model.from_pretrained.return_value = fake_embedding
        auto_seq = MagicMock()
        auto_seq.from_pretrained.return_value = fake_reranker

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoTokenizer=auto_tokenizer,
                AutoModel=auto_model,
                AutoModelForSequenceClassification=auto_seq,
            ),
        )
        monkeypatch.setattr(LLMManager, "_get_device", lambda self: "cpu")

        manager = LLMManager(
            open_source_models_hg_dir="/tmp",
            simple_mode=False,
            embedding_model="emb-model",
            rerank_model="rerank-model",
        )

        assert manager.get_tokenizer() is fake_tokenizer
        assert manager.get_embedding_model() is fake_embedding
        assert manager.get_rerank_model() is fake_reranker
        auto_tokenizer.from_pretrained.assert_called_with("emb-model")

    def test_simple_mode_skips_heavy_models(self):
        manager = LLMManager(simple_mode=True)
        assert manager.get_embedding_model() is None
        assert manager.get_rerank_model() is None
