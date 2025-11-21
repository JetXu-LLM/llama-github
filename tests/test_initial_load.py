import pytest
from unittest.mock import patch, MagicMock
from llama_github.llm_integration.initial_load import LLMManager

@pytest.fixture(autouse=True)
def reset_singleton():
    LLMManager._instance = None
    LLMManager._initialized = False
    yield
    LLMManager._instance = None

class TestLLMManager:
    @patch('llama_github.llm_integration.initial_load.ChatOpenAI')
    def test_init_openai(self, mock_chat_openai):
        manager = LLMManager(openai_api_key="sk-test", simple_mode=True)
        assert manager.model_type == "OpenAI"
        assert manager.llm is not None
        mock_chat_openai.assert_called()

    @patch('llama_github.llm_integration.initial_load.AutoTokenizer')
    @patch('llama_github.llm_integration.initial_load.AutoModel')
    @patch('llama_github.llm_integration.initial_load.AutoModelForSequenceClassification')
    def test_init_huggingface_full_mode(self, mock_seq, mock_model, mock_tokenizer):
        # Mock system checks
        with patch('sys.platform', 'linux'), \
             patch('subprocess.run'):
            
            manager = LLMManager(
                open_source_models_hg_dir="/tmp", 
                simple_mode=False,
                embedding_model="emb-model",
                rerank_model="rerank-model"
            )
            
            assert manager.tokenizer is not None
            assert manager.embedding_model is not None
            assert manager.rerank_model is not None
            mock_tokenizer.from_pretrained.assert_called_with("emb-model")

    def test_simple_mode_skips_heavy_models(self):
        manager = LLMManager(simple_mode=True)
        assert manager.embedding_model is None
        assert manager.rerank_model is None