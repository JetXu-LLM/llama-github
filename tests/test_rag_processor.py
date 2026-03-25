from unittest.mock import MagicMock, patch

import pytest

from llama_github.rag_processing.rag_processor import RAGProcessor


class TestRAGProcessor:
    @pytest.fixture
    def processor(self, mock_llm_handler):
        mock_api_handler = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_llm_simple.return_value = None
        mock_manager.get_rerank_model.return_value = None
        mock_manager.get_embedding_model.return_value = None
        mock_manager.get_tokenizer.return_value = None
        return RAGProcessor(
            github_api_handler=mock_api_handler,
            llm_manager=mock_manager,
            llm_handler=mock_llm_handler,
        )

    @pytest.mark.asyncio
    async def test_analyze_question(self, processor):
        mock_response = MagicMock()
        mock_response.question = "Refined Question"
        mock_response.answer = "Draft Answer"
        mock_response.code_search_logic = "Logic"
        mock_response.issue_search_logic = "Logic"

        processor.llm_handler.ainvoke.return_value = mock_response

        result = await processor.analyze_question("Raw Question")

        assert result[0] == "Refined Question"
        assert result[1] == "Draft Answer"

    @pytest.mark.asyncio
    async def test_get_code_search_criteria(self, processor):
        mock_response = MagicMock()
        mock_response.search_criteria = ["query1", "query2"]
        processor.llm_handler.ainvoke.return_value = mock_response

        criteria = await processor.get_code_search_criteria("Question")

        assert len(criteria) == 2
        assert "query1" in criteria

    def test_arrange_code_search_result(self, processor):
        raw_results = [
            {
                "content": "def foo(): pass",
                "url": "http://github.com",
                "repository_full_name": "owner/repo",
                "language": "python",
            }
        ]

        with patch.object(
            processor,
            "_split_content_into_chunks",
            return_value=["def foo(): pass"],
        ):
            arranged = processor._arrange_code_search_result(raw_results)

        assert len(arranged) == 1
        assert "Sample code from repository: owner/repo" in arranged[0]["context"]
        assert arranged[0]["url"] == "http://github.com"

    @pytest.mark.asyncio
    async def test_retrieve_topn_contexts_falls_back_without_models(self, processor):
        context_list = [
            {"context": "good context with query", "url": "url1"},
            {"context": "bad context", "url": "url2"},
        ]

        top_n = await processor.retrieve_topn_contexts(context_list, "query", top_n=1)

        assert top_n == [{"context": "good context with query", "url": "url1"}]
