import pytest
pytestmark = pytest.mark.asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.form_pipeline import FormPipeline

@pytest.fixture
def mock_cache_service():
    return MagicMock()

@pytest.fixture
def pipeline(mock_cache_service):
    # Patch FormAnalyzer and FormFiller to avoid real side effects
    with patch("app.services.form_pipeline.FormAnalyzer") as MockAnalyzer, \
         patch("app.services.form_pipeline.FormFiller") as MockFiller:
        MockAnalyzer.return_value.analyze_labels_fast = AsyncMock(return_value={
            "status": "success",
            "label_count": 2,
            "labels": ["<label>Name</label>", "<label>Email</label>"]
        })
        MockFiller.return_value = MagicMock()
        yield FormPipeline(
            openai_api_key="test-key",
            db_url="sqlite:///:memory:",
            cache_service=mock_cache_service
        )

@patch("app.services.form_pipeline.extract_question_from_label_html", return_value="What is your name?")
@patch("app.services.form_pipeline.embed_question", return_value=[0.1, 0.2, 0.3])
@pytest.mark.asyncio
async def test_run_complete_pipeline_success(mock_embed, mock_extract, pipeline):
    result = await pipeline.run_complete_pipeline(
        url="http://example.com",
        user_data={"name": "Test"},
        force_refresh=False,
        submit_form=False,
        preview_only=True
    )
    assert result["pipeline_status"] == "completed"
    assert "steps" in result
    assert "label_extraction" in result["steps"]
    assert result["steps"]["label_extraction"]["status"] == "success"
    assert result["steps"]["label_extraction"]["label_count"] == 2
    assert len(result["steps"]["label_extraction"]["labels"]) == 2

@pytest.mark.asyncio
async def test_run_complete_pipeline_label_extraction_failure(pipeline):
    # Patch analyze_labels_fast to simulate failure
    pipeline.form_analyzer.analyze_labels_fast = AsyncMock(return_value={
        "status": "error",
        "error": "Failed to extract labels"
    })
    result = await pipeline.run_complete_pipeline(
        url="http://example.com",
        user_data={},
        force_refresh=False,
        submit_form=False,
        preview_only=True
    )
    assert result["pipeline_status"] == "failed"
    assert "error" in result

@pytest.mark.asyncio
async def test_analyze_and_preview_calls_run_complete_pipeline(pipeline):
    with patch.object(pipeline, "run_complete_pipeline", new=AsyncMock(return_value={"pipeline_status": "completed"})) as mock_run:
        result = await pipeline.analyze_and_preview("http://example.com", {"name": "Test"})
        assert result["pipeline_status"] == "completed"
        mock_run.assert_awaited_once()

@pytest.mark.asyncio
async def test_analyze_and_fill_calls_run_complete_pipeline(pipeline):
    with patch.object(pipeline, "run_complete_pipeline", new=AsyncMock(return_value={"pipeline_status": "completed"})) as mock_run:
        result = await pipeline.analyze_and_fill("http://example.com", {"name": "Test"}, submit=True)
        assert result["pipeline_status"] == "completed"
        mock_run.assert_awaited_once()