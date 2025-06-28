import pytest
pytestmark = pytest.mark.asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.form_analyzer import FormAnalyzer

class AsyncContextManagerMock:
    def __init__(self, value):
        self.value = value
    async def __aenter__(self):
        return self.value
    async def __aexit__(self, exc_type, exc, tb):
        pass

@pytest.mark.asyncio
async def test_analyze_labels_fast_extracts_labels():
    # Simulate HTML with labels
    html_content = '''
    <html><body>
        <label for="name">Full Name</label>
        <label for="email">Email Address</label>
        <label for="phone">Phone Number</label>
    </body></html>
    '''

    with patch("app.services.form_analyzer.async_playwright") as mock_playwright:
        mock_p = MagicMock()
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_page = MagicMock()
        # Set up async context manager for playwright
        mock_playwright.return_value = AsyncContextManagerMock(mock_p)
        # Set up awaitable and context manager for chromium.launch
        mock_p.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
        mock_browser.__aexit__ = AsyncMock(return_value=None)
        mock_browser.close = AsyncMock()
        # Set up awaitable and context manager for new_context
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        # Set up awaitable and context manager for new_page
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_page.__aenter__ = AsyncMock(return_value=mock_page)
        mock_page.__aexit__ = AsyncMock(return_value=None)
        # Mock page.goto and page.content
        mock_page.goto = AsyncMock()
        mock_page.content = AsyncMock(return_value=html_content)
        mock_page.query_selector_all = AsyncMock(return_value=[MagicMock(), MagicMock(), MagicMock()])
        mock_page.route = AsyncMock()
        mock_page.wait_for_selector = AsyncMock()
        # Each label's evaluate returns the HTML string
        label_htmls = [
            '<label for="name">Full Name</label>',
            '<label for="email">Email Address</label>',
            '<label for="phone">Phone Number</label>'
        ]
        for mock_label, html in zip(mock_page.query_selector_all.return_value, label_htmls):
            mock_label.evaluate = AsyncMock(return_value=html)

        analyzer = FormAnalyzer(cache_service=None)
        result = await analyzer.analyze_labels_fast("http://fake-url.com", force_refresh=True)

        assert result["status"] == "success"
        assert result["label_count"] == 3
        assert any("Full Name" in label for label in result["labels"])
        assert any("Email Address" in label for label in result["labels"])
        assert any("Phone Number" in label for label in result["labels"]) 