#!/usr/bin/env python3
"""
End-to-end tests with Playwright
Tests complete user workflows including memory operations
"""

import pytest
import asyncio
import json
import os
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Test configuration
E2E_BASE_URL = os.getenv("E2E_BASE_URL", "http://localhost:3000")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

@pytest.mark.e2e
class TestEndToEndWorkflows:
    """End-to-end tests for complete user workflows"""
    
    @pytest.fixture(scope="class")
    async def browser_setup(self):
        """Setup browser for E2E tests"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1280, "height": 720},
                ignore_https_errors=True
            )
            page = await context.new_page()
            
            yield {"browser": browser, "context": context, "page": page}
            
            await context.close()
            await browser.close()
    
    async def test_user_authentication_flow(self, browser_setup):
        """Test complete user authentication workflow"""
        page = browser_setup["page"]
        
        # Navigate to login page
        await page.goto(f"{E2E_BASE_URL}/login")
        await page.wait_for_load_state("networkidle")
        
        # Check if login form exists
        login_form = await page.query_selector("form")
        if login_form:
            # Fill login form
            await page.fill('input[type="email"]', "test@example.com")
            await page.fill('input[type="password"]', "testpassword123")
            
            # Submit form
            await page.click('button[type="submit"]')
            await page.wait_for_load_state("networkidle")
            
            # Verify successful login (check for dashboard or user menu)
            dashboard_element = await page.query_selector('[data-testid="dashboard"]')
            user_menu = await page.query_selector('[data-testid="user-menu"]')
            
            assert dashboard_element or user_menu, "Login should redirect to authenticated area"
    
    async def test_memory_workflow(self, browser_setup):
        """Test memory storage and retrieval workflow"""
        page = browser_setup["page"]
        
        # Navigate to memory interface
        await page.goto(f"{E2E_BASE_URL}/memory")
        await page.wait_for_load_state("networkidle")
        
        # Test memory storage
        memory_input = await page.query_selector('input[data-testid="memory-input"]')
        if memory_input:
            test_memory = "This is a test memory for E2E testing"
            await page.fill('input[data-testid="memory-input"]', test_memory)
            
            # Save memory
            save_button = await page.query_selector('button[data-testid="save-memory"]')
            if save_button:
                await save_button.click()
                await page.wait_for_load_state("networkidle")
                
                # Verify memory was saved
                success_message = await page.query_selector('[data-testid="success-message"]')
                assert success_message, "Memory save should show success message"
        
        # Test memory retrieval
        search_input = await page.query_selector('input[data-testid="memory-search"]')
        if search_input:
            await page.fill('input[data-testid="memory-search"]', "test memory")
            
            search_button = await page.query_selector('button[data-testid="search-memory"]')
            if search_button:
                await search_button.click()
                await page.wait_for_load_state("networkidle")
                
                # Verify search results
                results = await page.query_selector_all('[data-testid="memory-result"]')
                assert len(results) > 0, "Memory search should return results"
    
    async def test_api_integration_workflow(self, browser_setup):
        """Test API integration through the UI"""
        page = browser_setup["page"]
        
        # Test API calls through browser
        response = await page.evaluate(f"""
            async () => {{
                try {{
                    const response = await fetch('{API_BASE_URL}/health');
                    const data = await response.json();
                    return {{ success: true, data }};
                }} catch (error) {{
                    return {{ success: false, error: error.message }};
                }}
            }}
        """)
        
        assert response["success"], f"API health check should succeed: {response.get('error', '')}"
        assert response["data"]["status"] == "healthy", "API should return healthy status"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])