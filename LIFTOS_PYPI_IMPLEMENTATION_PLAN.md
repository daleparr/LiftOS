# LiftOS PyPI Package - Implementation Plan

**Technical roadmap for creating the LiftOS Python SDK**

This document outlines the complete implementation strategy for packaging LiftOS as a PyPI distribution, enabling data scientists to access causal AI capabilities through simple Python imports.

---

## 🏗️ Package Architecture

### Core Package Structure
```
liftos/
├── __init__.py                 # Main package exports
├── client.py                   # Core client implementation
├── config.py                   # Configuration management
├── exceptions.py               # Custom exceptions
├── auth/
│   ├── __init__.py
│   ├── api_key.py             # API key authentication
│   └── oauth.py               # OAuth flow (future)
├── modules/
│   ├── __init__.py
│   ├── surfacing.py           # Surfacing module client
│   ├── causal.py              # Causal analysis client
│   ├── llm.py                 # LLM evaluation client
│   └── memory.py              # Memory system client
├── integrations/
│   ├── __init__.py
│   ├── pandas.py              # Pandas integration
│   ├── jupyter.py             # Jupyter notebook helpers
│   └── streamlit.py           # Streamlit components
├── analytics/
│   ├── __init__.py
│   ├── roi_tracker.py         # ROI calculation
│   └── performance.py         # Performance monitoring
├── training/
│   ├── __init__.py
│   └── custom_trainer.py      # Custom model training
└── utils/
    ├── __init__.py
    ├── validation.py          # Input validation
    ├── formatting.py          # Output formatting
    └── caching.py             # Response caching
```

### Setup Configuration
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="liftos",
    version="1.0.0",
    description="Causal AI platform for marketing optimization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="LiftOS Team",
    author_email="sdk@liftos.com",
    url="https://github.com/liftos/python-sdk",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "pydantic>=1.10.0",
        "httpx>=0.24.0",
        "aiohttp>=3.8.0",
        "python-dotenv>=0.19.0",
        "rich>=12.0.0",
        "typer>=0.7.0",
    ],
    extras_require={
        "jupyter": ["ipywidgets>=8.0.0", "plotly>=5.0.0"],
        "streamlit": ["streamlit>=1.25.0"],
        "dev": ["pytest>=7.0.0", "black>=22.0.0", "mypy>=0.991"],
        "async": ["aiohttp>=3.8.0", "asyncio>=3.4.3"],
    },
    entry_points={
        "console_scripts": [
            "liftos=liftos.cli:main",
        ],
    },
)
```

---

## 🔧 Core Implementation

### 1. Main Client Class

```python
# liftos/client.py
import os
from typing import Optional, Dict, Any
import httpx
from .config import Config
from .auth.api_key import APIKeyAuth
from .modules.surfacing import SurfacingClient
from .modules.causal import CausalClient
from .modules.llm import LLMClient
from .modules.memory import MemoryClient
from .exceptions import LiftOSError, AuthenticationError

class Client:
    """Main LiftOS client for accessing all modules."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.liftos.com",
        timeout: int = 30,
        max_retries: int = 3,
        config: Optional[Config] = None
    ):
        """Initialize LiftOS client.
        
        Args:
            api_key: LiftOS API key (or set LIFTOS_API_KEY env var)
            base_url: API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            config: Custom configuration object
        """
        self.api_key = api_key or os.getenv("LIFTOS_API_KEY")
        if not self.api_key:
            raise AuthenticationError("API key required. Set LIFTOS_API_KEY env var or pass api_key parameter.")
        
        self.base_url = base_url.rstrip("/")
        self.config = config or Config()
        
        # Initialize HTTP client
        self._http_client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": f"liftos-python/{self.config.version}",
                "Content-Type": "application/json"
            }
        )
        
        # Initialize module clients
        self._surfacing = None
        self._causal = None
        self._llm = None
        self._memory = None
    
    def surfacing(self) -> SurfacingClient:
        """Get Surfacing module client."""
        if self._surfacing is None:
            self._surfacing = SurfacingClient(self._http_client, self.config)
        return self._surfacing
    
    def causal(self) -> CausalClient:
        """Get Causal analysis client."""
        if self._causal is None:
            self._causal = CausalClient(self._http_client, self.config)
        return self._causal
    
    def llm(self) -> LLMClient:
        """Get LLM evaluation client."""
        if self._llm is None:
            self._llm = LLMClient(self._http_client, self.config)
        return self._llm
    
    def memory(self) -> MemoryClient:
        """Get Memory system client."""
        if self._memory is None:
            self._memory = MemoryClient(self._http_client, self.config)
        return self._memory
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics."""
        response = self._http_client.get("/v1/usage")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if LiftOS API is accessible."""
        try:
            response = self._http_client.get("/health")
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close HTTP client."""
        self._http_client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

### 2. Surfacing Module Implementation

```python
# liftos/modules/surfacing.py
from typing import Dict, List, Any, Optional, Union
import httpx
from pydantic import BaseModel
from ..exceptions import LiftOSError, ValidationError

class ProductAnalysisResult(BaseModel):
    """Result of product analysis."""
    revenue_impact: float
    conversion_lift: float
    seo_score: int
    competitive_score: float
    optimization_recommendations: List[Dict[str, Any]]
    keyword_opportunities: List[Dict[str, Any]]
    sentiment_analysis: Dict[str, Any]
    confidence_score: float
    processing_time: float

class SurfacingClient:
    """Client for LiftOS Surfacing module."""
    
    def __init__(self, http_client: httpx.Client, config):
        self.http_client = http_client
        self.config = config
    
    def analyze_product(
        self,
        product_description: str,
        competitor_data: Optional[List[str]] = None,
        price_point: Optional[float] = None,
        market_focus: str = "premium",
        analysis_depth: str = "standard"
    ) -> ProductAnalysisResult:
        """Analyze a product for optimization opportunities.
        
        Args:
            product_description: Product description text
            competitor_data: List of competitor product names/descriptions
            price_point: Product price for competitive analysis
            market_focus: Target market ("budget", "mid-tier", "premium", "luxury")
            analysis_depth: Analysis depth ("quick", "standard", "comprehensive")
        
        Returns:
            ProductAnalysisResult with optimization insights
        """
        if not product_description or len(product_description.strip()) < 10:
            raise ValidationError("Product description must be at least 10 characters")
        
        payload = {
            "product_description": product_description,
            "competitor_data": competitor_data or [],
            "price_point": price_point,
            "market_focus": market_focus,
            "analysis_depth": analysis_depth
        }
        
        try:
            response = self.http_client.post("/v1/surfacing/analyze", json=payload)
            response.raise_for_status()
            data = response.json()
            return ProductAnalysisResult(**data)
        except httpx.HTTPStatusError as e:
            raise LiftOSError(f"API request failed: {e.response.status_code}")
        except Exception as e:
            raise LiftOSError(f"Analysis failed: {str(e)}")
    
    def batch_analyze(
        self,
        products: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> List[ProductAnalysisResult]:
        """Analyze multiple products in batches.
        
        Args:
            products: List of product dictionaries with 'description' key
            batch_size: Number of products to process per batch
        
        Returns:
            List of ProductAnalysisResult objects
        """
        results = []
        
        for i in range(0, len(products), batch_size):
            batch = products[i:i + batch_size]
            
            payload = {
                "products": batch,
                "batch_size": batch_size
            }
            
            try:
                response = self.http_client.post("/v1/surfacing/batch-analyze", json=payload)
                response.raise_for_status()
                batch_results = response.json()["results"]
                
                for result_data in batch_results:
                    results.append(ProductAnalysisResult(**result_data))
                    
            except httpx.HTTPStatusError as e:
                raise LiftOSError(f"Batch analysis failed: {e.response.status_code}")
        
        return results
    
    def optimize_content(
        self,
        current_content: str,
        optimization_goals: List[str],
        target_audience: Optional[str] = None
    ) -> Dict[str, Any]:
        """Optimize content for specific goals.
        
        Args:
            current_content: Current content text
            optimization_goals: List of goals ("seo", "conversion", "engagement")
            target_audience: Target audience description
        
        Returns:
            Optimization recommendations and improved content
        """
        payload = {
            "current_content": current_content,
            "optimization_goals": optimization_goals,
            "target_audience": target_audience
        }
        
        response = self.http_client.post("/v1/surfacing/optimize", json=payload)
        response.raise_for_status()
        return response.json()
```

### 3. Causal Analysis Implementation

```python
# liftos/modules/causal.py
from typing import Dict, List, Any, Optional
import pandas as pd
import httpx
from pydantic import BaseModel
from ..exceptions import LiftOSError, InsufficientDataError

class AttributionResult(BaseModel):
    """Result of causal attribution analysis."""
    accuracy_score: float
    wasted_spend: float
    reallocation_opportunity: float
    true_roas: float
    channel_attribution: Dict[str, Dict[str, Any]]
    causal_insights: List[Dict[str, Any]]
    incrementality_analysis: Dict[str, float]
    confidence_intervals: Dict[str, Dict[str, float]]

class CausalClient:
    """Client for LiftOS Causal analysis module."""
    
    def __init__(self, http_client: httpx.Client, config):
        self.http_client = http_client
        self.config = config
    
    def analyze_attribution(
        self,
        campaigns: Union[pd.DataFrame, List[Dict[str, Any]]],
        revenue_data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
        time_period: str = "90d",
        attribution_model: str = "causal_forest",
        confidence_threshold: float = 0.85
    ) -> AttributionResult:
        """Perform causal attribution analysis.
        
        Args:
            campaigns: Campaign data with spend, impressions, clicks
            revenue_data: Revenue data with timestamps and amounts
            time_period: Analysis time window ("30d", "90d", "180d", "1y")
            attribution_model: Model type ("causal_forest", "linear", "last_touch")
            confidence_threshold: Minimum confidence for recommendations
        
        Returns:
            AttributionResult with causal insights
        """
        # Convert pandas DataFrame to dict if needed
        if isinstance(campaigns, pd.DataFrame):
            campaigns_data = campaigns.to_dict("records")
        else:
            campaigns_data = campaigns
        
        if isinstance(revenue_data, pd.DataFrame):
            revenue_data = revenue_data.to_dict("records")
        
        # Validate minimum data requirements
        if len(campaigns_data) < 30:
            raise InsufficientDataError("Need at least 30 campaign data points for reliable analysis")
        
        payload = {
            "campaigns": campaigns_data,
            "revenue_data": revenue_data,
            "time_period": time_period,
            "attribution_model": attribution_model,
            "confidence_threshold": confidence_threshold
        }
        
        try:
            response = self.http_client.post("/v1/causal/attribution", json=payload)
            response.raise_for_status()
            data = response.json()
            return AttributionResult(**data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                error_detail = e.response.json().get("detail", "Validation error")
                raise InsufficientDataError(f"Data validation failed: {error_detail}")
            raise LiftOSError(f"Attribution analysis failed: {e.response.status_code}")
    
    def optimize_budget(
        self,
        current_allocation: Dict[str, float],
        target_roas: float,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Optimize budget allocation across channels.
        
        Args:
            current_allocation: Current spend by channel
            target_roas: Target return on ad spend
            constraints: Budget constraints (min/max per channel)
        
        Returns:
            Optimized budget allocation with expected outcomes
        """
        payload = {
            "current_allocation": current_allocation,
            "target_roas": target_roas,
            "constraints": constraints or {}
        }
        
        response = self.http_client.post("/v1/causal/optimize-budget", json=payload)
        response.raise_for_status()
        return response.json()
    
    def measure_incrementality(
        self,
        test_data: Union[pd.DataFrame, List[Dict[str, Any]]],
        control_data: Union[pd.DataFrame, List[Dict[str, Any]]],
        method: str = "geo_experiment"
    ) -> Dict[str, Any]:
        """Measure true incrementality of marketing activities.
        
        Args:
            test_data: Data from test group/regions
            control_data: Data from control group/regions
            method: Measurement method ("geo_experiment", "holdout", "synthetic_control")
        
        Returns:
            Incrementality measurements and confidence intervals
        """
        if isinstance(test_data, pd.DataFrame):
            test_data = test_data.to_dict("records")
        if isinstance(control_data, pd.DataFrame):
            control_data = control_data.to_dict("records")
        
        payload = {
            "test_data": test_data,
            "control_data": control_data,
            "method": method
        }
        
        response = self.http_client.post("/v1/causal/incrementality", json=payload)
        response.raise_for_status()
        return response.json()
```

---

## 🔌 Integration Features

### Pandas Integration

```python
# liftos/integrations/pandas.py
import pandas as pd
from typing import Any, Dict, List
from ..client import Client

@pd.api.extensions.register_dataframe_accessor("liftos")
class LiftOSAccessor:
    """Pandas accessor for LiftOS functionality."""
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._client = None
    
    def set_client(self, client: Client):
        """Set LiftOS client for this accessor."""
        self._client = client
        return self
    
    def analyze_products(self, description_column: str = "description") -> pd.DataFrame:
        """Analyze products in DataFrame using Surfacing module."""
        if self._client is None:
            raise ValueError("LiftOS client not set. Use df.liftos.set_client(client) first.")
        
        surfacing = self._client.surfacing()
        results = []
        
        for _, row in self._obj.iterrows():
            try:
                analysis = surfacing.analyze_product(row[description_column])
                results.append({
                    "revenue_impact": analysis.revenue_impact,
                    "conversion_lift": analysis.conversion_lift,
                    "seo_score": analysis.seo_score,
                    "competitive_score": analysis.competitive_score
                })
            except Exception as e:
                results.append({
                    "revenue_impact": 0,
                    "conversion_lift": 0,
                    "seo_score": 0,
                    "competitive_score": 0,
                    "error": str(e)
                })
        
        result_df = pd.DataFrame(results)
        return pd.concat([self._obj, result_df], axis=1)
    
    def causal_attribution(
        self,
        spend_columns: List[str],
        revenue_column: str = "revenue",
        date_column: str = "date"
    ) -> Dict[str, Any]:
        """Perform causal attribution analysis on DataFrame."""
        if self._client is None:
            raise ValueError("LiftOS client not set. Use df.liftos.set_client(client) first.")
        
        causal = self._client.causal()
        
        # Prepare data for analysis
        campaign_data = self._obj[spend_columns + [revenue_column, date_column]].to_dict("records")
        
        return causal.analyze_attribution(campaign_data)
```

### Jupyter Notebook Helpers

```python
# liftos/integrations/jupyter.py
from IPython.display import display, HTML, Javascript
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List

class JupyterHelper:
    """Helper functions for Jupyter notebook integration."""
    
    @staticmethod
    def display_analysis_results(results: Dict[str, Any]):
        """Display analysis results with rich formatting."""
        html = f"""
        <div style="border: 1px solid #ddd; padding: 20px; border-radius: 8px; margin: 10px 0;">
            <h3 style="color: #1463FF;">📊 LiftOS Analysis Results</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                <div>
                    <h4>💰 Revenue Impact</h4>
                    <p style="font-size: 24px; color: #00C38C; font-weight: bold;">
                        ${results.get('revenue_impact', 0):,.0f}/month
                    </p>
                </div>
                <div>
                    <h4>📈 Conversion Lift</h4>
                    <p style="font-size: 24px; color: #1463FF; font-weight: bold;">
                        +{results.get('conversion_lift', 0):.1%}
                    </p>
                </div>
            </div>
        </div>
        """
        display(HTML(html))
    
    @staticmethod
    def plot_attribution_waterfall(attribution_data: Dict[str, Any]):
        """Create waterfall chart for attribution analysis."""
        channels = list(attribution_data.get('channel_attribution', {}).keys())
        values = [data.get('true_contribution', 0) 
                 for data in attribution_data.get('channel_attribution', {}).values()]
        
        fig = go.Figure(go.Waterfall(
            name="Attribution",
            orientation="v",
            measure=["relative"] * len(channels),
            x=channels,
            textposition="outside",
            text=[f"${v:,.0f}" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="True Causal Attribution by Channel",
            showlegend=False,
            height=500
        )
        
        fig.show()
    
    @staticmethod
    def create_optimization_dashboard(optimization_data: List[Dict[str, Any]]):
        """Create interactive optimization dashboard."""
        # Implementation for interactive dashboard
        pass
```

---

## 🧪 Testing Strategy

### Unit Tests

```python
# tests/test_surfacing.py
import pytest
from unittest.mock import Mock, patch
from liftos.modules.surfacing import SurfacingClient, ProductAnalysisResult
from liftos.exceptions import ValidationError, LiftOSError

class TestSurfacingClient:
    
    @pytest.fixture
    def mock_http_client(self):
        return Mock()
    
    @pytest.fixture
    def surfacing_client(self, mock_http_client):
        return SurfacingClient(mock_http_client, Mock())
    
    def test_analyze_product_success(self, surfacing_client, mock_http_client):
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "revenue_impact": 50000,
            "conversion_lift": 0.23,
            "seo_score": 87,
            "competitive_score": 8.2,
            "optimization_recommendations": [],
            "keyword_opportunities": [],
            "sentiment_analysis": {},
            "confidence_score": 0.89,
            "processing_time": 1.2
        }
        mock_http_client.post.return_value = mock_response
        
        result = surfacing_client.analyze_product("Test product description")
        
        assert isinstance(result, ProductAnalysisResult)
        assert result.revenue_impact == 50000
        assert result.conversion_lift == 0.23
        
    def test_analyze_product_validation_error(self, surfacing_client):
        with pytest.raises(ValidationError):
            surfacing_client.analyze_product("")  # Empty description
        
        with pytest.raises(ValidationError):
            surfacing_client.analyze_product("short")  # Too short
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
import os
from liftos import Client
from liftos.exceptions import AuthenticationError

class TestIntegration:
    
    @pytest.fixture
    def client(self):
        api_key = os.getenv("LIFTOS_TEST_API_KEY")
        if not api_key:
            pytest.skip("LIFTOS_TEST_API_KEY not set")
        return Client(api_key=api_key, base_url="https://api-test.liftos.com")
    
    def test_health_check(self, client):
        assert client.health_check() is True
    
    def test_surfacing_analysis(self, client):
        result = client.surfacing().analyze_product(
            "Premium wireless headphones with noise cancellation"
        )
        assert result.revenue_impact > 0
        assert 0 <= result.conversion_lift <= 1
        assert 0 <= result.seo_score <= 100
    
    def test_causal_attribution(self, client):
        # Test with sample data
        campaigns = [
            {"channel": "google", "spend": 1000, "revenue": 4000, "date": "2024-01-01"},
            {"channel": "facebook", "spend": 800, "revenue": 2400, "date": "2024-01-01"},
        ]
        
        result = client.causal().analyze_attribution(campaigns)
        assert result.true_roas > 0
        assert len(result.channel_attribution) > 0
```

---

## 📦 Distribution & Deployment

### PyPI Publishing Workflow

```yaml
# .github/workflows/publish.yml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

### Version Management

```python
# liftos/__version__.py
__version__ = "1.0.0"
__author__ = "LiftOS Team"
__email__ = "sdk@liftos.com"
__description__ = "Causal AI platform for marketing optimization"
```

### Documentation Generation

```python
# docs/generate_docs.py
import liftos
from liftos.modules import surfacing, causal, llm, memory

def generate_api_docs():
    """Generate API documentation from docstrings."""
    # Implementation for auto-generating docs
    pass

if __name__ == "__main__":
    generate_api_docs()
```

---

## 🚀 Launch Strategy

### Phase 1: Core Package (Weeks 1-2)
- [ ] Implement core client and authentication
- [ ] Build Surfacing and Causal modules
- [ ] Create basic error handling
- [ ] Write unit tests
- [ ] Set up CI/CD pipeline

### Phase 2: Enhanced Features (Weeks 3-4)
- [ ] Add LLM and Memory modules
- [ ] Implement pandas integration
- [ ] Create Jupyter helpers
- [ ] Add async support
- [ ] Performance optimization

### Phase 3: Advanced Integration (Weeks 5-6)
- [ ] Streamlit components
- [ ] Custom model training
- [ ] Advanced analytics
- [ ] Enterprise features
- [ ] Comprehensive documentation

### Phase 4: Community & Ecosystem (Weeks 7-8)
- [ ] Example notebooks
- [ ] Tutorial videos
- [ ] Community forum
- [ ] Plugin architecture
- [ ] Third-party integrations

---

## 📊 Success Metrics

### Technical Metrics
- **Package Downloads**: Target 10K+ downloads in first month
- **API Response Time**: <2 seconds for 95% of requests
- **Error Rate**: <1% of API calls
- **Test Coverage**: >90% code coverage

### Business Metrics
- **User Adoption**: 500+ active users in first quarter
- **Revenue Impact**: $1M+ in customer value generated
- **Customer Satisfaction**: >4.5/5 stars on PyPI
- **Community Growth**: 1000+ GitHub stars

### Usage Analytics
```python
# Built-in analytics tracking
from liftos.analytics import track_usage

@track_usage
def analyze_product(self, description: str):
    # Track API usage for optimization
    pass
```

---

This implementation plan provides a complete roadmap for creating a production-ready LiftOS PyPI package that delivers immediate value to data scientists while building toward comprehensive causal AI capabilities.