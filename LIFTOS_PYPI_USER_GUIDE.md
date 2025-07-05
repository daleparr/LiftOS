# LiftOS Python SDK - Data Science User Guide

**Transform your marketing data into causal insights and actionable optimizations**

LiftOS is the first causal AI platform designed specifically for data scientists who need to move beyond correlation to true causation in marketing analytics. This guide shows you how to use LiftOS in your favorite IDE to generate measurable business impact.

---

## 🚀 Quick Start

### Installation
```bash
pip install liftos
```

### 30-Second Setup
```python
import liftos

# Initialize client
client = liftos.Client(api_key="your-api-key")

# Analyze a product in 3 lines
surfacing = client.surfacing()
result = surfacing.analyze_product("iPhone 15 Pro Max 256GB")
print(f"Optimization potential: ${result.revenue_impact:,.0f}/month")
```

---

## 🔬 Core Modules & Business Value

### 1. Surfacing Module - Product Intelligence

**What it does**: Transforms product descriptions into revenue optimization insights using advanced NLP and market intelligence.

#### **Key Outputs & Business Metrics**

```python
from liftos import Surfacing

surfacing = Surfacing(api_key="your-key")
result = surfacing.analyze_product(
    product_description="Premium wireless headphones with noise cancellation",
    competitor_data=["Sony WH-1000XM5", "Bose QuietComfort 45"],
    price_point=299.99
)

# Business-Critical Outputs
print(f"Revenue Impact: ${result.revenue_impact:,.0f}/month")
print(f"Conversion Lift: +{result.conversion_lift:.1%}")
print(f"Search Visibility Score: {result.seo_score}/100")
print(f"Competitive Advantage: {result.competitive_score}/10")
```

#### **Detailed Output Structure**
```python
{
    "revenue_impact": 45000,           # Monthly revenue increase potential
    "conversion_lift": 0.23,           # Expected conversion rate improvement
    "seo_score": 87,                   # Search engine optimization score (0-100)
    "competitive_score": 8.2,          # Competitive positioning (0-10)
    "optimization_recommendations": [
        {
            "change": "Add 'studio-quality' to description",
            "impact": "$12,000/month",
            "confidence": 0.89,
            "effort": "low"
        }
    ],
    "keyword_opportunities": [
        {
            "keyword": "wireless noise cancelling headphones",
            "search_volume": 50000,
            "difficulty": 0.65,
            "revenue_potential": 18000
        }
    ],
    "sentiment_analysis": {
        "overall_sentiment": 0.82,      # Positive sentiment score (0-1)
        "emotional_triggers": ["premium", "studio-quality", "comfort"],
        "pain_points_addressed": ["noise", "battery life", "comfort"]
    }
}
```

#### **Hyperparameters & Tuning**
```python
# Advanced configuration for different use cases
surfacing = Surfacing(
    market_focus="premium",           # "budget", "mid-tier", "premium", "luxury"
    analysis_depth="comprehensive",   # "quick", "standard", "comprehensive"
    competitor_analysis=True,         # Include competitive intelligence
    seo_optimization=True,           # Include SEO recommendations
    sentiment_weighting=0.7,         # Weight of sentiment in scoring (0-1)
    market_data_freshness="7d"       # Use market data from last 7 days
)
```

### 2. Causal Module - Marketing Attribution

**What it does**: Reveals true causal relationships in marketing spend, eliminating attribution fraud and optimizing budget allocation.

#### **Key Outputs & Business Metrics**

```python
from liftos import Causal
import pandas as pd

causal = Causal(api_key="your-key")

# Load your marketing data
campaign_data = pd.read_csv("marketing_campaigns.csv")

result = causal.analyze_attribution(
    campaigns=campaign_data,
    revenue_data=revenue_df,
    time_period="90d"
)

# Business-Critical Outputs
print(f"Attribution Accuracy: {result.accuracy_score:.1%}")
print(f"Budget Waste Identified: ${result.wasted_spend:,.0f}")
print(f"Optimal Reallocation: ${result.reallocation_opportunity:,.0f}")
print(f"True ROAS: {result.true_roas:.2f}x")
```

#### **Detailed Output Structure**
```python
{
    "accuracy_score": 0.94,           # Attribution model accuracy vs ground truth
    "wasted_spend": 125000,           # Monthly budget being wasted
    "reallocation_opportunity": 340000, # Revenue opportunity from reallocation
    "true_roas": 4.2,                 # Causal ROAS (not correlation-based)
    "channel_attribution": {
        "google_ads": {
            "attributed_revenue": 450000,
            "true_contribution": 380000,  # Causal contribution
            "efficiency_score": 0.84,
            "recommended_budget": 95000
        },
        "facebook_ads": {
            "attributed_revenue": 200000,
            "true_contribution": 280000,  # Undervalued channel
            "efficiency_score": 1.4,
            "recommended_budget": 140000
        }
    },
    "causal_insights": [
        {
            "insight": "Facebook drives 40% more revenue than attributed",
            "confidence": 0.91,
            "action": "Increase Facebook budget by $45K",
            "expected_lift": "$67K additional revenue"
        }
    ],
    "incrementality_analysis": {
        "google_ads": 0.78,             # 78% of attributed revenue is incremental
        "facebook_ads": 0.92,           # 92% of attributed revenue is incremental
        "email": 0.45                   # 45% of attributed revenue is incremental
    }
}
```

#### **Hyperparameters & Tuning**
```python
causal = Causal(
    attribution_model="causal_forest",    # "last_touch", "first_touch", "linear", "causal_forest"
    confidence_threshold=0.85,            # Minimum confidence for recommendations
    time_window="90d",                    # Attribution window
    incrementality_method="geo_experiment", # "geo_experiment", "holdout", "synthetic_control"
    seasonality_adjustment=True,          # Adjust for seasonal patterns
    external_factors=["weather", "events"] # Include external variables
)
```

### 3. LLM Module - Content Intelligence

**What it does**: Evaluates and optimizes LLM performance for business outcomes, not just technical metrics.

#### **Key Outputs & Business Metrics**

```python
from liftos import LLM

llm = LLM(api_key="your-key")

# Evaluate model performance on business metrics
result = llm.evaluate_model(
    model_name="gpt-4",
    use_case="customer_support",
    test_data=customer_conversations,
    business_metrics=["satisfaction", "resolution_rate", "revenue_impact"]
)

# Business-Critical Outputs
print(f"Customer Satisfaction: {result.satisfaction_score:.1f}/10")
print(f"Revenue Impact: ${result.revenue_impact:,.0f}/month")
print(f"Cost Efficiency: {result.cost_per_interaction:.2f}")
print(f"Business ROI: {result.business_roi:.1f}x")
```

#### **Detailed Output Structure**
```python
{
    "satisfaction_score": 8.4,        # Customer satisfaction (1-10)
    "revenue_impact": 89000,          # Monthly revenue impact
    "cost_per_interaction": 0.23,     # Cost per customer interaction
    "business_roi": 12.4,             # Business ROI multiple
    "performance_metrics": {
        "resolution_rate": 0.87,      # First-contact resolution rate
        "escalation_rate": 0.08,      # Rate of escalation to humans
        "response_quality": 0.91,     # Quality score (0-1)
        "response_time": 1.2          # Average response time (seconds)
    },
    "optimization_recommendations": [
        {
            "change": "Fine-tune on industry-specific data",
            "expected_improvement": "+15% satisfaction",
            "cost": "$2,400",
            "roi": "8.2x"
        }
    ],
    "a_b_test_results": {
        "model_a": "gpt-4",
        "model_b": "claude-3",
        "winner": "gpt-4",
        "confidence": 0.94,
        "lift": "+12% revenue per conversation"
    }
}
```

#### **Hyperparameters & Tuning**
```python
llm = LLM(
    evaluation_framework="business_impact", # "technical", "business_impact", "hybrid"
    test_size=1000,                        # Number of test interactions
    metrics_weight={                       # Weight different business metrics
        "satisfaction": 0.4,
        "revenue": 0.4,
        "cost": 0.2
    },
    comparison_models=["gpt-4", "claude-3", "custom-model"],
    real_time_monitoring=True              # Monitor performance in production
)
```

### 4. Memory Module - Organizational Intelligence

**What it does**: Creates persistent organizational memory that learns from every interaction and decision.

#### **Key Outputs & Business Metrics**

```python
from liftos import Memory

memory = Memory(api_key="your-key")

# Store and retrieve organizational insights
memory.store_insight(
    key="q4_campaign_learnings",
    data={
        "campaign": "holiday_2024",
        "learnings": ["Video ads outperformed static by 34%", "Mobile conversion peaked at 8pm"],
        "budget_allocation": {"video": 0.6, "static": 0.4},
        "roi": 4.2
    },
    tags=["campaign", "q4", "video", "mobile"]
)

# Retrieve relevant insights for new campaigns
insights = memory.search_insights(
    query="video ad performance mobile",
    context="planning_q1_campaign"
)

# Business-Critical Outputs
print(f"Relevant Insights Found: {len(insights.results)}")
print(f"Confidence Score: {insights.avg_confidence:.2f}")
print(f"Potential Value: ${insights.estimated_value:,.0f}")
```

#### **Detailed Output Structure**
```python
{
    "results": [
        {
            "insight": "Video ads outperformed static by 34% in Q4",
            "confidence": 0.92,
            "relevance": 0.87,
            "source": "q4_campaign_learnings",
            "date": "2024-12-15",
            "business_impact": "$45,000 additional revenue",
            "actionable_recommendation": "Allocate 60% budget to video for Q1"
        }
    ],
    "avg_confidence": 0.89,
    "estimated_value": 67000,            # Estimated value of applying insights
    "knowledge_gaps": [
        "Limited data on video performance for B2B audience",
        "No insights on video length optimization"
    ],
    "trending_patterns": [
        "Video content consistently outperforming static",
        "Mobile optimization becoming critical"
    ]
}
```

---

## 💻 IDE Integration Examples

### Jupyter Notebook Workflow

```python
# Cell 1: Setup and Data Loading
import liftos
import pandas as pd
import matplotlib.pyplot as plt

client = liftos.Client(api_key="your-key")

# Load your data
products_df = pd.read_csv("products.csv")
campaigns_df = pd.read_csv("campaigns.csv")

# Cell 2: Product Analysis
surfacing = client.surfacing()

# Analyze all products
products_df['optimization'] = products_df['description'].apply(
    lambda x: surfacing.analyze_product(x)
)

# Extract revenue impact
products_df['revenue_potential'] = products_df['optimization'].apply(
    lambda x: x.revenue_impact
)

# Cell 3: Visualization
plt.figure(figsize=(12, 6))
plt.bar(products_df['product_name'], products_df['revenue_potential'])
plt.title('Revenue Optimization Potential by Product')
plt.xticks(rotation=45)
plt.ylabel('Monthly Revenue Potential ($)')
plt.show()

# Cell 4: Causal Analysis
causal = client.causal()
attribution = causal.analyze_attribution(campaigns_df)

print(f"Total Budget Waste: ${attribution.wasted_spend:,.0f}")
print(f"Reallocation Opportunity: ${attribution.reallocation_opportunity:,.0f}")
```

### VSCode with LiftOS Extension

```python
# main.py - Production data science workflow
from liftos import Client
import pandas as pd
from datetime import datetime, timedelta

def daily_optimization_report():
    """Generate daily optimization report for marketing team"""
    
    client = Client(api_key=os.getenv("LIFTOS_API_KEY"))
    
    # Get yesterday's data
    yesterday = datetime.now() - timedelta(days=1)
    campaigns = load_campaign_data(yesterday)
    
    # Run causal analysis
    causal = client.causal()
    results = causal.analyze_attribution(campaigns)
    
    # Generate recommendations
    recommendations = []
    for channel, data in results.channel_attribution.items():
        if data.efficiency_score < 0.8:
            recommendations.append({
                "channel": channel,
                "action": f"Reduce budget by ${data.current_budget - data.recommended_budget:,.0f}",
                "impact": f"Save ${data.waste_amount:,.0f}/month"
            })
    
    return recommendations

# Auto-run with VSCode tasks
if __name__ == "__main__":
    recs = daily_optimization_report()
    print(f"Generated {len(recs)} optimization recommendations")
```

### Google Colab - Zero Setup Analysis

```python
# Install and analyze in one notebook
!pip install liftos

import liftos
from google.colab import files

# Upload your data
uploaded = files.upload()

# Instant analysis
client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()

# Analyze uploaded product data
import pandas as pd
df = pd.read_csv(list(uploaded.keys())[0])

results = []
for _, row in df.iterrows():
    analysis = surfacing.analyze_product(row['description'])
    results.append({
        'product': row['name'],
        'current_revenue': row['monthly_revenue'],
        'potential_revenue': row['monthly_revenue'] + analysis.revenue_impact,
        'optimization_score': analysis.competitive_score
    })

results_df = pd.DataFrame(results)
print(f"Total optimization potential: ${results_df['potential_revenue'].sum() - results_df['current_revenue'].sum():,.0f}/month")
```

---

## 📊 Performance & Capacity

### API Rate Limits & Capacity

```python
# Check your current usage and limits
client = liftos.Client(api_key="your-key")
usage = client.get_usage_stats()

print(f"Requests this month: {usage.requests_used:,}/{usage.requests_limit:,}")
print(f"Data processed: {usage.data_processed_gb:.1f}GB/{usage.data_limit_gb:.1f}GB")
print(f"Compute hours used: {usage.compute_hours:.1f}/{usage.compute_limit:.1f}")
```

| Plan | Requests/Month | Data Limit | Compute Hours | Price |
|------|----------------|------------|---------------|-------|
| Starter | 10,000 | 10GB | 50 hours | $99/month |
| Professional | 100,000 | 100GB | 500 hours | $499/month |
| Enterprise | Unlimited | Unlimited | Unlimited | Custom |

### Response Times & Optimization

```python
# Performance monitoring
import time

start_time = time.time()
result = surfacing.analyze_product("Your product description")
response_time = time.time() - start_time

print(f"Analysis completed in {response_time:.2f} seconds")

# Batch processing for better performance
products = ["Product 1", "Product 2", "Product 3"]
batch_results = surfacing.batch_analyze(products)  # 3x faster than individual calls
```

**Typical Response Times:**
- Product Analysis: 0.5-2.0 seconds
- Causal Attribution: 2-10 seconds (depending on data size)
- LLM Evaluation: 1-5 seconds
- Memory Search: 0.1-0.5 seconds

---

## 🎯 Business Value Measurement

### ROI Tracking

```python
# Track business impact over time
from liftos.analytics import ROITracker

tracker = ROITracker(client)

# Before optimization
baseline_metrics = {
    "monthly_revenue": 500000,
    "conversion_rate": 0.023,
    "customer_acquisition_cost": 45
}

# After applying LiftOS recommendations
optimized_metrics = {
    "monthly_revenue": 634000,
    "conversion_rate": 0.031,
    "customer_acquisition_cost": 38
}

roi_report = tracker.calculate_roi(baseline_metrics, optimized_metrics)
print(f"LiftOS ROI: {roi_report.roi_multiple:.1f}x")
print(f"Monthly Value Created: ${roi_report.monthly_value:,.0f}")
print(f"Payback Period: {roi_report.payback_months:.1f} months")
```

### Success Metrics by Module

| Module | Key Metric | Typical Improvement | Business Impact |
|--------|------------|-------------------|-----------------|
| Surfacing | Conversion Rate | +15-35% | $50K-200K/month |
| Causal | Attribution Accuracy | +40-60% | $100K-500K/month |
| LLM | Customer Satisfaction | +20-40% | $30K-150K/month |
| Memory | Decision Speed | +50-80% | $25K-100K/month |

---

## 🔧 Advanced Configuration

### Custom Model Training

```python
# Train custom models on your data
from liftos.training import CustomTrainer

trainer = CustomTrainer(client)

# Train surfacing model on your product data
custom_model = trainer.train_surfacing_model(
    training_data=your_product_data,
    validation_split=0.2,
    epochs=50,
    model_name="your_company_surfacing_v1"
)

# Use your custom model
surfacing = client.surfacing(model=custom_model)
```

### Integration with Existing Tools

```python
# Integrate with your existing data pipeline
from liftos.integrations import DataPipeline

pipeline = DataPipeline(client)

# Connect to your data warehouse
pipeline.connect_warehouse(
    type="snowflake",
    connection_string="your_connection_string"
)

# Automated daily analysis
pipeline.schedule_analysis(
    frequency="daily",
    analysis_type="causal_attribution",
    output_destination="your_dashboard"
)
```

---

## 🚨 Error Handling & Debugging

```python
from liftos.exceptions import LiftOSError, RateLimitError, InsufficientDataError

try:
    result = surfacing.analyze_product(product_description)
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after {e.retry_after} seconds")
except InsufficientDataError as e:
    print(f"Need more data: {e.required_fields}")
except LiftOSError as e:
    print(f"LiftOS error: {e.message}")
    print(f"Error code: {e.error_code}")
```

---

## 📞 Support & Resources

- **Documentation**: https://docs.liftos.com
- **API Reference**: https://api.liftos.com/docs
- **Community**: https://community.liftos.com
- **Support**: support@liftos.com
- **Status Page**: https://status.liftos.com

### Getting Help

```python
# Built-in help system
liftos.help("surfacing.analyze_product")
liftos.examples("causal_attribution")
liftos.troubleshoot()  # Diagnostic tool
```

---

**Ready to transform correlation into causation?** Start with `pip install liftos` and begin generating measurable business impact in minutes, not months.