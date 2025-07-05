# LiftOS API Usage Guide
## Get Value from Surfacing & Causal Modules Without Frontend

## Quick Start: System Status

Your LiftOS system is **fully operational** with these services available:

### **Core Services**
- **Gateway**: `http://localhost:8000` - Main API entry point
- **Memory**: `http://localhost:8002` - KSE Memory System (100% operational)
- **Auth**: `http://localhost:8001` - Authentication service

### **Available Modules**
- **Surfacing Module**: `http://localhost:9005` - Product analysis and optimization
- **Causal Module**: `http://localhost:8008` - Marketing attribution and causal inference
- **Lift-Causal Module**: `http://localhost:9001` - Advanced causal modeling

---

## Authentication Setup

All module APIs require authentication. First, get a JWT token:

```bash
# Get authentication token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "your-email@example.com",
    "password": "your-password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

Use this token in all subsequent requests:
```bash
export LIFTOS_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

---

## Surfacing Module: Product Analysis & Optimization

### **1. Analyze a Single Product**

**Endpoint:** `POST /modules/surfacing/api/v1/analyze`

```bash
curl -X POST http://localhost:8000/modules/surfacing/api/v1/analyze \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example-store.com/product/wireless-headphones",
    "analysis_type": "comprehensive",
    "include_hybrid_analysis": true,
    "include_knowledge_graph": true,
    "optimization_level": "standard"
  }'
```

**Response:**
```json
{
  "message": "Product analysis completed successfully",
  "data": {
    "analysis": {
      "product_info": {
        "title": "Wireless Headphones",
        "price": 199.99,
        "category": "Electronics"
      },
      "visibility_score": 0.73,
      "optimization_opportunities": [
        {
          "type": "title_optimization",
          "current": "Wireless Headphones",
          "suggested": "Premium Wireless Bluetooth Headphones with Noise Cancellation",
          "impact_score": 0.85,
          "monthly_revenue_impact": 12000
        }
      ],
      "hybrid_analysis": {
        "semantic_relevance": 0.82,
        "keyword_density": 0.65,
        "competitive_positioning": "strong"
      }
    },
    "memory_context": {
      "similar_products": 3,
      "related_analyses": []
    },
    "correlation_id": "uuid-here",
    "processed_at": "2025-01-03T20:44:00Z"
  }
}
```

### **2. Batch Analyze Multiple Products**

```bash
curl -X POST http://localhost:8000/modules/surfacing/api/v1/batch-analyze \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      {
        "url": "https://example-store.com/product/wireless-headphones",
        "title": "Wireless Headphones",
        "price": 199.99
      },
      {
        "url": "https://example-store.com/product/bluetooth-speaker",
        "title": "Bluetooth Speaker",
        "price": 89.99
      }
    ],
    "optimization_level": "standard"
  }'
```

### **3. Optimize Product Data**

```bash
curl -X POST http://localhost:8000/modules/surfacing/api/v1/optimize \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "product_data": {
      "title": "Wireless Headphones",
      "description": "Good sound quality headphones",
      "price": 199.99,
      "category": "Electronics"
    },
    "optimization_level": "aggressive"
  }'
```

**Response:**
```json
{
  "message": "Product optimization completed successfully",
  "data": {
    "optimization": {
      "original": {
        "title": "Wireless Headphones",
        "description": "Good sound quality headphones"
      },
      "optimized": {
        "title": "Premium Wireless Bluetooth Headphones with Active Noise Cancellation",
        "description": "Experience superior audio quality with our premium wireless Bluetooth headphones featuring active noise cancellation, 30-hour battery life, and crystal-clear sound for music, calls, and gaming."
      },
      "improvements": [
        {
          "field": "title",
          "improvement": "Added premium positioning and key features",
          "seo_impact": "+45% search visibility",
          "conversion_impact": "+23% click-through rate"
        }
      ],
      "revenue_impact": {
        "monthly_increase": 15600,
        "confidence": 0.87
      }
    }
  }
}
```

---

## Causal Module: Marketing Attribution & Analysis

### **1. Analyze Marketing Attribution**

**Endpoint:** `POST /modules/causal/api/v1/attribution/analyze`

```bash
curl -X POST http://localhost:8000/modules/causal/api/v1/attribution/analyze \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_data": {
      "google_ads": {
        "spend": 50000,
        "impressions": 1000000,
        "clicks": 25000,
        "conversions": 500
      },
      "facebook_ads": {
        "spend": 30000,
        "impressions": 800000,
        "clicks": 20000,
        "conversions": 300
      },
      "email_marketing": {
        "spend": 5000,
        "sends": 100000,
        "opens": 25000,
        "conversions": 150
      }
    },
    "conversion_data": {
      "total_revenue": 250000,
      "total_conversions": 950,
      "attribution_window_days": 30
    },
    "user_id": "user_123",
    "attribution_window": 30,
    "model_type": "marketing_mix_model",
    "platforms": ["google_ads", "facebook_ads", "email_marketing"]
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "analysis": {
      "attribution_results": {
        "google_ads": {
          "attributed_revenue": 125000,
          "attributed_conversions": 475,
          "incremental_lift": 0.85,
          "efficiency_score": 2.5,
          "causal_confidence": 0.92
        },
        "facebook_ads": {
          "attributed_revenue": 75000,
          "attributed_conversions": 285,
          "incremental_lift": 0.73,
          "efficiency_score": 2.1,
          "causal_confidence": 0.88
        },
        "email_marketing": {
          "attributed_revenue": 50000,
          "attributed_conversions": 190,
          "incremental_lift": 0.95,
          "efficiency_score": 10.0,
          "causal_confidence": 0.96
        }
      },
      "insights": [
        "Email marketing shows highest incremental lift (95%) despite lowest spend",
        "Google Ads over-attributed by 23% in last-click model",
        "Facebook Ads shows strong assisted conversion value"
      ],
      "recommendations": [
        {
          "action": "increase_email_budget",
          "rationale": "Highest efficiency score and incremental lift",
          "projected_impact": "+$25K monthly revenue"
        }
      ]
    },
    "metadata": {
      "user_id": "user_123",
      "timestamp": "2025-01-03T20:44:00Z",
      "model_type": "marketing_mix_model"
    }
  }
}
```

### **2. Optimize Budget Allocation**

```bash
curl -X POST http://localhost:8000/modules/causal/api/v1/optimization/budget \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "current_allocation": {
      "google_ads": 50000,
      "facebook_ads": 30000,
      "email_marketing": 5000,
      "display_ads": 15000
    },
    "total_budget": 100000,
    "user_id": "user_123",
    "optimization_goal": "roi",
    "time_horizon": 30,
    "constraints": {
      "min_email_budget": 5000,
      "max_single_channel": 60000
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "optimization": {
      "current_allocation": {
        "google_ads": 50000,
        "facebook_ads": 30000,
        "email_marketing": 5000,
        "display_ads": 15000
      },
      "optimized_allocation": {
        "google_ads": 45000,
        "facebook_ads": 25000,
        "email_marketing": 20000,
        "display_ads": 10000
      },
      "performance_improvement": {
        "roi_increase": "34%",
        "revenue_increase": 85000,
        "efficiency_gain": "28%"
      },
      "reallocation_summary": [
        {
          "channel": "email_marketing",
          "change": "+$15,000",
          "rationale": "Highest incremental ROI"
        },
        {
          "channel": "google_ads",
          "change": "-$5,000",
          "rationale": "Diminishing returns at current spend level"
        }
      ]
    }
  }
}
```

### **3. Measure Campaign Lift**

```bash
curl -X POST http://localhost:8000/modules/causal/api/v1/lift/measure \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_id": "holiday_campaign_2024",
    "baseline_data": {
      "period": "2024-11-01 to 2024-11-15",
      "revenue": 180000,
      "conversions": 720,
      "traffic": 45000
    },
    "treatment_data": {
      "period": "2024-11-16 to 2024-11-30",
      "revenue": 285000,
      "conversions": 1140,
      "traffic": 52000
    },
    "user_id": "user_123",
    "measurement_type": "incremental_lift",
    "statistical_method": "difference_in_differences"
  }'
```

---

## Advanced Causal Modeling (Lift-Causal Module)

### **1. Create a Causal Model**

**Endpoint:** `POST /modules/lift_causal/api/v1/models`

```bash
curl -X POST http://localhost:8000/modules/lift_causal/api/v1/models \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "E-commerce Attribution Model",
    "description": "Causal model for e-commerce marketing attribution",
    "variables": [
      "google_ads_spend",
      "facebook_ads_spend", 
      "email_campaigns",
      "organic_traffic",
      "revenue",
      "conversions"
    ],
    "relationships": [
      {
        "cause": "google_ads_spend",
        "effect": "revenue",
        "relationship_type": "positive_linear",
        "strength": 0.75
      },
      {
        "cause": "email_campaigns", 
        "effect": "conversions",
        "relationship_type": "positive_nonlinear",
        "strength": 0.85
      }
    ],
    "metadata": {
      "industry": "e-commerce",
      "model_type": "marketing_attribution"
    }
  }'
```

### **2. Run Causal Analysis**

```bash
curl -X POST http://localhost:8000/modules/lift_causal/api/v1/analysis \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "causal_model_1704312240",
    "intervention": {
      "google_ads_spend": 75000,
      "email_campaigns": 12
    },
    "target_variables": ["revenue", "conversions"],
    "confidence_level": 0.95
  }'
```

---

## Memory System Integration

### **Search Previous Analyses**

```bash
curl -X GET "http://localhost:8000/modules/surfacing/api/v1/memory/search?query=wireless%20headphones&search_type=hybrid" \
  -H "Authorization: Bearer $LIFTOS_TOKEN"
```

### **Get Causal Insights**

```bash
curl -X POST http://localhost:8000/modules/lift_causal/api/v1/insights \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "email marketing attribution performance",
    "context": "e-commerce campaigns"
  }'
```

---

## Practical Workflows

### **Workflow 1: Product Optimization Pipeline**

```bash
# 1. Analyze current product performance
ANALYSIS_RESULT=$(curl -s -X POST http://localhost:8000/modules/surfacing/api/v1/analyze \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://your-store.com/product/item"}')

# 2. Extract optimization recommendations
echo $ANALYSIS_RESULT | jq '.data.analysis.optimization_opportunities'

# 3. Apply optimizations
curl -X POST http://localhost:8000/modules/surfacing/api/v1/optimize \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"product_data": {"title": "Current Title", "description": "Current Description"}}'
```

### **Workflow 2: Attribution Analysis & Budget Optimization**

```bash
# 1. Analyze current attribution
ATTRIBUTION_RESULT=$(curl -s -X POST http://localhost:8000/modules/causal/api/v1/attribution/analyze \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"campaign_data": {...}, "user_id": "user_123"}')

# 2. Get optimization recommendations
curl -X POST http://localhost:8000/modules/causal/api/v1/optimization/budget \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"current_allocation": {...}, "total_budget": 100000, "user_id": "user_123"}'
```

---

## Health Monitoring

### **Check System Health**

```bash
# Overall system health
curl http://localhost:8000/health/detailed

# Individual module health
curl http://localhost:8000/modules/surfacing/health
curl http://localhost:8000/modules/causal/health
curl http://localhost:8000/modules/lift_causal/health
```

---

## Getting Started Checklist

- [ ] **System Running**: Verify all services are up with `docker-compose ps`
- [ ] **Authentication**: Get JWT token from auth service
- [ ] **Test Surfacing**: Run a product analysis on one of your products
- [ ] **Test Causal**: Analyze attribution for your marketing campaigns
- [ ] **Review Results**: Check the optimization recommendations
- [ ] **Apply Changes**: Use the optimization suggestions in your actual systems
- [ ] **Monitor Impact**: Track the revenue/performance improvements

---

## Next Steps

1. **Start with Surfacing**: Analyze 5-10 of your top products to identify optimization opportunities
2. **Attribution Analysis**: Upload your marketing campaign data to get true causal attribution
3. **Budget Optimization**: Use the causal insights to reallocate your marketing budget
4. **Continuous Monitoring**: Set up regular API calls to track performance improvements

**The system is ready to deliver immediate value through these APIs!**