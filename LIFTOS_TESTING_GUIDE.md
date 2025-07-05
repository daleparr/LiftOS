# LiftOS Testing Guide
## Verify Your System and Get Immediate Value

## Quick System Verification

### **Step 1: Check System Status**

```bash
# Check if all services are running
docker-compose ps

# Check overall system health
curl http://localhost:8000/health/detailed
```

**Expected Response:**
```json
{
  "status": "healthy",
  "dependencies": {
    "auth": "healthy",
    "billing": "healthy", 
    "memory": "healthy",
    "observability": "healthy",
    "registry": "healthy"
  },
  "uptime": 3600
}
```

### **Step 2: Test Authentication**

```bash
# Test authentication (you may need to create a user first)
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "testpassword123",
    "name": "Test User"
  }'

# Get authentication token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com", 
    "password": "testpassword123"
  }'
```

**Save the token for subsequent requests:**
```bash
export LIFTOS_TOKEN="your-jwt-token-here"
```

---

## Test 1: Surfacing Module - Product Analysis

### **Test Product Analysis**

```bash
curl -X POST http://localhost:8000/modules/surfacing/api/v1/analyze \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "product_data": {
      "title": "Wireless Bluetooth Headphones",
      "description": "Good quality headphones with wireless connectivity",
      "price": 99.99,
      "category": "Electronics",
      "brand": "TechBrand"
    },
    "analysis_type": "comprehensive",
    "include_hybrid_analysis": true,
    "optimization_level": "standard"
  }'
```

### **Expected Value:**
- **Optimization recommendations** for title, description, pricing
- **Visibility score** showing current search performance
- **Revenue impact estimates** for suggested improvements
- **Memory integration** storing analysis for future reference

### **Test Product Optimization**

```bash
curl -X POST http://localhost:8000/modules/surfacing/api/v1/optimize \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "product_data": {
      "title": "Headphones",
      "description": "Audio device",
      "price": 99.99
    },
    "optimization_level": "aggressive"
  }'
```

### **Expected Results:**
- **Improved titles** with better SEO keywords
- **Enhanced descriptions** with compelling copy
- **Revenue impact projections** (e.g., "+$15K monthly")
- **Confidence scores** for each recommendation

---

## Test 2: Causal Module - Marketing Attribution

### **Test Attribution Analysis**

```bash
curl -X POST http://localhost:8000/modules/causal/api/v1/attribution/analyze \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_data": {
      "google_ads": {
        "spend": 10000,
        "impressions": 500000,
        "clicks": 12500,
        "conversions": 250
      },
      "facebook_ads": {
        "spend": 8000,
        "impressions": 400000,
        "clicks": 10000,
        "conversions": 180
      },
      "email_marketing": {
        "spend": 2000,
        "sends": 50000,
        "opens": 12500,
        "conversions": 75
      }
    },
    "conversion_data": {
      "total_revenue": 125000,
      "total_conversions": 505,
      "attribution_window_days": 30
    },
    "user_id": "test_user_123",
    "attribution_window": 30,
    "model_type": "marketing_mix_model",
    "platforms": ["google_ads", "facebook_ads", "email_marketing"]
  }'
```

### **Expected Value:**
- **True causal attribution** vs. last-click attribution
- **Incremental lift measurements** for each channel
- **Over-attribution detection** (e.g., "claiming 240% of actual sales")
- **Efficiency scores** showing real ROI per channel

### **Test Budget Optimization**

```bash
curl -X POST http://localhost:8000/modules/causal/api/v1/optimization/budget \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "current_allocation": {
      "google_ads": 10000,
      "facebook_ads": 8000,
      "email_marketing": 2000,
      "display_ads": 5000
    },
    "total_budget": 25000,
    "user_id": "test_user_123",
    "optimization_goal": "roi",
    "time_horizon": 30
  }'
```

### **Expected Results:**
- **Optimized budget allocation** based on causal analysis
- **ROI improvement projections** (e.g., "+34% ROI increase")
- **Specific reallocation recommendations** with rationale
- **Revenue impact estimates** for the optimization

---

## Test 3: Memory System Integration

### **Test Memory Search**

```bash
curl -X GET "http://localhost:8000/modules/surfacing/api/v1/memory/search?query=headphones%20optimization&search_type=hybrid" \
  -H "Authorization: Bearer $LIFTOS_TOKEN"
```

### **Test Memory Storage Verification**

```bash
# Check if previous analyses are stored
curl -X POST http://localhost:8000/memory/search \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "product analysis wireless headphones",
    "search_type": "hybrid",
    "limit": 5
  }'
```

---

## Test 4: Advanced Causal Modeling

### **Test Causal Model Creation**

```bash
curl -X POST http://localhost:8000/modules/lift_causal/api/v1/models \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Marketing Attribution Model",
    "description": "Simple test model for marketing attribution",
    "variables": [
      "google_ads_spend",
      "facebook_ads_spend",
      "email_campaigns",
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
    ]
  }'
```

### **Test Causal Analysis**

```bash
# Use the model_id from the previous response
curl -X POST http://localhost:8000/modules/lift_causal/api/v1/analysis \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "causal_model_1704312240",
    "intervention": {
      "google_ads_spend": 15000,
      "email_campaigns": 8
    },
    "target_variables": ["revenue", "conversions"],
    "confidence_level": 0.95
  }'
```

---

## Real-World Test Scenarios

### **Scenario 1: E-commerce Product Optimization**

**Goal:** Optimize product listings for better visibility and sales

```bash
# Test with actual product data
curl -X POST http://localhost:8000/modules/surfacing/api/v1/analyze \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-actual-store.com/product/your-product",
    "analysis_type": "comprehensive",
    "include_hybrid_analysis": true,
    "optimization_level": "aggressive"
  }'
```

**Expected Immediate Value:**
- Identify products losing $X daily due to poor visibility
- Get specific optimization recommendations
- See projected revenue impact of changes
- Store insights for team collaboration

### **Scenario 2: Marketing Attribution Reality Check**

**Goal:** Discover true attribution vs. platform-reported attribution

```bash
# Upload your actual campaign data
curl -X POST http://localhost:8000/modules/causal/api/v1/attribution/analyze \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_data": {
      "google_ads": {
        "spend": YOUR_ACTUAL_GOOGLE_SPEND,
        "conversions": YOUR_ACTUAL_GOOGLE_CONVERSIONS
      },
      "facebook_ads": {
        "spend": YOUR_ACTUAL_FB_SPEND,
        "conversions": YOUR_ACTUAL_FB_CONVERSIONS
      }
    },
    "conversion_data": {
      "total_revenue": YOUR_ACTUAL_REVENUE,
      "total_conversions": YOUR_ACTUAL_CONVERSIONS
    },
    "user_id": "your_user_id"
  }'
```

**Expected Revelations:**
- Discover over-attribution (e.g., platforms claiming 240% of actual sales)
- Find hidden high-performing channels
- Identify budget reallocation opportunities worth $X monthly
- Get causal confidence scores for each attribution claim

---

## Success Metrics to Track

### **Immediate Value Indicators**

1. **Product Optimization Impact**
   - Revenue increase projections from title/description improvements
   - Visibility score improvements
   - Number of optimization opportunities identified

2. **Attribution Accuracy**
   - Difference between platform-reported and causal attribution
   - Over-attribution percentage discovered
   - Budget reallocation opportunities identified

3. **System Performance**
   - API response times (<2 seconds for analysis)
   - Memory system search accuracy
   - Cross-module data integration success

### **Business Impact Tracking**

```bash
# Track optimization implementations
echo "Product optimizations applied: X"
echo "Projected monthly revenue increase: $X"
echo "Attribution corrections made: X channels"
echo "Budget reallocated based on causal insights: $X"
```

---

## Troubleshooting Common Issues

### **Authentication Issues**

```bash
# Check auth service health
curl http://localhost:8000/auth/health

# Verify token format
echo $LIFTOS_TOKEN | cut -d'.' -f2 | base64 -d
```

### **Module Connectivity Issues**

```bash
# Check module registration
curl http://localhost:8000/registry/modules

# Check individual module health
curl http://localhost:8000/modules/surfacing/health
curl http://localhost:8000/modules/causal/health
```

### **Memory System Issues**

```bash
# Check memory service health
curl http://localhost:8000/memory/health

# Test memory connectivity
curl -X POST http://localhost:8000/memory/store \
  -H "Authorization: Bearer $LIFTOS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content": "test memory storage", "memory_type": "test"}'
```

---

## Next Steps After Testing

### **If Tests Pass:**
1. **Start with real data**: Use your actual product URLs and campaign data
2. **Implement recommendations**: Apply the optimization suggestions
3. **Track results**: Monitor revenue/performance improvements
4. **Scale usage**: Automate API calls for continuous optimization

### **If Tests Fail:**
1. **Check system status**: Ensure all Docker containers are running
2. **Verify authentication**: Confirm user creation and token generation
3. **Review logs**: Check container logs for specific error messages
4. **Contact support**: Share specific error responses for debugging

---

## Expected Timeline for Value

- **Day 1**: System verification and first product analysis
- **Week 1**: Implement 5-10 product optimizations
- **Week 2**: Complete attribution analysis and budget reallocation
- **Month 1**: Measure actual revenue/performance improvements
- **Month 2**: Scale to full product catalog and campaign portfolio

**The system is designed to deliver measurable value within the first week of testing!**