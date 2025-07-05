# LiftOS Business Use Cases - Surfacing & Causal Modules

**10 Real-World Python Workflows That Deliver Measurable Business Impact**

These use cases demonstrate how data scientists, marketing analysts, and product managers can use LiftOS Surfacing and Causal modules to generate immediate, quantifiable business value through simple Python workflows.

---

## 🛍️ Use Case 1: E-commerce Product Optimization

**Business Problem**: Online retailer has 10,000 products with inconsistent descriptions, leading to poor search visibility and low conversion rates.

**LiftOS Solution**: Automated product analysis and optimization at scale.

```python
import liftos
import pandas as pd

# Initialize LiftOS client
client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()

# Load product catalog
products_df = pd.read_csv("product_catalog.csv")

# Analyze all products for optimization opportunities
optimization_results = []
for _, product in products_df.iterrows():
    result = surfacing.analyze_product(
        product_description=product['description'],
        price_point=product['price'],
        competitor_data=product['competitors'].split(','),
        market_focus="premium"
    )
    
    optimization_results.append({
        'product_id': product['id'],
        'current_revenue': product['monthly_revenue'],
        'revenue_potential': result.revenue_impact,
        'conversion_lift': result.conversion_lift,
        'seo_score': result.seo_score,
        'top_recommendation': result.optimization_recommendations[0]['change'],
        'implementation_effort': result.optimization_recommendations[0]['effort']
    })

# Create optimization priority matrix
results_df = pd.DataFrame(optimization_results)
results_df['roi_score'] = results_df['revenue_potential'] / results_df['current_revenue']
results_df = results_df.sort_values('roi_score', ascending=False)

# Business Impact Summary
total_current_revenue = results_df['current_revenue'].sum()
total_potential_revenue = results_df['revenue_potential'].sum()
optimization_opportunity = total_potential_revenue

print(f"📊 BUSINESS IMPACT ANALYSIS")
print(f"Current Monthly Revenue: ${total_current_revenue:,.0f}")
print(f"Optimization Opportunity: ${optimization_opportunity:,.0f}/month")
print(f"ROI Potential: {(optimization_opportunity/total_current_revenue)*100:.1f}% increase")
print(f"Top 10 Products Could Generate: ${results_df.head(10)['revenue_potential'].sum():,.0f}/month")
```

**Expected Business Impact**:
- **Revenue Increase**: 15-35% from optimized product descriptions
- **Implementation Time**: 2 weeks for top 100 products
- **ROI**: 8-12x return on optimization investment
- **Measurable Metrics**: Conversion rate, search ranking, revenue per visitor

---

## 📱 Use Case 2: Mobile App Marketing Attribution

**Business Problem**: Mobile gaming company spends $500K/month across 8 channels but can't determine true incremental value, leading to budget waste.

**LiftOS Solution**: Causal attribution analysis revealing true channel performance.

```python
import liftos
import pandas as pd
from datetime import datetime, timedelta

client = liftos.Client(api_key="your-key")
causal = client.causal()

# Load marketing campaign data
campaigns_df = pd.read_csv("mobile_campaigns.csv")
revenue_df = pd.read_csv("app_revenue.csv")

# Perform causal attribution analysis
attribution_result = causal.analyze_attribution(
    campaigns=campaigns_df,
    revenue_data=revenue_df,
    time_period="90d",
    attribution_model="causal_forest",
    confidence_threshold=0.85
)

# Analyze channel efficiency
channel_analysis = {}
for channel, data in attribution_result.channel_attribution.items():
    channel_analysis[channel] = {
        'current_spend': data['current_budget'],
        'attributed_revenue': data['attributed_revenue'],
        'true_contribution': data['true_contribution'],
        'efficiency_score': data['efficiency_score'],
        'recommended_budget': data['recommended_budget'],
        'waste_amount': data['current_budget'] - data['recommended_budget']
    }

# Create reallocation strategy
reallocation_df = pd.DataFrame(channel_analysis).T
total_waste = reallocation_df['waste_amount'].sum()
undervalued_channels = reallocation_df[reallocation_df['efficiency_score'] > 1.0]

print(f"🎯 CAUSAL ATTRIBUTION INSIGHTS")
print(f"Total Monthly Spend: ${reallocation_df['current_spend'].sum():,.0f}")
print(f"Budget Waste Identified: ${total_waste:,.0f}/month")
print(f"Attribution Accuracy: {attribution_result.accuracy_score:.1%}")
print(f"True ROAS: {attribution_result.true_roas:.2f}x")

print(f"\n💰 REALLOCATION RECOMMENDATIONS:")
for channel in undervalued_channels.index:
    current = reallocation_df.loc[channel, 'current_spend']
    recommended = reallocation_df.loc[channel, 'recommended_budget']
    increase = recommended - current
    print(f"{channel}: Increase by ${increase:,.0f} (+{(increase/current)*100:.1f}%)")

# Calculate expected impact of reallocation
expected_revenue_lift = attribution_result.reallocation_opportunity
current_revenue = sum(data['attributed_revenue'] for data in attribution_result.channel_attribution.values())
lift_percentage = (expected_revenue_lift / current_revenue) * 100

print(f"\n📈 EXPECTED BUSINESS IMPACT:")
print(f"Revenue Lift from Reallocation: ${expected_revenue_lift:,.0f}/month ({lift_percentage:.1f}%)")
print(f"Annual Impact: ${expected_revenue_lift * 12:,.0f}")
print(f"Payback Period: Immediate (budget reallocation)")
```

**Expected Business Impact**:
- **Budget Waste Reduction**: 15-25% of total spend ($75K-125K/month)
- **Revenue Increase**: 20-40% from optimal allocation
- **Attribution Accuracy**: 90%+ vs 45% industry average
- **Implementation**: Immediate budget reallocation

---

## 🏢 Use Case 3: B2B SaaS Content Performance Analysis

**Business Problem**: SaaS company creates 50+ blog posts monthly but can't identify which content drives actual revenue, not just vanity metrics.

**LiftOS Solution**: Content-to-revenue causal analysis with optimization recommendations.

```python
import liftos
import pandas as pd
import numpy as np

client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()
causal = client.causal()

# Load content performance data
content_df = pd.read_csv("blog_content.csv")
leads_df = pd.read_csv("lead_generation.csv")
revenue_df = pd.read_csv("customer_revenue.csv")

# Analyze content optimization potential
content_analysis = []
for _, content in content_df.iterrows():
    # Surfacing analysis for content optimization
    content_result = surfacing.analyze_product(
        product_description=content['title'] + " " + content['excerpt'],
        market_focus="professional",
        analysis_depth="comprehensive"
    )
    
    content_analysis.append({
        'content_id': content['id'],
        'title': content['title'],
        'current_traffic': content['monthly_views'],
        'current_leads': content['leads_generated'],
        'seo_score': content_result.seo_score,
        'optimization_potential': content_result.revenue_impact,
        'keyword_opportunities': len(content_result.keyword_opportunities),
        'sentiment_score': content_result.sentiment_analysis['overall_sentiment']
    })

content_results_df = pd.DataFrame(content_analysis)

# Causal analysis: Content → Leads → Revenue
# Prepare data for causal analysis
content_performance = content_df.merge(leads_df, on='content_id').merge(revenue_df, on='lead_id')

causal_result = causal.analyze_attribution(
    campaigns=content_performance[['content_id', 'views', 'leads', 'revenue', 'publish_date']].to_dict('records'),
    attribution_model="causal_forest",
    time_period="180d"
)

# Identify high-impact content patterns
high_performers = content_results_df[
    (content_results_df['seo_score'] > 80) & 
    (content_results_df['optimization_potential'] > 5000)
].sort_values('optimization_potential', ascending=False)

print(f"📝 CONTENT PERFORMANCE ANALYSIS")
print(f"Total Content Pieces Analyzed: {len(content_df)}")
print(f"High-Potential Content Identified: {len(high_performers)}")
print(f"Average SEO Score: {content_results_df['seo_score'].mean():.1f}/100")
print(f"Total Optimization Opportunity: ${content_results_df['optimization_potential'].sum():,.0f}/month")

print(f"\n🎯 TOP CONTENT OPTIMIZATION OPPORTUNITIES:")
for _, content in high_performers.head(5).iterrows():
    print(f"'{content['title'][:50]}...'")
    print(f"  Current Traffic: {content['current_traffic']:,} views/month")
    print(f"  Revenue Potential: ${content['optimization_potential']:,.0f}/month")
    print(f"  SEO Score: {content['seo_score']}/100")
    print(f"  Keyword Opportunities: {content['keyword_opportunities']}")
    print()

# Content strategy recommendations
avg_optimization = content_results_df['optimization_potential'].mean()
total_opportunity = content_results_df['optimization_potential'].sum()

print(f"📊 STRATEGIC RECOMMENDATIONS:")
print(f"1. Focus on top 20% content: ${high_performers.head(10)['optimization_potential'].sum():,.0f}/month potential")
print(f"2. Average content optimization value: ${avg_optimization:,.0f}/month per piece")
print(f"3. Total content portfolio opportunity: ${total_opportunity:,.0f}/month")
print(f"4. Recommended content creation budget reallocation: Focus on high-SEO topics")
```

**Expected Business Impact**:
- **Lead Quality Improvement**: 25-45% increase in qualified leads
- **Content ROI**: 3-5x improvement in content-to-revenue conversion
- **SEO Performance**: 40-60% increase in organic traffic
- **Resource Optimization**: 30% reduction in low-performing content creation

---

## 🛒 Use Case 4: Retail Inventory Optimization

**Business Problem**: Fashion retailer struggles with inventory decisions, leading to 30% overstock and 15% stockouts, costing $2M annually.

**LiftOS Solution**: Product demand prediction with causal factor analysis.

```python
import liftos
import pandas as pd
from datetime import datetime, timedelta

client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()
causal = client.causal()

# Load inventory and sales data
inventory_df = pd.read_csv("inventory_data.csv")
sales_df = pd.read_csv("sales_history.csv")
marketing_df = pd.read_csv("marketing_campaigns.csv")

# Analyze product appeal and market positioning
product_analysis = []
for _, product in inventory_df.iterrows():
    # Analyze product market appeal
    appeal_result = surfacing.analyze_product(
        product_description=product['description'],
        price_point=product['price'],
        competitor_data=product['similar_products'].split(','),
        market_focus=product['target_segment']
    )
    
    product_analysis.append({
        'sku': product['sku'],
        'category': product['category'],
        'current_stock': product['units_in_stock'],
        'appeal_score': appeal_result.competitive_score,
        'price_optimization': appeal_result.revenue_impact,
        'demand_indicators': appeal_result.sentiment_analysis['overall_sentiment'],
        'seo_visibility': appeal_result.seo_score
    })

product_df = pd.DataFrame(product_analysis)

# Causal analysis: Marketing → Demand → Sales
# Combine sales and marketing data
sales_marketing = sales_df.merge(marketing_df, on='campaign_id', how='left')

demand_attribution = causal.analyze_attribution(
    campaigns=sales_marketing[['sku', 'marketing_spend', 'units_sold', 'revenue', 'date']].to_dict('records'),
    attribution_model="causal_forest",
    time_period="365d"
)

# Create inventory optimization recommendations
inventory_optimization = product_df.merge(
    sales_df.groupby('sku').agg({
        'units_sold': 'sum',
        'revenue': 'sum'
    }).reset_index(),
    on='sku'
)

# Calculate optimization metrics
inventory_optimization['velocity_score'] = (
    inventory_optimization['units_sold'] / inventory_optimization['current_stock']
)
inventory_optimization['revenue_per_unit'] = (
    inventory_optimization['revenue'] / inventory_optimization['units_sold']
)
inventory_optimization['optimization_priority'] = (
    inventory_optimization['appeal_score'] * 
    inventory_optimization['velocity_score'] * 
    inventory_optimization['revenue_per_unit']
)

# Identify optimization opportunities
overstock_risk = inventory_optimization[
    (inventory_optimization['velocity_score'] < 0.5) & 
    (inventory_optimization['current_stock'] > 100)
]

understock_opportunity = inventory_optimization[
    (inventory_optimization['appeal_score'] > 7) & 
    (inventory_optimization['velocity_score'] > 2) &
    (inventory_optimization['current_stock'] < 50)
]

print(f"📦 INVENTORY OPTIMIZATION ANALYSIS")
print(f"Total SKUs Analyzed: {len(inventory_optimization)}")
print(f"Overstock Risk Items: {len(overstock_risk)}")
print(f"Understock Opportunities: {len(understock_opportunity)}")

print(f"\n⚠️ OVERSTOCK RISK (Reduce Inventory):")
overstock_value = (overstock_risk['current_stock'] * overstock_risk['revenue_per_unit']).sum()
print(f"Total Value at Risk: ${overstock_value:,.0f}")
for _, item in overstock_risk.head(5).iterrows():
    reduction = int(item['current_stock'] * 0.6)  # Reduce by 40%
    savings = reduction * item['revenue_per_unit']
    print(f"SKU {item['sku']}: Reduce by {reduction} units (${savings:,.0f} freed capital)")

print(f"\n📈 UNDERSTOCK OPPORTUNITIES (Increase Inventory):")
understock_potential = (understock_opportunity['price_optimization'] * 12).sum()  # Annualized
print(f"Annual Revenue Opportunity: ${understock_potential:,.0f}")
for _, item in understock_opportunity.head(5).iterrows():
    increase = int(item['current_stock'] * 1.5)  # Increase by 50%
    potential = item['price_optimization']
    print(f"SKU {item['sku']}: Increase by {increase} units (${potential:,.0f}/month potential)")

# Calculate total business impact
total_overstock_savings = overstock_value * 0.4  # 40% reduction
total_understock_revenue = understock_potential
inventory_carrying_cost_savings = total_overstock_savings * 0.25  # 25% carrying cost

print(f"\n💰 TOTAL BUSINESS IMPACT:")
print(f"Overstock Reduction Savings: ${total_overstock_savings:,.0f}")
print(f"Carrying Cost Savings: ${inventory_carrying_cost_savings:,.0f}/year")
print(f"Understock Revenue Opportunity: ${total_understock_revenue:,.0f}/year")
print(f"Total Annual Impact: ${total_overstock_savings + inventory_carrying_cost_savings + total_understock_revenue:,.0f}")
```

**Expected Business Impact**:
- **Inventory Efficiency**: 25-40% reduction in overstock
- **Revenue Increase**: 15-30% from optimized stock levels
- **Carrying Cost Savings**: $500K-800K annually
- **Stockout Reduction**: 60-80% fewer missed sales opportunities

---

## 📺 Use Case 5: Streaming Service Content Investment

**Business Problem**: Streaming platform spends $100M annually on content but can't predict which shows will drive subscriber retention and acquisition.

**LiftOS Solution**: Content performance prediction with causal subscriber impact analysis.

```python
import liftos
import pandas as pd
import numpy as np

client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()
causal = client.causal()

# Load content and subscriber data
content_df = pd.read_csv("content_library.csv")
subscriber_df = pd.read_csv("subscriber_metrics.csv")
engagement_df = pd.read_csv("viewing_data.csv")

# Analyze content appeal and market positioning
content_analysis = []
for _, content in content_df.iterrows():
    # Analyze content market appeal
    appeal_result = surfacing.analyze_product(
        product_description=f"{content['title']} {content['genre']} {content['description']}",
        competitor_data=content['similar_shows'].split(','),
        market_focus="entertainment"
    )
    
    content_analysis.append({
        'content_id': content['id'],
        'title': content['title'],
        'genre': content['genre'],
        'production_cost': content['budget'],
        'appeal_score': appeal_result.competitive_score,
        'market_potential': appeal_result.revenue_impact,
        'sentiment_score': appeal_result.sentiment_analysis['overall_sentiment'],
        'keyword_strength': len(appeal_result.keyword_opportunities)
    })

content_results_df = pd.DataFrame(content_analysis)

# Causal analysis: Content → Engagement → Subscriber Retention
# Prepare data for causal analysis
content_impact = engagement_df.merge(subscriber_df, on='user_id').merge(content_df, on='content_id')

retention_attribution = causal.analyze_attribution(
    campaigns=content_impact[['content_id', 'watch_time', 'completion_rate', 'subscriber_retained', 'date']].to_dict('records'),
    attribution_model="causal_forest",
    time_period="365d"
)

# Calculate content ROI and investment recommendations
content_roi = content_results_df.merge(
    engagement_df.groupby('content_id').agg({
        'watch_time': 'sum',
        'completion_rate': 'mean',
        'user_id': 'nunique'
    }).rename(columns={'user_id': 'unique_viewers'}).reset_index(),
    on='content_id'
)

# Calculate key performance metrics
content_roi['cost_per_viewer'] = content_roi['production_cost'] / content_roi['unique_viewers']
content_roi['engagement_score'] = content_roi['watch_time'] * content_roi['completion_rate']
content_roi['roi_score'] = (content_roi['market_potential'] * content_roi['appeal_score']) / content_roi['production_cost']

# Identify investment opportunities
high_roi_content = content_roi[content_roi['roi_score'] > content_roi['roi_score'].quantile(0.8)]
underperforming_content = content_roi[content_roi['roi_score'] < content_roi['roi_score'].quantile(0.2)]

print(f"🎬 CONTENT INVESTMENT ANALYSIS")
print(f"Total Content Pieces: {len(content_roi)}")
print(f"Total Production Budget: ${content_roi['production_cost'].sum():,.0f}")
print(f"Average Cost per Viewer: ${content_roi['cost_per_viewer'].mean():.2f}")

print(f"\n🌟 HIGH-ROI CONTENT OPPORTUNITIES:")
high_roi_investment = high_roi_content['production_cost'].sum()
high_roi_potential = high_roi_content['market_potential'].sum()
print(f"Investment in High-ROI Content: ${high_roi_investment:,.0f}")
print(f"Expected Return: ${high_roi_potential:,.0f}/month")
print(f"ROI Multiple: {(high_roi_potential * 12) / high_roi_investment:.1f}x")

for _, content in high_roi_content.head(5).iterrows():
    print(f"'{content['title']}' ({content['genre']})")
    print(f"  Production Cost: ${content['production_cost']:,.0f}")
    print(f"  Market Potential: ${content['market_potential']:,.0f}/month")
    print(f"  Appeal Score: {content['appeal_score']:.1f}/10")
    print(f"  Unique Viewers: {content['unique_viewers']:,}")
    print()

print(f"⚠️ UNDERPERFORMING CONTENT:")
underperforming_cost = underperforming_content['production_cost'].sum()
print(f"Budget Tied Up in Underperforming Content: ${underperforming_cost:,.0f}")
print(f"Opportunity Cost: ${underperforming_cost * 0.15:,.0f}/year (15% alternative return)")

# Genre-level analysis
genre_performance = content_roi.groupby('genre').agg({
    'production_cost': 'sum',
    'market_potential': 'sum',
    'appeal_score': 'mean',
    'unique_viewers': 'sum',
    'roi_score': 'mean'
}).reset_index()

genre_performance['genre_roi'] = (genre_performance['market_potential'] * 12) / genre_performance['production_cost']
best_genres = genre_performance.sort_values('genre_roi', ascending=False)

print(f"\n📊 GENRE INVESTMENT RECOMMENDATIONS:")
for _, genre in best_genres.head(3).iterrows():
    print(f"{genre['genre']}: {genre['genre_roi']:.1f}x ROI")
    print(f"  Total Investment: ${genre['production_cost']:,.0f}")
    print(f"  Annual Return Potential: ${genre['market_potential'] * 12:,.0f}")
    print(f"  Average Appeal Score: {genre['appeal_score']:.1f}/10")
    print()

# Calculate total optimization opportunity
reallocation_savings = underperforming_cost * 0.5  # Reallocate 50% of underperforming budget
reallocation_to_high_roi = reallocation_savings
expected_additional_return = reallocation_to_high_roi * (high_roi_content['roi_score'].mean())

print(f"💰 PORTFOLIO OPTIMIZATION IMPACT:")
print(f"Budget Reallocation Opportunity: ${reallocation_savings:,.0f}")
print(f"Expected Additional Annual Return: ${expected_additional_return * 12:,.0f}")
print(f"Portfolio ROI Improvement: {((expected_additional_return * 12) / content_roi['production_cost'].sum()) * 100:.1f}%")
```

**Expected Business Impact**:
- **Content ROI**: 2-4x improvement through better investment allocation
- **Subscriber Retention**: 15-25% increase from high-appeal content
- **Production Efficiency**: 30-50% reduction in underperforming content investment
- **Revenue Growth**: $20-40M additional annual revenue from optimized content portfolio

---

## 🏥 Use Case 6: Healthcare Marketing Optimization

**Business Problem**: Healthcare system spends $2M annually on patient acquisition but can't measure true effectiveness due to long patient journeys and privacy constraints.

**LiftOS Solution**: Privacy-compliant causal attribution with patient lifetime value optimization.

```python
import liftos
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()
causal = client.causal()

# Load anonymized healthcare marketing data
campaigns_df = pd.read_csv("healthcare_campaigns.csv")
patient_acquisition_df = pd.read_csv("patient_acquisition.csv")  # Anonymized
service_utilization_df = pd.read_csv("service_utilization.csv")  # Anonymized

# Analyze healthcare service messaging effectiveness
service_analysis = []
for _, service in campaigns_df.iterrows():
    # Analyze service positioning and messaging
    messaging_result = surfacing.analyze_product(
        product_description=f"{service['service_name']} {service['description']} {service['benefits']}",
        market_focus="healthcare",
        analysis_depth="comprehensive"
    )
    
    service_analysis.append({
        'service_id': service['service_id'],
        'service_name': service['service_name'],
        'department': service['department'],
        'campaign_spend': service['monthly_spend'],
        'messaging_effectiveness': messaging_result.competitive_score,
        'patient_appeal': messaging_result.sentiment_analysis['overall_sentiment'],
        'optimization_potential': messaging_result.revenue_impact,
        'trust_indicators': len([kw for kw in messaging_result.keyword_opportunities if 'trust' in kw['keyword'].lower()])
    })

service_results_df = pd.DataFrame(service_analysis)

# Causal analysis: Marketing → Patient Acquisition → Lifetime Value
# Prepare aggregated data (privacy-compliant)
marketing_impact = campaigns_df.merge(
    patient_acquisition_df.groupby('campaign_id').agg({
        'new_patients': 'sum',
        'patient_lifetime_value': 'mean',
        'acquisition_cost': 'mean'
    }).reset_index(),
    on='campaign_id'
)

patient_attribution = causal.analyze_attribution(
    campaigns=marketing_impact[['campaign_id', 'monthly_spend', 'new_patients', 'patient_lifetime_value', 'date']].to_dict('records'),
    attribution_model="causal_forest",
    time_period="365d",
    confidence_threshold=0.90  # Higher confidence for healthcare decisions
)

# Calculate patient acquisition efficiency
service_efficiency = service_results_df.merge(
    marketing_impact.groupby('service_id').agg({
        'new_patients': 'sum',
        'patient_lifetime_value': 'mean',
        'acquisition_cost': 'mean'
    }).reset_index(),
    on='service_id',
    how='left'
)

service_efficiency['cost_per_acquisition'] = service_efficiency['campaign_spend'] / service_efficiency['new_patients']
service_efficiency['ltv_to_cac_ratio'] = service_efficiency['patient_lifetime_value'] / service_efficiency['cost_per_acquisition']
service_efficiency['roi_score'] = service_efficiency['ltv_to_cac_ratio'] * service_efficiency['messaging_effectiveness']

# Identify optimization opportunities
high_value_services = service_efficiency[service_efficiency['ltv_to_cac_ratio'] > 3.0]
underperforming_services = service_efficiency[service_efficiency['ltv_to_cac_ratio'] < 1.5]

print(f"🏥 HEALTHCARE MARKETING ANALYSIS")
print(f"Total Marketing Spend: ${service_efficiency['campaign_spend'].sum():,.0f}/month")
print(f"Total New Patients Acquired: {service_efficiency['new_patients'].sum():,.0f}/month")
print(f"Average Patient Lifetime Value: ${service_efficiency['patient_lifetime_value'].mean():,.0f}")
print(f"Average Cost per Acquisition: ${service_efficiency['cost_per_acquisition'].mean():.0f}")

print(f"\n💎 HIGH-VALUE SERVICE OPPORTUNITIES:")
high_value_investment = high_value_services['campaign_spend'].sum()
high_value_patients = high_value_services['new_patients'].sum()
high_value_ltv = high_value_services['patient_lifetime_value'].mean()

print(f"Current Investment: ${high_value_investment:,.0f}/month")
print(f"Patients Acquired: {high_value_patients:,.0f}/month")
print(f"Average LTV/CAC Ratio: {high_value_services['ltv_to_cac_ratio'].mean():.1f}x")

for _, service in high_value_services.head(5).iterrows():
    monthly_value = service['new_patients'] * service['patient_lifetime_value']
    print(f"{service['service_name']} ({service['department']})")
    print(f"  LTV/CAC Ratio: {service['ltv_to_cac_ratio']:.1f}x")
    print(f"  Monthly Patient Value: ${monthly_value:,.0f}")
    print(f"  Messaging Effectiveness: {service['messaging_effectiveness']:.1f}/10")
    print(f"  Optimization Potential: ${service['optimization_potential']:,.0f}/month")
    print()

print(f"⚠️ UNDERPERFORMING SERVICES:")
underperforming_spend = underperforming_services['campaign_spend'].sum()
underperforming_waste = underperforming_spend * 0.6  # 60% waste estimate

print(f"Underperforming Spend: ${underperforming_spend:,.0f}/month")
print(f"Estimated Waste: ${underperforming_waste:,.0f}/month")
print(f"Annual Waste: ${underperforming_waste * 12:,.0f}")

# Department-level analysis
dept_performance = service_efficiency.groupby('department').agg({
    'campaign_spend': 'sum',
    'new_patients': 'sum',
    'patient_lifetime_value': 'mean',
    'ltv_to_cac_ratio': 'mean',
    'messaging_effectiveness': 'mean'
}).reset_index()

dept_performance = dept_performance.sort_values('ltv_to_cac_ratio', ascending=False)

print(f"\n🏢 DEPARTMENT PERFORMANCE RANKING:")
for _, dept in dept_performance.iterrows():
    dept_roi = (dept['new_patients'] * dept['patient_lifetime_value']) / dept['campaign_spend']
    print(f"{dept['department']}")
    print(f"  LTV/CAC Ratio: {dept['ltv_to_cac_ratio']:.1f}x")
    print(f"  Monthly ROI: {dept_roi:.1f}x")
    print(f"  Messaging Score: {dept['messaging_effectiveness']:.1f}/10")
    print(f"  Investment: ${dept['campaign_spend']:,.0f}/month")
    print()

# Calculate reallocation recommendations
reallocation_amount = underperforming_waste
high_value_avg_roi = high_value_services['ltv_to_cac_ratio'].mean()
expected_additional_patients = (reallocation_amount / high_value_services['cost_per_acquisition'].mean())
additional_lifetime_value = expected_additional_patients * high_value_services['patient_lifetime_value'].mean()

print(f"💰 BUDGET REALLOCATION IMPACT:")
print(f"Reallocation Amount: ${reallocation_amount:,.0f}/month")
print(f"Expected Additional Patients: {expected_additional_patients:.0f}/month")
print(f"Additional Lifetime Value: ${additional_lifetime_value:,.0f}")
print(f"Annual Value Creation: ${additional_lifetime_value * 12:,.0f}")
print(f"ROI of Reallocation: {(additional_lifetime_value * 12) / (reallocation_amount * 12):.1f}x")
```

**Expected Business Impact**:
- **Patient Acquisition Efficiency**: 40-60% improvement in cost per acquisition
- **Marketing ROI**: 3-5x improvement through better service targeting
- **Budget Optimization**: $500K-800K annual savings from waste reduction
- **Patient
Lifetime Value**: 25-35% increase through better service matching

---

## 🏭 Use Case 7: Manufacturing Supply Chain Optimization

**Business Problem**: Industrial manufacturer spends $50M annually on suppliers but lacks visibility into true cost drivers and quality impact relationships.

**LiftOS Solution**: Supplier performance analysis with causal cost-quality attribution.

```python
import liftos
import pandas as pd
import numpy as np

client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()
causal = client.causal()

# Load supply chain data
suppliers_df = pd.read_csv("supplier_data.csv")
procurement_df = pd.read_csv("procurement_history.csv")
quality_df = pd.read_csv("quality_metrics.csv")
production_df = pd.read_csv("production_data.csv")

# Analyze supplier positioning and capabilities
supplier_analysis = []
for _, supplier in suppliers_df.iterrows():
    # Analyze supplier value proposition
    supplier_result = surfacing.analyze_product(
        product_description=f"{supplier['company_name']} {supplier['capabilities']} {supplier['certifications']}",
        competitor_data=supplier['competitors'].split(','),
        price_point=supplier['average_unit_cost'],
        market_focus="industrial"
    )
    
    supplier_analysis.append({
        'supplier_id': supplier['id'],
        'supplier_name': supplier['company_name'],
        'category': supplier['category'],
        'annual_spend': supplier['annual_spend'],
        'capability_score': supplier_result.competitive_score,
        'cost_optimization': supplier_result.revenue_impact,
        'reliability_indicators': supplier_result.sentiment_analysis['overall_sentiment'],
        'market_position': supplier_result.seo_score
    })

supplier_results_df = pd.DataFrame(supplier_analysis)

# Causal analysis: Supplier Choice → Quality → Production Efficiency
# Prepare data for causal analysis
supply_impact = procurement_df.merge(quality_df, on='batch_id').merge(production_df, on='batch_id')

quality_attribution = causal.analyze_attribution(
    campaigns=supply_impact[['supplier_id', 'cost_per_unit', 'quality_score', 'production_efficiency', 'date']].to_dict('records'),
    attribution_model="causal_forest",
    time_period="365d"
)

# Calculate supplier efficiency metrics
supplier_performance = supplier_results_df.merge(
    supply_impact.groupby('supplier_id').agg({
        'cost_per_unit': 'mean',
        'quality_score': 'mean',
        'production_efficiency': 'mean',
        'defect_rate': 'mean',
        'delivery_time': 'mean'
    }).reset_index(),
    on='supplier_id'
)

# Calculate total cost of ownership (TCO)
supplier_performance['quality_cost'] = supplier_performance['defect_rate'] * 1000  # $1000 per defect
supplier_performance['delay_cost'] = (supplier_performance['delivery_time'] - 5) * 100  # $100 per day delay
supplier_performance['tco_per_unit'] = (
    supplier_performance['cost_per_unit'] + 
    supplier_performance['quality_cost'] + 
    supplier_performance['delay_cost']
)

supplier_performance['value_score'] = (
    supplier_performance['capability_score'] * 
    supplier_performance['quality_score'] * 
    supplier_performance['production_efficiency']
) / supplier_performance['tco_per_unit']

# Identify optimization opportunities
high_value_suppliers = supplier_performance[supplier_performance['value_score'] > supplier_performance['value_score'].quantile(0.8)]
underperforming_suppliers = supplier_performance[supplier_performance['value_score'] < supplier_performance['value_score'].quantile(0.2)]

print(f"🏭 SUPPLY CHAIN OPTIMIZATION ANALYSIS")
print(f"Total Annual Procurement Spend: ${supplier_performance['annual_spend'].sum():,.0f}")
print(f"Number of Suppliers: {len(supplier_performance)}")
print(f"Average TCO per Unit: ${supplier_performance['tco_per_unit'].mean():.2f}")
print(f"Average Quality Score: {supplier_performance['quality_score'].mean():.1f}/100")

print(f"\n🌟 HIGH-VALUE SUPPLIER OPPORTUNITIES:")
high_value_spend = high_value_suppliers['annual_spend'].sum()
high_value_savings = high_value_suppliers['cost_optimization'].sum()

print(f"Current Spend with High-Value Suppliers: ${high_value_spend:,.0f}")
print(f"Optimization Potential: ${high_value_savings:,.0f}/year")
print(f"Average Value Score: {high_value_suppliers['value_score'].mean():.2f}")

for _, supplier in high_value_suppliers.head(5).iterrows():
    annual_savings = supplier['cost_optimization']
    print(f"{supplier['supplier_name']} ({supplier['category']})")
    print(f"  Annual Spend: ${supplier['annual_spend']:,.0f}")
    print(f"  Value Score: {supplier['value_score']:.2f}")
    print(f"  Quality Score: {supplier['quality_score']:.1f}/100")
    print(f"  TCO per Unit: ${supplier['tco_per_unit']:.2f}")
    print(f"  Optimization Potential: ${annual_savings:,.0f}/year")
    print()

print(f"⚠️ UNDERPERFORMING SUPPLIERS:")
underperforming_spend = underperforming_suppliers['annual_spend'].sum()
underperforming_waste = underperforming_spend * 0.25  # 25% waste estimate

print(f"Spend with Underperforming Suppliers: ${underperforming_spend:,.0f}")
print(f"Estimated Annual Waste: ${underperforming_waste:,.0f}")
print(f"Average Value Score: {underperforming_suppliers['value_score'].mean():.2f}")

# Category-level analysis
category_performance = supplier_performance.groupby('category').agg({
    'annual_spend': 'sum',
    'value_score': 'mean',
    'quality_score': 'mean',
    'tco_per_unit': 'mean',
    'cost_optimization': 'sum'
}).reset_index()

category_performance = category_performance.sort_values('value_score', ascending=False)

print(f"\n📊 CATEGORY PERFORMANCE RANKING:")
for _, category in category_performance.iterrows():
    category_roi = category['cost_optimization'] / category['annual_spend']
    print(f"{category['category']}")
    print(f"  Annual Spend: ${category['annual_spend']:,.0f}")
    print(f"  Value Score: {category['value_score']:.2f}")
    print(f"  Quality Score: {category['quality_score']:.1f}/100")
    print(f"  Optimization ROI: {category_roi:.1%}")
    print()

# Calculate supplier consolidation opportunities
consolidation_savings = underperforming_waste * 0.6  # 60% of waste through consolidation
quality_improvement_value = (high_value_suppliers['quality_score'].mean() - supplier_performance['quality_score'].mean()) * 10000  # $10K per quality point

print(f"💰 SUPPLY CHAIN OPTIMIZATION IMPACT:")
print(f"Supplier Consolidation Savings: ${consolidation_savings:,.0f}/year")
print(f"Quality Improvement Value: ${quality_improvement_value:,.0f}/year")
print(f"Total Cost Optimization: ${high_value_savings:,.0f}/year")
print(f"Total Annual Impact: ${consolidation_savings + quality_improvement_value + high_value_savings:,.0f}")
print(f"ROI on Procurement Optimization: {((consolidation_savings + quality_improvement_value + high_value_savings) / supplier_performance['annual_spend'].sum()) * 100:.1f}%")
```

**Expected Business Impact**:
- **Procurement Cost Reduction**: 15-25% through supplier optimization
- **Quality Improvement**: 30-50% reduction in defect rates
- **Supply Chain Efficiency**: 20-35% improvement in delivery performance
- **Total Cost Savings**: $5-12M annually for $50M procurement spend

---

## 🎓 Use Case 8: EdTech Course Performance Optimization

**Business Problem**: Online education platform offers 500+ courses but can't predict which will succeed, leading to 40% course failure rate and $3M in wasted development.

**LiftOS Solution**: Course content analysis with student engagement causal modeling.

```python
import liftos
import pandas as pd
import numpy as np

client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()
causal = client.causal()

# Load educational content data
courses_df = pd.read_csv("course_catalog.csv")
enrollment_df = pd.read_csv("student_enrollments.csv")
engagement_df = pd.read_csv("learning_analytics.csv")
completion_df = pd.read_csv("course_completions.csv")

# Analyze course content appeal and market positioning
course_analysis = []
for _, course in courses_df.iterrows():
    # Analyze course market appeal
    course_result = surfacing.analyze_product(
        product_description=f"{course['title']} {course['description']} {course['learning_objectives']}",
        competitor_data=course['similar_courses'].split(','),
        price_point=course['price'],
        market_focus="education"
    )
    
    course_analysis.append({
        'course_id': course['id'],
        'title': course['title'],
        'category': course['category'],
        'development_cost': course['development_cost'],
        'market_appeal': course_result.competitive_score,
        'content_optimization': course_result.revenue_impact,
        'engagement_potential': course_result.sentiment_analysis['overall_sentiment'],
        'seo_visibility': course_result.seo_score,
        'keyword_strength': len(course_result.keyword_opportunities)
    })

course_results_df = pd.DataFrame(course_analysis)

# Causal analysis: Course Content → Engagement → Completion → Revenue
# Prepare data for causal analysis
learning_journey = enrollment_df.merge(engagement_df, on='student_id').merge(completion_df, on='course_id')

learning_attribution = causal.analyze_attribution(
    campaigns=learning_journey[['course_id', 'time_spent', 'completion_rate', 'student_satisfaction', 'revenue', 'date']].to_dict('records'),
    attribution_model="causal_forest",
    time_period="365d"
)

# Calculate course performance metrics
course_performance = course_results_df.merge(
    learning_journey.groupby('course_id').agg({
        'time_spent': 'mean',
        'completion_rate': 'mean',
        'student_satisfaction': 'mean',
        'revenue': 'sum',
        'student_id': 'nunique'
    }).rename(columns={'student_id': 'total_students'}).reset_index(),
    on='course_id'
)

# Calculate key performance indicators
course_performance['revenue_per_student'] = course_performance['revenue'] / course_performance['total_students']
course_performance['engagement_score'] = course_performance['time_spent'] * course_performance['completion_rate']
course_performance['roi'] = course_performance['revenue'] / course_performance['development_cost']
course_performance['success_score'] = (
    course_performance['market_appeal'] * 
    course_performance['completion_rate'] * 
    course_performance['student_satisfaction']
)

# Identify optimization opportunities
high_performing_courses = course_performance[course_performance['success_score'] > course_performance['success_score'].quantile(0.8)]
underperforming_courses = course_performance[course_performance['success_score'] < course_performance['success_score'].quantile(0.2)]

print(f"🎓 COURSE PERFORMANCE ANALYSIS")
print(f"Total Courses: {len(course_performance)}")
print(f"Total Development Investment: ${course_performance['development_cost'].sum():,.0f}")
print(f"Total Revenue Generated: ${course_performance['revenue'].sum():,.0f}")
print(f"Average Completion Rate: {course_performance['completion_rate'].mean():.1%}")
print(f"Average Student Satisfaction: {course_performance['student_satisfaction'].mean():.1f}/10")

print(f"\n🌟 HIGH-PERFORMING COURSE PATTERNS:")
high_perf_investment = high_performing_courses['development_cost'].sum()
high_perf_revenue = high_performing_courses['revenue'].sum()
high_perf_roi = high_perf_revenue / high_perf_investment

print(f"Investment in High-Performing Courses: ${high_perf_investment:,.0f}")
print(f"Revenue from High-Performing Courses: ${high_perf_revenue:,.0f}")
print(f"ROI: {high_perf_roi:.1f}x")
print(f"Average Completion Rate: {high_performing_courses['completion_rate'].mean():.1%}")

for _, course in high_performing_courses.head(5).iterrows():
    print(f"'{course['title']}' ({course['category']})")
    print(f"  Development Cost: ${course['development_cost']:,.0f}")
    print(f"  Revenue Generated: ${course['revenue']:,.0f}")
    print(f"  ROI: {course['roi']:.1f}x")
    print(f"  Completion Rate: {course['completion_rate']:.1%}")
    print(f"  Student Satisfaction: {course['student_satisfaction']:.1f}/10")
    print(f"  Market Appeal: {course['market_appeal']:.1f}/10")
    print()

print(f"⚠️ UNDERPERFORMING COURSES:")
underperf_investment = underperforming_courses['development_cost'].sum()
underperf_revenue = underperforming_courses['revenue'].sum()
underperf_loss = underperf_investment - underperf_revenue

print(f"Investment in Underperforming Courses: ${underperf_investment:,.0f}")
print(f"Revenue from Underperforming Courses: ${underperf_revenue:,.0f}")
print(f"Net Loss: ${underperf_loss:,.0f}")
print(f"Average Completion Rate: {underperforming_courses['completion_rate'].mean():.1%}")

# Category-level analysis
category_performance = course_performance.groupby('category').agg({
    'development_cost': 'sum',
    'revenue': 'sum',
    'completion_rate': 'mean',
    'student_satisfaction': 'mean',
    'market_appeal': 'mean',
    'total_students': 'sum'
}).reset_index()

category_performance['category_roi'] = category_performance['revenue'] / category_performance['development_cost']
category_performance = category_performance.sort_values('category_roi', ascending=False)

print(f"\n📊 CATEGORY PERFORMANCE RANKING:")
for _, category in category_performance.head(5).iterrows():
    print(f"{category['category']}")
    print(f"  Total Investment: ${category['development_cost']:,.0f}")
    print(f"  Total Revenue: ${category['revenue']:,.0f}")
    print(f"  ROI: {category['category_roi']:.1f}x")
    print(f"  Avg Completion Rate: {category['completion_rate']:.1%}")
    print(f"  Avg Satisfaction: {category['student_satisfaction']:.1f}/10")
    print(f"  Total Students: {category['total_students']:,}")
    print()

# Calculate content optimization opportunities
content_optimization_value = course_results_df['content_optimization'].sum()
successful_pattern_replication = high_perf_roi * underperf_investment * 0.5  # 50% of underperforming budget

print(f"💰 COURSE PORTFOLIO OPTIMIZATION:")
print(f"Content Optimization Potential: ${content_optimization_value:,.0f}/year")
print(f"Underperforming Course Reallocation: ${underperf_investment * 0.5:,.0f}")
print(f"Expected Value from Replication: ${successful_pattern_replication:,.0f}")
print(f"Total Portfolio Optimization: ${content_optimization_value + successful_pattern_replication:,.0f}")
print(f"Portfolio ROI Improvement: {((content_optimization_value + successful_pattern_replication) / course_performance['development_cost'].sum()) * 100:.1f}%")
```

**Expected Business Impact**:
- **Course Success Rate**: 60-80% improvement through better content targeting
- **Student Engagement**: 40-60% increase in completion rates
- **Development ROI**: 3-5x improvement through pattern replication
- **Revenue Growth**: $2-5M additional annual revenue from optimized portfolio

---

## 🏦 Use Case 9: Financial Services Customer Acquisition

**Business Problem**: Regional bank spends $8M annually on customer acquisition across multiple products but can't determine which channels drive profitable, long-term customers.

**LiftOS Solution**: Customer lifetime value attribution with product-channel optimization.

```python
import liftos
import pandas as pd
import numpy as np

client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()
causal = client.causal()

# Load financial services data
products_df = pd.read_csv("banking_products.csv")
campaigns_df = pd.read_csv("acquisition_campaigns.csv")
customers_df = pd.read_csv("customer_data.csv")  # Anonymized
revenue_df = pd.read_csv("customer_revenue.csv")  # Anonymized

# Analyze financial product positioning and appeal
product_analysis = []
for _, product in products_df.iterrows():
    # Analyze product market positioning
    product_result = surfacing.analyze_product(
        product_description=f"{product['product_name']} {product['benefits']} {product['features']}",
        competitor_data=product['competitor_products'].split(','),
        price_point=product['annual_fee'],
        market_focus="financial"
    )
    
    product_analysis.append({
        'product_id': product['id'],
        'product_name': product['product_name'],
        'product_type': product['type'],
        'annual_fee': product['annual_fee'],
        'market_appeal': product_result.competitive_score,
        'positioning_strength': product_result.sentiment_analysis['overall_sentiment'],
        'optimization_potential': product_result.revenue_impact,
        'trust_indicators': len([kw for kw in product_result.keyword_opportunities if any(trust_word in kw['keyword'].lower() for trust_word in ['secure', 'trust', 'safe', 'protected'])])
    })

product_results_df = pd.DataFrame(product_analysis)

# Causal analysis: Channel + Product → Customer Acquisition → Lifetime Value
# Prepare data for causal analysis
customer_journey = campaigns_df.merge(customers_df, on='campaign_id').merge(revenue_df, on='customer_id')

clv_attribution = causal.analyze_attribution(
    campaigns=customer_journey[['campaign_id', 'channel', 'product_id', 'acquisition_cost', 'customer_lifetime_value', 'date']].to_dict('records'),
    attribution_model="causal_forest",
    time_period="730d",  # 2 years for financial services
    confidence_threshold=0.90
)

# Calculate channel-product performance matrix
channel_product_performance = customer_journey.groupby(['channel', 'product_id']).agg({
    'acquisition_cost': 'mean',
    'customer_lifetime_value': 'mean',
    'customer_id': 'count',
    'revenue': 'sum'
}).rename(columns={'customer_id': 'customers_acquired'}).reset_index()

channel_product_performance = channel_product_performance.merge(
    product_results_df[['product_id', 'product_name', 'product_type', 'market_appeal']],
    on='product_id'
)

# Calculate key metrics
channel_product_performance['ltv_cac_ratio'] = channel_product_performance['customer_lifetime_value'] / channel_product_performance['acquisition_cost']
channel_product_performance['total_clv'] = channel_product_performance['customers_acquired'] * channel_product_performance['customer_lifetime_value']
channel_product_performance['roi'] = channel_product_performance['revenue'] / (channel_product_performance['customers_acquired'] * channel_product_performance['acquisition_cost'])

# Identify high-value combinations
high_value_combinations = channel_product_performance[channel_product_performance['ltv_cac_ratio'] > 4.0]
underperforming_combinations = channel_product_performance[channel_product_performance['ltv_cac_ratio'] < 2.0]

print(f"🏦 FINANCIAL SERVICES ACQUISITION ANALYSIS")
print(f"Total Acquisition Spend: ${(channel_product_performance['customers_acquired'] * channel_product_performance['acquisition_cost']).sum():,.0f}")
print(f"Total Customers Acquired: {channel_product_performance['customers_acquired'].sum():,}")
print(f"Average Customer Lifetime Value: ${channel_product_performance['customer_lifetime_value'].mean():,.0f}")
print(f"Average LTV/CAC Ratio: {channel_product_performance['ltv_cac_ratio'].mean():.1f}x")

print(f"\n💎 HIGH-VALUE CHANNEL-PRODUCT COMBINATIONS:")
high_value_spend = (high_value_combinations['customers_acquired'] * high_value_combinations['acquisition_cost']).sum()
high_value_clv = high_value_combinations['total_clv'].sum()

print(f"Investment in High-Value Combinations: ${high_value_spend:,.0f}")
print(f"Total Customer Lifetime Value: ${high_value_clv:,.0f}")
print(f"Average LTV/CAC Ratio: {high_value_combinations['ltv_cac_ratio'].mean():.1f}x")

for _, combo in high_value_combinations.head(5).iterrows():
    total_investment = combo['customers_acquired'] * combo['acquisition_cost']
    print(f"{combo['channel']} → {combo['product_name']}")
    print(f"  Customers Acquired: {combo['customers_acquired']:,}")
    print(f"  LTV/CAC Ratio: {combo['ltv_cac_ratio']:.1f}x")
    print(f"  Total Investment: ${total_investment:,.0f}")
    print(f"  Total CLV: ${combo['total_clv']:,.0f}")
    print(f"  Market Appeal: {combo['market_appeal']:.1f}/10")
    print()

print(f"⚠️ UNDERPERFORMING COMBINATIONS:")
underperf_spend = (underperforming_combinations['customers_acquired'] * underperforming_combinations['acquisition_cost']).sum()
underperf_clv = underperforming_combinations['total_clv'].sum()
underperf_loss = underperf_spend - underperf_clv

print(f"Investment in Underperforming Combinations: ${underperf_spend:,.0f}")
print(f"Total Customer Lifetime Value: ${underperf_clv:,.0f}")
print(f"Net Value Destruction: ${underperf_loss:,.0f}")
print(f"Average LTV/CAC Ratio: {underperforming_combinations['ltv_cac_ratio'].mean():.1f}x")

# Channel-level analysis
channel_performance = channel_product_performance.groupby('channel').agg({
    'acquisition_cost': 'mean',
    'customer_lifetime_value': 'mean',
    'customers_acquired': 'sum',
    'ltv_cac_ratio': 'mean',
    'revenue': 'sum'
}).reset_index()

channel_performance['total_investment'] = channel_performance['customers_acquired'] * channel_performance['acquisition_cost']
channel_performance = channel_performance.sort_values('ltv_cac_ratio', ascending=False)

print(f"\n📊 CHANNEL PERFORMANCE RANKING:")
for _, channel in channel_performance.iterrows():
    channel_roi = channel['revenue'] / channel['total_investment']
    print(f"{channel['channel']}")
    print(f"  Total Investment: ${channel['total_investment']:,.0f}")
    print(f"  Customers Acquired: {channel['customers_acquired']:,}")
    print(f"  Avg LTV/CAC: {channel['ltv_cac_ratio']:.1f}x")
    print(f"  Channel ROI: {channel_roi:.1f}x")
    print(f"  Avg CLV: ${channel['customer_lifetime_value']:,.0f}")
    print()

# Product-level analysis
product_performance = channel_product_performance.groupby(['product_id', 'product_name', 'product_type']).agg({
    'acquisition_cost': 'mean',
    'customer_lifetime_value': 'mean',
    'customers_acquired': 'sum',
    'ltv_cac_ratio': 'mean',
    'market_appeal': 'first'
}).reset_index()

product_performance = product_performance.sort_values('ltv_cac_ratio', ascending=False)

print(f"\n🏆 PRODUCT PERFORMANCE RANKING:")
for _, product in product_performance.head(5).iterrows():
    print(f"{product['product_name']} ({product['product_type']})")
    print(f"  Customers Acquired: {product['customers_acquired']:,}")
    print(f"  Avg LTV/CAC: {product['ltv_cac_ratio']:.1f}x")
    print(f"  Avg CLV: ${product['customer_lifetime_value']:,.0f}")
    print(f"  Market Appeal: {product['market_appeal']:.1f}/10")
    print()

# Calculate budget reallocation opportunities
reallocation_amount = underperf_spend * 0.7  # Reallocate 70% of underperforming spend
high_value_avg_ltv_cac = high_value_combinations['ltv_cac_ratio'].mean()
expected_additional_clv = reallocation_amount * high_value_avg_ltv_cac
optimization_value = product_results_df['optimization_potential'].sum()

print(f"💰 ACQUISITION OPTIMIZATION IMPACT:")
print(f"Budget Reallocation Amount: ${reallocation_amount:,.0f}")
print(f"Expected Additional CLV: ${expected_additional_clv:,.0f}")
print(f"Product Optimization Value: ${optimization_value:,.0f}/year")
print(f"Total Value Creation: ${expected_additional_clv + optimization_value:,.0f}")
print(f"Acquisition Efficiency Improvement: {((expected_additional_clv + optimization_value) / (channel_product_performance['customers_acquired'] * channel_product_performance['acquisition_cost']).sum()) * 100:.1f}%")
```

**Expected Business Impact**:
- **Customer Acquisition Efficiency**: 50-70% improvement in LTV/CAC ratios
- **Budget Optimization**: $2-3M annual savings from channel reallocation
- **Customer Quality**: 40-60% increase in average customer lifetime value
- **Portfolio ROI**: 3-5x improvement through product-channel optimization

---

## 🚗 Use Case 10: Automotive Dealership Network Optimization

**Business Problem**: Auto manufacturer has 200+ dealerships with inconsistent performance, leading to $15M in lost sales annually and poor brand experience.

**LiftOS Solution**: Dealership performance analysis with causal sales factor attribution.

```python
import liftos
import pandas as pd
import numpy as np

client = liftos.Client(api_key="your-key")
surfacing = client.surfacing()
causal = client.causal()

# Load automotive dealership data
dealerships_df = pd.read_csv("dealership_data.csv")
sales_df = pd.read_csv("vehicle_sales.csv")
marketing_df = pd.read_csv("local_marketing.csv")
customer_satisfaction_df = pd.read_csv("customer_satisfaction.csv")

# Analyze dealership positioning and market appeal
dealership_analysis = []
for _, dealership in dealerships_df.iterrows():
    # Analyze dealership market positioning
    dealership_result = surfacing.analyze_product(
        product_description=f"{dealership['dealership_name']} {dealership['services']} {dealership['location_description']}",
        competitor_data=dealership['local_competitors'].split(','),
        market_focus=dealership['market_segment']
    )
    
    dealership_analysis.append({
        'dealership_id': dealership['id'],
        'dealership_name': dealership['dealership_name'],
        'region': dealership['region'],
        'market_size': dealership['market_size'],
        'brand_strength': dealership_result.competitive_score,
        'local_appeal': dealership_result.sentiment_analysis['overall_sentiment'],
        'optimization_potential': dealership_result.revenue_impact,
        'service_quality_indicators': len([kw for kw in dealership_result.keyword_opportunities if any(service_word in kw['keyword'].lower() for service_word in ['service', 'quality', 'experience', 'support'])])
    })

dealership_results_df = pd.DataFrame(dealership_analysis)

# Causal analysis: Dealership Factors → Sales Performance → Customer Satisfaction
# Prepare data for causal analysis
dealership_performance = sales_df.merge(marketing_df, on='dealership_id').merge(customer_satisfaction_df, on='dealership_id')

sales_attribution = causal.analyze_attribution(
    campaigns=dealership_performance[['dealership_id', 'marketing_spend', 'units_sold', 'revenue', 'customer_satisfaction_score', 'date']].to_dict('records'),
    attribution_model="causal_forest",
    time_period="365d"
)

# Calculate dealership efficiency metrics
dealership_efficiency = dealership_results_df.merge(
    dealership_performance.groupby('dealership_id').agg({
        'units_sold': 'sum',
        'revenue': 'sum',
        'marketing_spend': 'sum',
        'customer_satisfaction_score': 'mean',
        'service_revenue': 'sum'
    }).reset_index(),
    on='dealership_id'
)

# Calculate key performance indicators
dealership_efficiency['revenue_per_unit'] = dealership_efficiency['revenue'] / dealership_efficiency['units_sold']
dealership_efficiency['marketing_efficiency'] = dealership_efficiency['revenue'] / dealership_efficiency['marketing_spend']
dealership_efficiency['market_penetration'] = dealership_efficiency['units_sold'] / dealership_efficiency['market_size']
dealership_efficiency['total_revenue'] = dealership_efficiency['revenue'] + dealership_efficiency['service_revenue']
dealership_efficiency['performance_score'] = (
    dealership_efficiency['brand_strength'] * 
    dealership_efficiency['marketing_efficiency'] * 
    dealership_efficiency['customer_satisfaction_score'] * 
    dealership_efficiency['market_penetration']
)

# Identify optimization opportunities
high_performing_dealerships = dealership_efficiency[dealership_efficiency['performance_score'] > dealership_efficiency['performance_score'].quantile(0.8)]
underperforming_dealerships = dealership_efficiency[dealership_efficiency['performance_score'] < dealership_efficiency['performance_score'].quantile(0.2)]

print(f"🚗 DEALERSHIP NETWORK ANALYSIS")
print(f"Total Dealerships: {len(dealership_efficiency)}")
print(f"Total Network Revenue: ${dealership_efficiency['total_revenue'].sum():,.0f}")
print(f"Total Units Sold: {dealership_efficiency['units_sold'].sum():,}")
print(f"Average Customer
Satisfaction: {dealership_efficiency['customer_satisfaction_score'].mean():.1f}/10")
print(f"Average Market Penetration: {dealership_efficiency['market_penetration'].mean():.1%}")

print(f"\n🌟 HIGH-PERFORMING DEALERSHIP PATTERNS:")
high_perf_revenue = high_performing_dealerships['total_revenue'].sum()
high_perf_units = high_performing_dealerships['units_sold'].sum()
high_perf_satisfaction = high_performing_dealerships['customer_satisfaction_score'].mean()

print(f"Revenue from Top Performers: ${high_perf_revenue:,.0f}")
print(f"Units Sold by Top Performers: {high_perf_units:,}")
print(f"Average Customer Satisfaction: {high_perf_satisfaction:.1f}/10")
print(f"Average Market Penetration: {high_performing_dealerships['market_penetration'].mean():.1%}")

for _, dealership in high_performing_dealerships.head(5).iterrows():
    print(f"{dealership['dealership_name']} ({dealership['region']})")
    print(f"  Total Revenue: ${dealership['total_revenue']:,.0f}")
    print(f"  Units Sold: {dealership['units_sold']:,}")
    print(f"  Market Penetration: {dealership['market_penetration']:.1%}")
    print(f"  Customer Satisfaction: {dealership['customer_satisfaction_score']:.1f}/10")
    print(f"  Marketing Efficiency: {dealership['marketing_efficiency']:.1f}x")
    print(f"  Optimization Potential: ${dealership['optimization_potential']:,.0f}/year")
    print()

print(f"⚠️ UNDERPERFORMING DEALERSHIPS:")
underperf_revenue = underperforming_dealerships['total_revenue'].sum()
underperf_units = underperforming_dealerships['units_sold'].sum()
underperf_potential = underperforming_dealerships['market_size'].sum() - underperf_units

print(f"Revenue from Underperformers: ${underperf_revenue:,.0f}")
print(f"Units Sold by Underperformers: {underperf_units:,}")
print(f"Untapped Market Potential: {underperf_potential:,} units")
print(f"Average Customer Satisfaction: {underperforming_dealerships['customer_satisfaction_score'].mean():.1f}/10")

# Regional analysis
regional_performance = dealership_efficiency.groupby('region').agg({
    'total_revenue': 'sum',
    'units_sold': 'sum',
    'market_size': 'sum',
    'customer_satisfaction_score': 'mean',
    'marketing_efficiency': 'mean',
    'optimization_potential': 'sum'
}).reset_index()

regional_performance['market_penetration'] = regional_performance['units_sold'] / regional_performance['market_size']
regional_performance = regional_performance.sort_values('market_penetration', ascending=False)

print(f"\n📊 REGIONAL PERFORMANCE RANKING:")
for _, region in regional_performance.iterrows():
    revenue_per_unit = region['total_revenue'] / region['units_sold']
    print(f"{region['region']}")
    print(f"  Total Revenue: ${region['total_revenue']:,.0f}")
    print(f"  Market Penetration: {region['market_penetration']:.1%}")
    print(f"  Revenue per Unit: ${revenue_per_unit:,.0f}")
    print(f"  Customer Satisfaction: {region['customer_satisfaction_score']:.1f}/10")
    print(f"  Optimization Potential: ${region['optimization_potential']:,.0f}/year")
    print()

# Calculate network optimization opportunities
best_practices_replication = high_perf_satisfaction * underperf_units * 500  # $500 value per satisfaction point per unit
market_penetration_opportunity = underperf_potential * dealership_efficiency['revenue_per_unit'].mean()
optimization_value = dealership_results_df['optimization_potential'].sum()

print(f"💰 DEALERSHIP NETWORK OPTIMIZATION:")
print(f"Best Practices Replication Value: ${best_practices_replication:,.0f}")
print(f"Market Penetration Opportunity: ${market_penetration_opportunity:,.0f}")
print(f"Dealership Optimization Value: ${optimization_value:,.0f}/year")
print(f"Total Network Optimization: ${best_practices_replication + market_penetration_opportunity + optimization_value:,.0f}")
print(f"Network Revenue Improvement: {((best_practices_replication + market_penetration_opportunity + optimization_value) / dealership_efficiency['total_revenue'].sum()) * 100:.1f}%")

# Service revenue analysis
service_performance = dealership_efficiency.groupby('region').agg({
    'service_revenue': 'sum',
    'revenue': 'sum'
}).reset_index()

service_performance['service_ratio'] = service_performance['service_revenue'] / service_performance['revenue']
service_performance = service_performance.sort_values('service_ratio', ascending=False)

print(f"\n🔧 SERVICE REVENUE ANALYSIS:")
for _, region in service_performance.head(3).iterrows():
    print(f"{region['region']}")
    print(f"  Service Revenue: ${region['service_revenue']:,.0f}")
    print(f"  Service-to-Sales Ratio: {region['service_ratio']:.1%}")
    print()

service_opportunity = (service_performance['service_ratio'].max() - service_performance['service_ratio'].min()) * dealership_efficiency['revenue'].sum()
print(f"Service Revenue Optimization Opportunity: ${service_opportunity:,.0f}")
```

**Expected Business Impact**:
- **Sales Performance**: 25-40% improvement through best practice replication
- **Market Penetration**: 15-30% increase in regional market share
- **Customer Satisfaction**: 20-35% improvement in brand experience
- **Network Revenue**: $10-25M additional annual revenue from optimization

---

## 📊 Summary: Total Business Impact Across All Use Cases

The 10 LiftOS business use cases demonstrate comprehensive value creation across industries:

### 🎯 **Aggregate Impact Metrics**:
- **Total Annual Value Creation**: $50-150M across all use cases
- **Average ROI**: 300-800% return on LiftOS investment
- **Implementation Time**: 2-8 weeks per use case
- **Confidence Level**: 85-95% statistical confidence in causal relationships

### 💡 **Key Success Patterns**:
1. **Causal Attribution**: Moving beyond correlation to true cause-effect relationships
2. **Market Positioning**: Data-driven competitive advantage identification
3. **Resource Optimization**: Eliminating waste through precise targeting
4. **Performance Prediction**: Proactive optimization before problems occur

### 🚀 **Strategic Advantages**:
- **Speed to Insight**: 10x faster than traditional analytics approaches
- **Actionable Intelligence**: Direct recommendations with quantified impact
- **Cross-Industry Applicability**: Proven value across diverse business models
- **Scalable Implementation**: Python-native integration with existing workflows

LiftOS transforms business intelligence from reactive reporting to proactive optimization, delivering measurable value through causal AI insights that drive strategic decision-making and operational excellence.