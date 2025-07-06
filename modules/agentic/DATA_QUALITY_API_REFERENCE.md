# Data Quality API Reference

This document provides comprehensive API reference for the LiftOS Agentic microservice data quality evaluation endpoints.

## Overview

The Data Quality API provides comprehensive evaluation of data quality across 8 dimensions to ensure highly accurate outcomes for agent testing and evaluation. The API implements a weighted scoring system with 5-tier quality classification.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, no authentication is required for data quality endpoints.

## Quality Dimensions

The API evaluates data across 8 key dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **Completeness** | 20% | Measures the extent to which data is present and not missing |
| **Accuracy** | 25% | Measures how well data represents real-world values |
| **Consistency** | 15% | Measures uniformity of data format and representation |
| **Validity** | 15% | Measures conformance to defined formats and business rules |
| **Uniqueness** | 10% | Measures absence of duplicate records |
| **Timeliness** | 5% | Measures how current and up-to-date the data is |
| **Relevance** | 5% | Measures applicability of data to the intended use case |
| **Integrity** | 5% | Measures referential and structural data integrity |

## Quality Levels

| Level | Score Range | Description |
|-------|-------------|-------------|
| **Excellent** | 95-100% | Production-ready data with minimal issues |
| **Good** | 85-94% | High-quality data with minor issues |
| **Acceptable** | 70-84% | Moderate quality, use with caution |
| **Poor** | 50-69% | Significant issues, remediation required |
| **Critical** | <50% | Severe problems, immediate action required |

## API Endpoints

### 1. Evaluate Data Quality

Performs comprehensive data quality assessment across all 8 dimensions.

**Endpoint:** `POST /data-quality/evaluate`

**Request Body:**
```json
{
  "data": {
    "customer_id": [1, 2, 3, 4, 5],
    "name": ["Alice Johnson", "Bob Smith", "Carol Davis", "David Wilson", "Eva Brown"],
    "email": ["alice@email.com", "bob@email.com", "carol@email.com", "david@email.com", "eva@email.com"],
    "age": [25, 30, 35, 28, 32],
    "purchase_amount": [100.50, 250.75, 175.25, 320.00, 89.99]
  },
  "dataset_id": "customer_data_001",
  "test_type": "general",
  "include_profiling": true
}
```

**Request Parameters:**
- `data` (required): Data to evaluate (dict, list of dicts, or DataFrame-compatible)
- `dataset_id` (required): Unique identifier for the dataset
- `test_type` (optional): Type of test, defaults to "general"
- `include_profiling` (optional): Whether to include data profiling, defaults to true

**Response:**
```json
{
  "dataset_id": "customer_data_001",
  "evaluation_timestamp": "2024-01-20T10:30:00Z",
  "overall_score": 0.923,
  "overall_level": "EXCELLENT",
  "dimension_scores": {
    "COMPLETENESS": {
      "score": 1.0,
      "level": "EXCELLENT",
      "description": "All required fields are present",
      "issues_found": [],
      "recommendations": [],
      "metadata": {
        "missing_values": 0,
        "total_values": 25,
        "completeness_rate": 1.0
      }
    },
    "ACCURACY": {
      "score": 0.95,
      "level": "EXCELLENT",
      "description": "Data values appear accurate with minor anomalies",
      "issues_found": ["Some outlier values detected"],
      "recommendations": ["Review outlier values for accuracy"],
      "metadata": {
        "outliers_detected": 1,
        "accuracy_checks_passed": 24
      }
    }
  },
  "critical_issues": [],
  "recommendations": [
    "Consider implementing automated data validation",
    "Monitor data quality trends over time"
  ],
  "data_profile": {
    "row_count": 5,
    "column_count": 5,
    "data_types": {
      "customer_id": "integer",
      "name": "string",
      "email": "string",
      "age": "integer",
      "purchase_amount": "float"
    },
    "missing_value_summary": {
      "total_missing": 0,
      "columns_with_missing": []
    }
  }
}
```

### 2. Validate Data for Agent Testing

Validates if data meets quality standards required for reliable agent testing.

**Endpoint:** `POST /data-quality/validate-for-testing`

**Request Body:**
```json
{
  "data": {
    "campaign_id": [1, 2, 3],
    "target_audience": ["young_adults", "professionals", "seniors"],
    "budget": [1000, 2500, 1500],
    "conversion_rate": [0.05, 0.08, 0.03]
  },
  "test_type": "marketing_campaign"
}
```

**Request Parameters:**
- `data` (required): Data to validate
- `test_type` (required): Type of agent test (e.g., "marketing_campaign", "customer_segmentation")

**Response:**
```json
{
  "is_valid": true,
  "validation_result": "PASS",
  "overall_score": 0.887,
  "overall_level": "GOOD",
  "critical_issues": [],
  "recommendations": [
    "Data quality is sufficient for agent testing",
    "Monitor conversion rate consistency"
  ],
  "quality_summary": {
    "COMPLETENESS": {
      "score": 1.0,
      "level": "EXCELLENT"
    },
    "ACCURACY": {
      "score": 0.85,
      "level": "GOOD"
    },
    "CONSISTENCY": {
      "score": 0.90,
      "level": "EXCELLENT"
    }
  }
}
```

### 3. Get Quality Dimensions Information

Returns detailed information about quality dimensions and their weights.

**Endpoint:** `GET /data-quality/dimensions`

**Response:**
```json
{
  "dimensions": {
    "completeness": {
      "weight": 0.20,
      "description": "Measures the extent to which data is present and not missing",
      "importance": "Critical for avoiding biased evaluations"
    },
    "accuracy": {
      "weight": 0.25,
      "description": "Measures how well data represents real-world values",
      "importance": "Highest priority - directly impacts agent testing results"
    }
  },
  "quality_levels": {
    "excellent": {
      "range": "95-100%",
      "description": "Production-ready data"
    },
    "good": {
      "range": "85-94%",
      "description": "High-quality with minor issues"
    }
  },
  "validation_thresholds": {
    "agent_testing_minimum": 0.85,
    "critical_issues_allowed": 0,
    "key_dimensions_minimum": 0.80
  }
}
```

### 4. Health Check

Performs health check on the data quality engine.

**Endpoint:** `GET /data-quality/health`

**Response:**
```json
{
  "status": "healthy",
  "engine_initialized": true,
  "test_evaluation_successful": true,
  "sample_score": 0.95,
  "timestamp": "2024-01-20T10:30:00Z"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid data format provided"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "dataset_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Failed to evaluate data quality: Internal processing error"
}
```

### 503 Service Unavailable
```json
{
  "detail": "Data quality engine not initialized"
}
```

## Usage Examples

### Python Example

```python
import requests
import json

# Evaluate data quality
data = {
    "customer_id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Carol", "David", "Eva"],
    "email": ["alice@email.com", "bob@email.com", "carol@email.com", "david@email.com", "eva@email.com"]
}

response = requests.post(
    "http://localhost:8000/data-quality/evaluate",
    json={
        "data": data,
        "dataset_id": "customer_data_001",
        "include_profiling": True
    }
)

result = response.json()
print(f"Overall Score: {result['overall_score']:.3f}")
print(f"Quality Level: {result['overall_level']}")

# Validate for agent testing
validation_response = requests.post(
    "http://localhost:8000/data-quality/validate-for-testing",
    json={
        "data": data,
        "test_type": "customer_segmentation"
    }
)

validation_result = validation_response.json()
if validation_result["is_valid"]:
    print("Data is suitable for agent testing")
else:
    print("Data quality issues detected:")
    for issue in validation_result["critical_issues"]:
        print(f"- {issue}")
```

### cURL Example

```bash
# Evaluate data quality
curl -X POST "http://localhost:8000/data-quality/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "id": [1, 2, 3],
      "value": [10, 20, 30]
    },
    "dataset_id": "test_data_001"
  }'

# Get quality dimensions
curl -X GET "http://localhost:8000/data-quality/dimensions"

# Health check
curl -X GET "http://localhost:8000/data-quality/health"
```

### JavaScript Example

```javascript
// Evaluate data quality
const evaluateDataQuality = async (data, datasetId) => {
  const response = await fetch('http://localhost:8000/data-quality/evaluate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      data: data,
      dataset_id: datasetId,
      include_profiling: true
    })
  });
  
  const result = await response.json();
  return result;
};

// Usage
const sampleData = {
  customer_id: [1, 2, 3],
  name: ['Alice', 'Bob', 'Carol'],
  email: ['alice@email.com', 'bob@email.com', 'carol@email.com']
};

evaluateDataQuality(sampleData, 'customer_data_001')
  .then(result => {
    console.log(`Quality Score: ${result.overall_score}`);
    console.log(`Quality Level: ${result.overall_level}`);
  })
  .catch(error => {
    console.error('Error evaluating data quality:', error);
  });
```

## Best Practices

### 1. Data Preparation
- Ensure data is in a consistent format (dict with lists or pandas DataFrame)
- Use meaningful dataset IDs for tracking and debugging
- Include representative sample sizes for accurate assessment

### 2. Quality Thresholds
- Use minimum score of 0.85 for agent testing validation
- Address critical issues before proceeding with agent evaluation
- Monitor quality trends over time

### 3. Performance Optimization
- For large datasets (>10,000 rows), consider sampling for initial assessment
- Use `include_profiling: false` for faster evaluation when profiling not needed
- Implement caching for repeated evaluations of the same dataset

### 4. Error Handling
- Always check response status codes
- Implement retry logic for transient failures
- Log quality assessment results for audit trails

### 5. Integration Patterns
- Integrate quality checks into data pipelines
- Set up automated alerts for quality degradation
- Use quality scores to prioritize data remediation efforts

## Quality Assessment Workflow

1. **Initial Assessment**: Use `/data-quality/evaluate` to get comprehensive quality report
2. **Issue Analysis**: Review dimension scores and critical issues
3. **Data Remediation**: Address identified quality issues
4. **Validation**: Use `/data-quality/validate-for-testing` to confirm suitability
5. **Monitoring**: Implement ongoing quality monitoring

## Support and Troubleshooting

### Common Issues

**Low Completeness Scores**
- Check for missing values in critical fields
- Verify data extraction processes
- Implement data validation at source

**Poor Accuracy Scores**
- Review outlier detection results
- Validate data transformation logic
- Check for data entry errors

**Consistency Issues**
- Standardize data formats across sources
- Implement data normalization procedures
- Review data integration processes

### Getting Help

For additional support with the Data Quality API:
1. Check the health endpoint for system status
2. Review error messages and response codes
3. Consult the comprehensive test suite for usage examples
4. Monitor application logs for detailed error information

## Changelog

### Version 1.0.0 (Current)
- Initial release with 8-dimensional quality assessment
- Agent testing validation capabilities
- Comprehensive API endpoints
- Performance optimization for large datasets
- Extensive test coverage and documentation