"""
Data Source Configuration and Validation Service
Validates data quality, consistency, and reliability across platforms
"""

import asyncio
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
import logging

from shared.models.marketing import DataSource
from shared.connectors.connector_factory import get_connector_manager
from shared.services.platform_connection_service import get_platform_connection_service
from shared.utils.logging import setup_logging

logger = setup_logging("data_source_validator")

class DataQualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class ValidationStatus(Enum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ValidationRule:
    """Represents a data validation rule"""
    rule_id: str
    name: str
    description: str
    severity: str  # critical, high, medium, low
    platform_specific: bool = False
    applicable_platforms: List[str] = None

@dataclass
class ValidationResult:
    """Result of a validation check"""
    rule_id: str
    status: ValidationStatus
    score: float  # 0-100
    message: str
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    connection_id: str
    platform: str
    overall_score: float
    quality_level: DataQualityLevel
    validation_results: List[ValidationResult]
    data_freshness: Dict[str, Any]
    completeness_metrics: Dict[str, Any]
    consistency_metrics: Dict[str, Any]
    reliability_metrics: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime

class DataSourceValidator:
    """Service for validating data source quality and configuration"""
    
    def __init__(self):
        self.connector_manager = get_connector_manager()
        self.platform_service = None
        self.validation_rules = self._initialize_validation_rules()
    
    async def initialize(self):
        """Initialize the validator service"""
        self.platform_service = get_platform_connection_service()
    
    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules for data quality checks"""
        return [
            # Data Freshness Rules
            ValidationRule(
                rule_id="data_freshness_daily",
                name="Daily Data Freshness",
                description="Data should be updated within the last 24 hours",
                severity="high"
            ),
            ValidationRule(
                rule_id="data_freshness_weekly",
                name="Weekly Data Availability",
                description="Data should be available for the past 7 days",
                severity="medium"
            ),
            
            # Data Completeness Rules
            ValidationRule(
                rule_id="required_fields_present",
                name="Required Fields Present",
                description="All required fields should be present in the data",
                severity="critical"
            ),
            ValidationRule(
                rule_id="data_volume_consistency",
                name="Data Volume Consistency",
                description="Data volume should be consistent with historical patterns",
                severity="medium"
            ),
            
            # Data Accuracy Rules
            ValidationRule(
                rule_id="metric_relationships",
                name="Metric Relationships",
                description="Related metrics should have logical relationships (e.g., CTR = Clicks/Impressions)",
                severity="high"
            ),
            ValidationRule(
                rule_id="value_ranges",
                name="Value Range Validation",
                description="Metric values should be within expected ranges",
                severity="medium"
            ),
            
            # Platform-Specific Rules
            ValidationRule(
                rule_id="meta_business_account_structure",
                name="Meta Business Account Structure",
                description="Account hierarchy should be properly structured",
                severity="medium",
                platform_specific=True,
                applicable_platforms=["meta_business"]
            ),
            ValidationRule(
                rule_id="google_ads_quality_score",
                name="Google Ads Quality Score",
                description="Quality scores should be within acceptable ranges",
                severity="low",
                platform_specific=True,
                applicable_platforms=["google_ads"]
            ),
            
            # Data Consistency Rules
            ValidationRule(
                rule_id="cross_platform_consistency",
                name="Cross-Platform Consistency",
                description="Similar metrics across platforms should be consistent",
                severity="medium"
            ),
            ValidationRule(
                rule_id="temporal_consistency",
                name="Temporal Consistency",
                description="Data should be consistent across time periods",
                severity="medium"
            )
        ]
    
    async def validate_connection(self, user_id: str, org_id: str, 
                                connection_id: str) -> DataQualityReport:
        """Validate a single platform connection"""
        try:
            # Get connection details
            connections = await self.platform_service.get_user_connections(user_id, org_id)
            connection = next((c for c in connections if c.id == connection_id), None)
            
            if not connection:
                raise ValueError(f"Connection {connection_id} not found")
            
            # Get connector
            connector = await self.connector_manager.get_connector(
                connection_id,
                DataSource(connection.platform),
                connection.credentials
            )
            
            # Extract sample data for validation
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=7)
            
            sample_data = await connector.extract_data(start_date, end_date)
            
            # Run validation checks
            validation_results = await self._run_validation_checks(
                connection, sample_data
            )
            
            # Calculate overall score and quality level
            overall_score = self._calculate_overall_score(validation_results)
            quality_level = self._determine_quality_level(overall_score)
            
            # Generate metrics
            data_freshness = self._analyze_data_freshness(sample_data)
            completeness_metrics = self._analyze_completeness(sample_data)
            consistency_metrics = self._analyze_consistency(sample_data)
            reliability_metrics = self._analyze_reliability(connection, sample_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(validation_results, connection)
            
            return DataQualityReport(
                connection_id=connection_id,
                platform=connection.platform,
                overall_score=overall_score,
                quality_level=quality_level,
                validation_results=validation_results,
                data_freshness=data_freshness,
                completeness_metrics=completeness_metrics,
                consistency_metrics=consistency_metrics,
                reliability_metrics=reliability_metrics,
                recommendations=recommendations,
                generated_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Failed to validate connection {connection_id}: {str(e)}")
            raise
    
    async def validate_all_connections(self, user_id: str, org_id: str) -> List[DataQualityReport]:
        """Validate all user connections"""
        try:
            connections = await self.platform_service.get_user_connections(user_id, org_id)
            
            validation_tasks = [
                self.validate_connection(user_id, org_id, conn.id)
                for conn in connections if conn.status == "active"
            ]
            
            reports = await asyncio.gather(*validation_tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            valid_reports = []
            for i, report in enumerate(reports):
                if isinstance(report, Exception):
                    logger.error(f"Validation failed for connection {connections[i].id}: {str(report)}")
                else:
                    valid_reports.append(report)
            
            return valid_reports
            
        except Exception as e:
            logger.error(f"Failed to validate all connections: {str(e)}")
            raise
    
    async def _run_validation_checks(self, connection: Any, 
                                   sample_data: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Run all applicable validation checks"""
        results = []
        
        for rule in self.validation_rules:
            # Skip platform-specific rules if not applicable
            if rule.platform_specific and connection.platform not in rule.applicable_platforms:
                continue
            
            try:
                result = await self._execute_validation_rule(rule, connection, sample_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Validation rule {rule.rule_id} failed: {str(e)}")
                results.append(ValidationResult(
                    rule_id=rule.rule_id,
                    status=ValidationStatus.FAILED,
                    score=0.0,
                    message=f"Validation failed: {str(e)}",
                    details={"error": str(e)},
                    recommendations=["Review validation rule implementation"]
                ))
        
        return results
    
    async def _execute_validation_rule(self, rule: ValidationRule, connection: Any,
                                     sample_data: List[Dict[str, Any]]) -> ValidationResult:
        """Execute a specific validation rule"""
        
        if rule.rule_id == "data_freshness_daily":
            return self._check_daily_freshness(sample_data)
        elif rule.rule_id == "data_freshness_weekly":
            return self._check_weekly_availability(sample_data)
        elif rule.rule_id == "required_fields_present":
            return self._check_required_fields(connection.platform, sample_data)
        elif rule.rule_id == "data_volume_consistency":
            return self._check_volume_consistency(sample_data)
        elif rule.rule_id == "metric_relationships":
            return self._check_metric_relationships(sample_data)
        elif rule.rule_id == "value_ranges":
            return self._check_value_ranges(connection.platform, sample_data)
        elif rule.rule_id == "cross_platform_consistency":
            return self._check_cross_platform_consistency(connection, sample_data)
        elif rule.rule_id == "temporal_consistency":
            return self._check_temporal_consistency(sample_data)
        else:
            # Default implementation for unknown rules
            return ValidationResult(
                rule_id=rule.rule_id,
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="Rule implementation not found",
                details={},
                recommendations=[]
            )
    
    def _check_daily_freshness(self, sample_data: List[Dict[str, Any]]) -> ValidationResult:
        """Check if data is fresh (updated within 24 hours)"""
        if not sample_data:
            return ValidationResult(
                rule_id="data_freshness_daily",
                status=ValidationStatus.FAILED,
                score=0.0,
                message="No data available",
                details={"data_count": 0},
                recommendations=["Check connection status and sync settings"]
            )
        
        # Find the most recent data point
        most_recent = None
        for record in sample_data:
            record_date = record.get('date')
            if record_date:
                if isinstance(record_date, str):
                    record_date = datetime.fromisoformat(record_date.replace('Z', '+00:00')).date()
                elif isinstance(record_date, datetime):
                    record_date = record_date.date()
                
                if most_recent is None or record_date > most_recent:
                    most_recent = record_date
        
        if most_recent is None:
            return ValidationResult(
                rule_id="data_freshness_daily",
                status=ValidationStatus.FAILED,
                score=0.0,
                message="No date information found in data",
                details={},
                recommendations=["Verify data format and date fields"]
            )
        
        days_old = (datetime.utcnow().date() - most_recent).days
        
        if days_old <= 1:
            score = 100.0
            status = ValidationStatus.PASSED
            message = f"Data is fresh (last updated {days_old} days ago)"
        elif days_old <= 3:
            score = 75.0
            status = ValidationStatus.WARNING
            message = f"Data is moderately fresh (last updated {days_old} days ago)"
        else:
            score = max(0.0, 100.0 - (days_old * 10))
            status = ValidationStatus.FAILED
            message = f"Data is stale (last updated {days_old} days ago)"
        
        return ValidationResult(
            rule_id="data_freshness_daily",
            status=status,
            score=score,
            message=message,
            details={"most_recent_date": most_recent.isoformat(), "days_old": days_old},
            recommendations=["Enable automatic sync", "Check platform API status"] if days_old > 1 else []
        )
    
    def _check_weekly_availability(self, sample_data: List[Dict[str, Any]]) -> ValidationResult:
        """Check if data is available for the past week"""
        if not sample_data:
            return ValidationResult(
                rule_id="data_freshness_weekly",
                status=ValidationStatus.FAILED,
                score=0.0,
                message="No data available",
                details={"data_count": 0},
                recommendations=["Check connection and sync data"]
            )
        
        # Count unique dates in the past 7 days
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=7)
        
        unique_dates = set()
        for record in sample_data:
            record_date = record.get('date')
            if record_date:
                if isinstance(record_date, str):
                    record_date = datetime.fromisoformat(record_date.replace('Z', '+00:00')).date()
                elif isinstance(record_date, datetime):
                    record_date = record_date.date()
                
                if start_date <= record_date <= end_date:
                    unique_dates.add(record_date)
        
        coverage_percentage = (len(unique_dates) / 7) * 100
        
        if coverage_percentage >= 85:
            status = ValidationStatus.PASSED
            score = 100.0
        elif coverage_percentage >= 60:
            status = ValidationStatus.WARNING
            score = 75.0
        else:
            status = ValidationStatus.FAILED
            score = coverage_percentage
        
        return ValidationResult(
            rule_id="data_freshness_weekly",
            status=status,
            score=score,
            message=f"Weekly data coverage: {coverage_percentage:.1f}%",
            details={
                "unique_dates": len(unique_dates),
                "expected_dates": 7,
                "coverage_percentage": coverage_percentage
            },
            recommendations=["Increase sync frequency", "Check for data gaps"] if coverage_percentage < 85 else []
        )
    
    def _check_required_fields(self, platform: str, sample_data: List[Dict[str, Any]]) -> ValidationResult:
        """Check if required fields are present"""
        if not sample_data:
            return ValidationResult(
                rule_id="required_fields_present",
                status=ValidationStatus.FAILED,
                score=0.0,
                message="No data to validate",
                details={},
                recommendations=["Sync data from platform"]
            )
        
        # Define required fields by platform
        required_fields = {
            'meta_business': ['campaign_id', 'spend', 'impressions', 'clicks'],
            'google_ads': ['campaign_id', 'cost', 'impressions', 'clicks'],
            'klaviyo': ['campaign_id', 'recipients', 'delivered', 'opened'],
            'shopify': ['orders', 'revenue'],
            'default': ['date', 'campaign_id']
        }
        
        platform_fields = required_fields.get(platform, required_fields['default'])
        
        # Check field presence across all records
        field_presence = {field: 0 for field in platform_fields}
        total_records = len(sample_data)
        
        for record in sample_data:
            for field in platform_fields:
                if field in record and record[field] is not None:
                    field_presence[field] += 1
        
        # Calculate completeness percentage for each field
        field_completeness = {
            field: (count / total_records) * 100
            for field, count in field_presence.items()
        }
        
        overall_completeness = sum(field_completeness.values()) / len(field_completeness)
        
        if overall_completeness >= 95:
            status = ValidationStatus.PASSED
            score = 100.0
        elif overall_completeness >= 80:
            status = ValidationStatus.WARNING
            score = 85.0
        else:
            status = ValidationStatus.FAILED
            score = overall_completeness
        
        missing_fields = [field for field, completeness in field_completeness.items() if completeness < 95]
        
        return ValidationResult(
            rule_id="required_fields_present",
            status=status,
            score=score,
            message=f"Field completeness: {overall_completeness:.1f}%",
            details={
                "field_completeness": field_completeness,
                "missing_fields": missing_fields,
                "total_records": total_records
            },
            recommendations=[f"Review data extraction for fields: {', '.join(missing_fields)}"] if missing_fields else []
        )
    
    def _check_metric_relationships(self, sample_data: List[Dict[str, Any]]) -> ValidationResult:
        """Check logical relationships between metrics"""
        if not sample_data:
            return ValidationResult(
                rule_id="metric_relationships",
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="No data to validate",
                details={},
                recommendations=[]
            )
        
        relationship_violations = []
        total_checks = 0
        
        for record in sample_data:
            # Check CTR = (Clicks / Impressions) * 100
            if 'clicks' in record and 'impressions' in record and 'ctr' in record:
                clicks = record.get('clicks', 0)
                impressions = record.get('impressions', 0)
                reported_ctr = record.get('ctr', 0)
                
                if impressions > 0:
                    calculated_ctr = (clicks / impressions) * 100
                    if abs(calculated_ctr - reported_ctr) > 0.1:  # Allow small rounding differences
                        relationship_violations.append({
                            'metric': 'CTR',
                            'calculated': calculated_ctr,
                            'reported': reported_ctr,
                            'record_id': record.get('id', 'unknown')
                        })
                total_checks += 1
            
            # Check CPC = Spend / Clicks
            if 'spend' in record and 'clicks' in record and 'cpc' in record:
                spend = record.get('spend', 0)
                clicks = record.get('clicks', 0)
                reported_cpc = record.get('cpc', 0)
                
                if clicks > 0:
                    calculated_cpc = spend / clicks
                    if abs(calculated_cpc - reported_cpc) > 0.01:
                        relationship_violations.append({
                            'metric': 'CPC',
                            'calculated': calculated_cpc,
                            'reported': reported_cpc,
                            'record_id': record.get('id', 'unknown')
                        })
                total_checks += 1
        
        if total_checks == 0:
            return ValidationResult(
                rule_id="metric_relationships",
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="No metric relationships to validate",
                details={},
                recommendations=[]
            )
        
        violation_rate = len(relationship_violations) / total_checks
        score = max(0.0, (1 - violation_rate) * 100)
        
        if violation_rate == 0:
            status = ValidationStatus.PASSED
        elif violation_rate < 0.05:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            rule_id="metric_relationships",
            status=status,
            score=score,
            message=f"Metric relationship accuracy: {(1-violation_rate)*100:.1f}%",
            details={
                "total_checks": total_checks,
                "violations": len(relationship_violations),
                "violation_rate": violation_rate,
                "sample_violations": relationship_violations[:5]
            },
            recommendations=["Review data extraction logic", "Check platform API documentation"] if violation_rate > 0 else []
        )
    
    def _check_volume_consistency(self, sample_data: List[Dict[str, Any]]) -> ValidationResult:
        """Check data volume consistency"""
        if len(sample_data) < 3:
            return ValidationResult(
                rule_id="data_volume_consistency",
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="Insufficient data for volume analysis",
                details={"record_count": len(sample_data)},
                recommendations=[]
            )
        
        # Group data by date and count records per day
        daily_counts = {}
        for record in sample_data:
            record_date = record.get('date')
            if record_date:
                if isinstance(record_date, str):
                    record_date = datetime.fromisoformat(record_date.replace('Z', '+00:00')).date()
                elif isinstance(record_date, datetime):
                    record_date = record_date.date()
                
                daily_counts[record_date] = daily_counts.get(record_date, 0) + 1
        
        if len(daily_counts) < 2:
            return ValidationResult(
                rule_id="data_volume_consistency",
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="Insufficient date range for volume analysis",
                details={"unique_dates": len(daily_counts)},
                recommendations=[]
            )
        
        # Calculate coefficient of variation
        counts = list(daily_counts.values())
        mean_count = statistics.mean(counts)
        std_count = statistics.stdev(counts) if len(counts) > 1 else 0
        
        cv = (std_count / mean_count) if mean_count > 0 else 0
        
        # Score based on coefficient of variation
        if cv <= 0.2:  # Low variation
            score = 100.0
            status = ValidationStatus.PASSED
        elif cv <= 0.5:  # Moderate variation
            score = 75.0
            status = ValidationStatus.WARNING
        else:  # High variation
            score = max(0.0, 100.0 - (cv * 100))
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            rule_id="data_volume_consistency",
            status=status,
            score=score,
            message=f"Data volume consistency: CV = {cv:.2f}",
            details={
                "daily_counts": daily_counts,
                "mean_count": mean_count,
                "std_count": std_count,
                "coefficient_of_variation": cv
            },
            recommendations=["Review sync schedule", "Check for data gaps"] if cv > 0.5 else []
        )
    
    def _check_value_ranges(self, platform: str, sample_data: List[Dict[str, Any]]) -> ValidationResult:
        """Check if metric values are within expected ranges"""
        if not sample_data:
            return ValidationResult(
                rule_id="value_ranges",
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="No data to validate",
                details={},
                recommendations=[]
            )
        
        # Define expected ranges by platform and metric
        expected_ranges = {
            'meta_business': {
                'ctr': (0, 20),  # CTR should be 0-20%
                'cpc': (0, 50),  # CPC should be $0-$50
                'frequency': (1, 10)  # Frequency should be 1-10
            },
            'google_ads': {
                'ctr': (0, 15),
                'average_cpc': (0, 100),
                'quality_score': (1, 10)
            },
            'default': {
                'ctr': (0, 25),
                'cpc': (0, 100)
            }
        }
        
        platform_ranges = expected_ranges.get(platform, expected_ranges['default'])
        
        violations = []
        total_checks = 0
        
        for record in sample_data:
            for metric, (min_val, max_val) in platform_ranges.items():
                if metric in record:
                    value = record[metric]
                    if value is not None and (value < min_val or value > max_val):
                        violations.append({
                            'metric': metric,
                            'value': value,
                            'expected_range': (min_val, max_val),
                            'record_id': record.get('id', 'unknown')
                        })
                    total_checks += 1
        
        if total_checks == 0:
            return ValidationResult(
                rule_id="value_ranges",
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="No metrics to validate",
                details={},
                recommendations=[]
            )
        
        violation_rate = len(violations) / total_checks
        score = max(0.0, (1 - violation_rate) * 100)
        
        if violation_rate == 0:
            status = ValidationStatus.PASSED
        elif violation_rate < 0.1:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            rule_id="value_ranges",
            status=status,
            score=score,
            message=f"Value range compliance: {(1-violation_rate)*100:.1f}%",
            details={
                "total_checks": total_checks,
                "violations": len(violations),
                "violation_rate": violation_rate,
                "sample_violations": violations[:5]
            },
            recommendations=["Review outlier values", "Check data extraction logic"] if violation_rate > 0 else []
        )
    
    def _check_cross_platform_consistency(self, connection: Any, sample_data: List[Dict[str, Any]]) -> ValidationResult:
        """Check consistency across platforms (placeholder for future implementation)"""
        return ValidationResult(
            rule_id="cross_platform_consistency",
            status=ValidationStatus.SKIPPED,
            score=100.0,
            message="Cross-platform validation not yet implemented",
            details={},
            recommendations=[]
        )
    
    def _check_temporal_consistency(self, sample_data: List[Dict[str, Any]]) -> ValidationResult:
        """Check temporal consistency of data"""
        if len(sample_data) < 3:
            return ValidationResult(
                rule_id="temporal_consistency",
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="Insufficient data for temporal analysis",
                details={"record_count": len(sample_data)},
                recommendations=[]
            )
        
        # Sort data by date
        dated_records = []
        for record in sample_data:
            record_date = record.get('date')
            if record_date:
                if isinstance(record_date, str):
                    record_date = datetime.fromisoformat(record_date.replace('Z', '+00:00')).date()
                elif isinstance(record_date, datetime):
                    record_date = record_date.date()
                dated_records.append((record_date, record))
        
        dated_records.sort(key=lambda x: x[0])
        
        if len(dated_records) < 3:
            return ValidationResult(
                rule_id="temporal_consistency",
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="Insufficient dated records",
                details={"dated_records": len(dated_records)},
                recommendations=[]
            )
        
        # Check for temporal anomalies (simplified implementation)
        anomalies = 0
        total_checks = 0
        
        # Look for sudden spikes or drops in key metrics
        for metric in ['spend', 'impressions', 'clicks', 'revenue']:
            values = []
            for date, record in dated_records:
                if metric in record and record[metric] is not None:
                    values.append(record[metric])
            
            if len(values) >= 3:
                # Simple anomaly detection: values more than 3 standard deviations from mean
                if len(values) > 1:
                    mean_val = statistics.mean(values)
                    std_val = statistics.stdev(values) if len(values) > 1 else 0
                    
                    for value in values:
                        if std_val > 0 and abs(value - mean_val) > 3 * std_val:
                            anomalies += 1
                        total_checks += 1
        
        if total_checks == 0:
            return ValidationResult(
                rule_id="temporal_consistency",
                status=ValidationStatus.SKIPPED,
                score=100.0,
                message="No metrics for temporal analysis",
                details={},
                recommendations=[]
            )
        
        anomaly_rate = anomalies / total_checks
        score = max(0.0, (1 - anomaly_rate) * 100)
        
        if anomaly_rate == 0:
            status = ValidationStatus.PASSED
        elif anomaly_rate < 0.05:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED
        
        return ValidationResult(
            rule_id="temporal_consistency",
            status=status,
            score=score,
            message=f"Temporal consistency: {(1-anomaly_rate)*100:.1f}%",
            details={
                "total_checks": total_checks,
                "anomalies": anomalies,
                "anomaly_rate": anomaly_rate
            },
            recommendations=["Review data for outliers", "Check for campaign changes"] if anomaly_rate > 0 else []
        )
    
    def _calculate_overall_score(self, validation_results: List[ValidationResult]) -> float:
        """Calculate overall quality score from validation results"""
        if not validation_results:
            return 0.0
        
        # Weight scores by severity
        severity_weights = {
            'critical': 3.0,
            'high': 2.0,
            'medium': 1.5,
            'low': 1.0
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for result in validation_results:
            if result.status != ValidationStatus.SKIPPED:
                rule = next((r for r in self.validation_rules if r.rule_id == result.rule_id), None)
                weight = severity_weights.get(rule.severity if rule else 'medium', 1.0)
                
                weighted_sum += result.score * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """Determine quality level from score"""
        if score >= 90:
            return DataQualityLevel.EXCELLENT
        elif score >= 75:
            return DataQualityLevel.GOOD
        elif score >= 60:
            return DataQualityLevel.FAIR
        elif score >= 40:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL
    
    def _analyze_data_freshness(self, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data freshness metrics"""
        if not sample_data:
            return {
                "most_recent_date": None,
                "oldest_date": None,
                "data_span_days": 0,
                "freshness_score": 0.0
            }
        
        dates = []
        for record in sample_data:
            record_date = record.get('date')
            if record_date:
                if isinstance(record_date, str):
                    record_date = datetime.fromisoformat(record_date.replace('Z', '+00:00')).date()
                elif isinstance(record_date, datetime):
                    record_date = record_date.date()
                dates.append(record_date)
        
        if not dates:
            return {
                "most_recent_date": None,
                "oldest_date": None,
                "data_span_days": 0,
                "freshness_score": 0.0
            }
        
        most_recent = max(dates)
        oldest = min(dates)
        data_span_days = (most_recent - oldest).days
        days_since_recent = (datetime.utcnow().date() - most_recent).days
        
        # Calculate freshness score (100 for same day, decreasing)
        freshness_score = max(0.0, 100.0 - (days_since_recent * 10))
        
        return {
            "most_recent_date": most_recent.isoformat(),
            "oldest_date": oldest.isoformat(),
            "data_span_days": data_span_days,
            "days_since_recent": days_since_recent,
            "freshness_score": freshness_score,
            "unique_dates": len(set(dates))
        }
    
    def _analyze_completeness(self, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data completeness metrics"""
        if not sample_data:
            return {
                "total_records": 0,
                "field_completeness": {},
                "overall_completeness": 0.0
            }
        
        total_records = len(sample_data)
        all_fields = set()
        
        # Collect all possible fields
        for record in sample_data:
            all_fields.update(record.keys())
        
        # Calculate completeness for each field
        field_completeness = {}
        for field in all_fields:
            non_null_count = sum(1 for record in sample_data if record.get(field) is not None)
            field_completeness[field] = (non_null_count / total_records) * 100
        
        overall_completeness = sum(field_completeness.values()) / len(field_completeness) if field_completeness else 0.0
        
        return {
            "total_records": total_records,
            "total_fields": len(all_fields),
            "field_completeness": field_completeness,
            "overall_completeness": overall_completeness,
            "complete_records": sum(1 for record in sample_data if all(v is not None for v in record.values()))
        }
    
    def _analyze_consistency(self, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data consistency metrics"""
        if not sample_data:
            return {
                "consistency_score": 0.0,
                "data_type_consistency": {},
                "value_distribution": {}
            }
        
        # Analyze data type consistency
        field_types = {}
        for record in sample_data:
            for field, value in record.items():
                if value is not None:
                    value_type = type(value).__name__
                    if field not in field_types:
                        field_types[field] = {}
                    field_types[field][value_type] = field_types[field].get(value_type, 0) + 1
        
        # Calculate type consistency score
        type_consistency_scores = {}
        for field, types in field_types.items():
            total_values = sum(types.values())
            max_type_count = max(types.values()) if types else 0
            type_consistency_scores[field] = (max_type_count / total_values) * 100 if total_values > 0 else 0
        
        overall_consistency = sum(type_consistency_scores.values()) / len(type_consistency_scores) if type_consistency_scores else 0.0
        
        # Analyze value distributions for key metrics
        value_distributions = {}
        for metric in ['spend', 'impressions', 'clicks', 'revenue']:
            values = [record.get(metric) for record in sample_data if record.get(metric) is not None]
            if values:
                value_distributions[metric] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values)
                }
        
        return {
            "consistency_score": overall_consistency,
            "data_type_consistency": type_consistency_scores,
            "value_distribution": value_distributions,
            "field_types": field_types
        }
    
    def _analyze_reliability(self, connection: Any, sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze data reliability metrics"""
        # This would typically include historical sync success rates,
        # API response times, error rates, etc.
        # For now, providing a basic implementation
        
        return {
            "connection_status": connection.status,
            "last_sync": connection.last_sync_at.isoformat() if connection.last_sync_at else None,
            "data_volume": len(sample_data),
            "reliability_score": 85.0,  # Placeholder - would be calculated from historical data
            "estimated_completeness": min(100.0, (len(sample_data) / 30) * 100) if sample_data else 0.0  # Assuming 30 records expected per week
        }
    
    def _generate_recommendations(self, validation_results: List[ValidationResult],
                                connection: Any) -> List[str]:
        """Generate actionable recommendations based on validation results"""
        recommendations = []
        
        # Collect all recommendations from validation results
        for result in validation_results:
            recommendations.extend(result.recommendations)
        
        # Add general recommendations based on overall patterns
        failed_rules = [r for r in validation_results if r.status == ValidationStatus.FAILED]
        warning_rules = [r for r in validation_results if r.status == ValidationStatus.WARNING]
        
        if len(failed_rules) > 3:
            recommendations.append("Consider reviewing connection configuration due to multiple validation failures")
        
        if len(warning_rules) > 2:
            recommendations.append("Monitor data quality trends and consider adjusting sync frequency")
        
        # Platform-specific recommendations
        if connection.platform == "meta_business":
            recommendations.append("Ensure Meta Business account has proper permissions for data access")
        elif connection.platform == "google_ads":
            recommendations.append("Verify Google Ads API access and account linking")
        
        # Remove duplicates and return
        return list(set(recommendations))

# Global service instance
_validator_service = None

async def get_data_source_validator() -> DataSourceValidator:
    """Get the global data source validator instance"""
    global _validator_service
    if _validator_service is None:
        _validator_service = DataSourceValidator()
        await _validator_service.initialize()
    return _validator_service