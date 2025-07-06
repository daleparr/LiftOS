"""
Data Quality Engine for the Agentic microservice.

This module provides comprehensive data quality evaluation capabilities to ensure
highly accurate outcomes in AI agent testing and evaluation processes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import json
import re
from statistics import mean, median, stdev

from utils.config import AgenticConfig
from utils.logging_config import get_logger


class DataQualityDimension(str, Enum):
    """Data quality dimensions for evaluation."""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"
    RELEVANCE = "relevance"
    INTEGRITY = "integrity"


class QualityLevel(str, Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"           # 85-94%
    ACCEPTABLE = "acceptable"  # 70-84%
    POOR = "poor"           # 50-69%
    CRITICAL = "critical"   # <50%


@dataclass
class QualityMetric:
    """Individual quality metric result."""
    dimension: DataQualityDimension
    score: float  # 0.0 to 1.0
    level: QualityLevel
    description: str
    issues_found: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report."""
    dataset_id: str
    evaluation_timestamp: datetime
    overall_score: float
    overall_level: QualityLevel
    dimension_scores: Dict[DataQualityDimension, QualityMetric]
    critical_issues: List[str]
    recommendations: List[str]
    data_profile: Dict[str, Any]
    quality_trends: Optional[Dict[str, List[float]]] = None


class DataQualityRule(BaseModel):
    """Data quality validation rule."""
    rule_id: str
    name: str
    dimension: DataQualityDimension
    description: str
    rule_type: str  # "threshold", "pattern", "range", "custom"
    parameters: Dict[str, Any]
    weight: float = 1.0
    is_critical: bool = False


class DataQualityEngine:
    """
    Comprehensive data quality evaluation engine for ensuring accurate outcomes.
    
    This engine evaluates data across multiple dimensions and provides detailed
    reports with actionable recommendations for improvement.
    """
    
    def __init__(self, config: AgenticConfig):
        """Initialize the data quality engine."""
        self.config = config
        self.logger = get_logger(__name__)
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.95,
            QualityLevel.GOOD: 0.85,
            QualityLevel.ACCEPTABLE: 0.70,
            QualityLevel.POOR: 0.50
        }
        
        # Dimension weights for overall score calculation
        self.dimension_weights = {
            DataQualityDimension.COMPLETENESS: 0.20,
            DataQualityDimension.ACCURACY: 0.25,
            DataQualityDimension.CONSISTENCY: 0.15,
            DataQualityDimension.VALIDITY: 0.15,
            DataQualityDimension.UNIQUENESS: 0.10,
            DataQualityDimension.TIMELINESS: 0.05,
            DataQualityDimension.RELEVANCE: 0.05,
            DataQualityDimension.INTEGRITY: 0.05
        }
    
    async def evaluate_data_quality(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        dataset_id: str,
        custom_rules: Optional[List[DataQualityRule]] = None,
        include_profiling: bool = True
    ) -> DataQualityReport:
        """
        Perform comprehensive data quality evaluation.
        
        Args:
            data: Data to evaluate (DataFrame, dict, or list of dicts)
            dataset_id: Unique identifier for the dataset
            custom_rules: Additional custom quality rules
            include_profiling: Whether to include data profiling
            
        Returns:
            Comprehensive data quality report
        """
        self.logger.info(f"Starting data quality evaluation for dataset: {dataset_id}")
        
        try:
            # Convert data to DataFrame for consistent processing
            df = self._normalize_data(data)
            
            # Evaluate each quality dimension
            dimension_results = {}
            
            # Completeness evaluation
            dimension_results[DataQualityDimension.COMPLETENESS] = await self._evaluate_completeness(df)
            
            # Accuracy evaluation
            dimension_results[DataQualityDimension.ACCURACY] = await self._evaluate_accuracy(df)
            
            # Consistency evaluation
            dimension_results[DataQualityDimension.CONSISTENCY] = await self._evaluate_consistency(df)
            
            # Validity evaluation
            dimension_results[DataQualityDimension.VALIDITY] = await self._evaluate_validity(df)
            
            # Uniqueness evaluation
            dimension_results[DataQualityDimension.UNIQUENESS] = await self._evaluate_uniqueness(df)
            
            # Timeliness evaluation
            dimension_results[DataQualityDimension.TIMELINESS] = await self._evaluate_timeliness(df)
            
            # Relevance evaluation
            dimension_results[DataQualityDimension.RELEVANCE] = await self._evaluate_relevance(df)
            
            # Integrity evaluation
            dimension_results[DataQualityDimension.INTEGRITY] = await self._evaluate_integrity(df)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(dimension_results)
            overall_level = self._determine_quality_level(overall_score)
            
            # Identify critical issues
            critical_issues = self._identify_critical_issues(dimension_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(dimension_results, critical_issues)
            
            # Generate data profile if requested
            data_profile = {}
            if include_profiling:
                data_profile = await self._generate_data_profile(df)
            
            # Create comprehensive report
            report = DataQualityReport(
                dataset_id=dataset_id,
                evaluation_timestamp=datetime.utcnow(),
                overall_score=overall_score,
                overall_level=overall_level,
                dimension_scores=dimension_results,
                critical_issues=critical_issues,
                recommendations=recommendations,
                data_profile=data_profile
            )
            
            self.logger.info(f"Data quality evaluation completed. Overall score: {overall_score:.3f} ({overall_level})")
            return report
            
        except Exception as e:
            self.logger.error(f"Error during data quality evaluation: {str(e)}")
            raise
    
    def _normalize_data(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """Convert input data to pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    async def _evaluate_completeness(self, df: pd.DataFrame) -> QualityMetric:
        """Evaluate data completeness."""
        issues = []
        recommendations = []
        
        # Calculate completeness ratio
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_ratio = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # Check column-wise completeness
        column_completeness = {}
        for column in df.columns:
            col_completeness = 1 - (df[column].isnull().sum() / len(df))
            column_completeness[column] = col_completeness
            
            if col_completeness < 0.9:
                issues.append(f"Column '{column}' has low completeness: {col_completeness:.2%}")
                recommendations.append(f"Investigate and address missing values in column '{column}'")
        
        level = self._determine_quality_level(completeness_ratio)
        
        return QualityMetric(
            dimension=DataQualityDimension.COMPLETENESS,
            score=completeness_ratio,
            level=level,
            description=f"Data completeness: {completeness_ratio:.2%}",
            issues_found=issues,
            recommendations=recommendations,
            metadata={"column_completeness": column_completeness}
        )
    
    async def _evaluate_accuracy(self, df: pd.DataFrame) -> QualityMetric:
        """Evaluate data accuracy."""
        issues = []
        recommendations = []
        accuracy_score = 1.0
        
        # Check for outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for column in numeric_columns:
            if df[column].notna().sum() > 0:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
                outlier_ratio = len(outliers) / len(df[column].dropna())
                outlier_info[column] = {
                    "count": len(outliers),
                    "ratio": outlier_ratio,
                    "bounds": [lower_bound, upper_bound]
                }
                
                if outlier_ratio > 0.05:  # More than 5% outliers
                    issues.append(f"Column '{column}' has {outlier_ratio:.2%} outliers")
                    recommendations.append(f"Review outliers in column '{column}' for data entry errors")
                    accuracy_score *= (1 - outlier_ratio * 0.5)
        
        level = self._determine_quality_level(accuracy_score)
        
        return QualityMetric(
            dimension=DataQualityDimension.ACCURACY,
            score=accuracy_score,
            level=level,
            description=f"Data accuracy assessment: {accuracy_score:.2%}",
            issues_found=issues,
            recommendations=recommendations,
            metadata={"outlier_info": outlier_info}
        )
    
    async def _evaluate_consistency(self, df: pd.DataFrame) -> QualityMetric:
        """Evaluate data consistency."""
        issues = []
        recommendations = []
        consistency_score = 1.0
        
        # Check format consistency within columns
        format_consistency = {}
        for column in df.select_dtypes(include=['object']).columns:
            if df[column].notna().sum() > 0:
                values = df[column].dropna().astype(str)
                if len(values) > 0:
                    all_upper = values.str.isupper().sum()
                    all_lower = values.str.islower().sum()
                    mixed_case = len(values) - all_upper - all_lower
                    
                    if mixed_case > 0 and (all_upper > 0 or all_lower > 0):
                        inconsistency_ratio = mixed_case / len(values)
                        format_consistency[column] = {
                            "case_inconsistency": inconsistency_ratio,
                            "all_upper": all_upper,
                            "all_lower": all_lower,
                            "mixed": mixed_case
                        }
                        
                        if inconsistency_ratio > 0.1:
                            issues.append(f"Column '{column}' has inconsistent case formatting")
                            recommendations.append(f"Standardize case formatting in column '{column}'")
                            consistency_score *= (1 - inconsistency_ratio * 0.3)
        
        level = self._determine_quality_level(consistency_score)
        
        return QualityMetric(
            dimension=DataQualityDimension.CONSISTENCY,
            score=consistency_score,
            level=level,
            description=f"Data consistency: {consistency_score:.2%}",
            issues_found=issues,
            recommendations=recommendations,
            metadata={"format_consistency": format_consistency}
        )
    
    async def _evaluate_validity(self, df: pd.DataFrame) -> QualityMetric:
        """Evaluate data validity."""
        issues = []
        recommendations = []
        validity_score = 1.0
        validation_results = {}
        
        # Email validation
        email_columns = [col for col in df.columns if 'email' in col.lower()]
        for column in email_columns:
            if df[column].notna().sum() > 0:
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid_emails = df[column].dropna().str.match(email_pattern).sum()
                total_emails = df[column].notna().sum()
                validity_ratio = valid_emails / total_emails if total_emails > 0 else 1
                
                validation_results[column] = {
                    "type": "email",
                    "valid_count": valid_emails,
                    "total_count": total_emails,
                    "validity_ratio": validity_ratio
                }
                
                if validity_ratio < 0.95:
                    issues.append(f"Column '{column}' has {(1-validity_ratio):.2%} invalid email addresses")
                    recommendations.append(f"Validate and correct email addresses in column '{column}'")
                    validity_score *= validity_ratio
        
        level = self._determine_quality_level(validity_score)
        
        return QualityMetric(
            dimension=DataQualityDimension.VALIDITY,
            score=validity_score,
            level=level,
            description=f"Data validity: {validity_score:.2%}",
            issues_found=issues,
            recommendations=recommendations,
            metadata={"validation_results": validation_results}
        )
    
    async def _evaluate_uniqueness(self, df: pd.DataFrame) -> QualityMetric:
        """Evaluate data uniqueness."""
        issues = []
        recommendations = []
        uniqueness_score = 1.0
        
        # Check for duplicate rows
        total_rows = len(df)
        unique_rows = len(df.drop_duplicates())
        duplicate_ratio = (total_rows - unique_rows) / total_rows if total_rows > 0 else 0
        
        if duplicate_ratio > 0:
            issues.append(f"Found {total_rows - unique_rows} duplicate rows ({duplicate_ratio:.2%})")
            recommendations.append("Remove or investigate duplicate rows")
            uniqueness_score *= (1 - duplicate_ratio)
        
        level = self._determine_quality_level(uniqueness_score)
        
        return QualityMetric(
            dimension=DataQualityDimension.UNIQUENESS,
            score=uniqueness_score,
            level=level,
            description=f"Data uniqueness: {uniqueness_score:.2%}",
            issues_found=issues,
            recommendations=recommendations,
            metadata={"duplicate_ratio": duplicate_ratio}
        )
    
    async def _evaluate_timeliness(self, df: pd.DataFrame) -> QualityMetric:
        """Evaluate data timeliness."""
        issues = []
        recommendations = []
        timeliness_score = 1.0
        
        # Check for timestamp columns
        timestamp_columns = []
        for column in df.columns:
            if any(keyword in column.lower() for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                timestamp_columns.append(column)
        
        timeliness_info = {}
        current_time = datetime.now()
        
        for column in timestamp_columns:
            try:
                date_series = pd.to_datetime(df[column], errors='coerce')
                valid_dates = date_series.dropna()
                
                if len(valid_dates) > 0:
                    latest_date = valid_dates.max()
                    age_days = (current_time - latest_date).days
                    
                    timeliness_info[column] = {
                        "latest_date": latest_date.isoformat() if pd.notna(latest_date) else None,
                        "age_days": age_days
                    }
                    
                    # Check if data is too old (configurable threshold)
                    max_age_days = 30  # Default threshold
                    if age_days > max_age_days:
                        issues.append(f"Data in column '{column}' is {age_days} days old")
                        recommendations.append(f"Update data in column '{column}' to ensure timeliness")
                        timeliness_score *= max(0.5, 1 - (age_days - max_age_days) / 365)
            except:
                pass
        
        level = self._determine_quality_level(timeliness_score)
        
        return QualityMetric(
            dimension=DataQualityDimension.TIMELINESS,
            score=timeliness_score,
            level=level,
            description=f"Data timeliness: {timeliness_score:.2%}",
            issues_found=issues,
            recommendations=recommendations,
            metadata={"timeliness_info": timeliness_info}
        )
    
    async def _evaluate_relevance(self, df: pd.DataFrame) -> QualityMetric:
        """Evaluate data relevance."""
        issues = []
        recommendations = []
        relevance_score = 1.0
        
        # Check for empty or mostly empty columns
        column_relevance = {}
        for column in df.columns:
            non_null_ratio = df[column].notna().sum() / len(df) if len(df) > 0 else 0
            unique_ratio = df[column].nunique() / len(df) if len(df) > 0 else 0
            
            column_relevance[column] = {
                "non_null_ratio": non_null_ratio,
                "unique_ratio": unique_ratio,
                "relevance_score": (non_null_ratio + unique_ratio) / 2
            }
            
            if non_null_ratio < 0.1:
                issues.append(f"Column '{column}' is mostly empty ({non_null_ratio:.2%} filled)")
                recommendations.append(f"Consider removing or investigating column '{column}'")
                relevance_score *= 0.9
        
        level = self._determine_quality_level(relevance_score)
        
        return QualityMetric(
            dimension=DataQualityDimension.RELEVANCE,
            score=relevance_score,
            level=level,
            description=f"Data relevance: {relevance_score:.2%}",
            issues_found=issues,
            recommendations=recommendations,
            metadata={"column_relevance": column_relevance}
        )
    
    async def _evaluate_integrity(self, df: pd.DataFrame) -> QualityMetric:
        """Evaluate data integrity."""
        issues = []
        recommendations = []
        integrity_score = 1.0
        
        # Check for referential integrity (basic checks)
        integrity_checks = {}
        
        # Check for orphaned references (columns ending with _id)
        id_columns = [col for col in df.columns if col.endswith('_id')]
        for column in id_columns:
            if df[column].notna().sum() > 0:
                null_references = df[column].isnull().sum()
                total_references = len(df)
                null_ratio = null_references / total_references if total_references > 0 else 0
                
                integrity_checks[column] = {
                    "null_references": null_references,
                    "total_references": total_references,
                    "null_ratio": null_ratio
                }
                
                if null_ratio > 0.05:
                    issues.append(f"Column '{column}' has {null_ratio:.2%} null references")
                    recommendations.append(f"Investigate null references in column '{column}'")
                    integrity_score *= (1 - null_ratio * 0.5)
        
        level = self._determine_quality_level(integrity_score)
        
        return QualityMetric(
            dimension=DataQualityDimension.INTEGRITY,
            score=integrity_score,
            level=level,
            description=f"Data integrity: {integrity_score:.2%}",
            issues_found=issues,
            recommendations=recommendations,
            metadata={"integrity_checks": integrity_checks}
        )
    
    def _calculate_overall_score(self, dimension_results: Dict[DataQualityDimension, QualityMetric]) -> float:
        """Calculate weighted overall quality score."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for dimension, metric in dimension_results.items():
            weight = self.dimension_weights.get(dimension, 1.0)
            total_weighted_score += metric.score * weight
            total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level based on score."""
        if score >= self.quality_thresholds[QualityLevel.EXCELLENT]:
            return QualityLevel.EXCELLENT
        elif score >= self.quality_thresholds[QualityLevel.GOOD]:
            return QualityLevel.GOOD
        elif score >= self.quality_thresholds[QualityLevel.ACCEPTABLE]:
            return QualityLevel.ACCEPTABLE
        elif score >= self.quality_thresholds[QualityLevel.POOR]:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL
    
    def _identify_critical_issues(self, dimension_results: Dict[DataQualityDimension, QualityMetric]) -> List[str]:
        """Identify critical data quality issues."""
        critical_issues = []
        
        for dimension, metric in dimension_results.items():
            if metric.level == QualityLevel.CRITICAL:
                critical_issues.append(f"CRITICAL: {dimension.value} score is critically low ({metric.score:.2%})")
        
        return critical_issues
    
    def _generate_recommendations(
        self,
        dimension_results: Dict[DataQualityDimension, QualityMetric],
        critical_issues: List[str]
    ) -> List[str]:
        """Generate actionable recommendations for data quality improvement."""
        recommendations = []
        
        # Priority recommendations for critical issues
        if critical_issues:
            recommendations.append("IMMEDIATE ACTION REQUIRED: Address critical data quality issues before proceeding")
        
        # Dimension-specific recommendations
        for dimension, metric in dimension_results.items():
            if metric.level in [QualityLevel.POOR, QualityLevel.CRITICAL]:
                recommendations.extend(metric.recommendations)
        
        return list(set(recommendations))  # Remove duplicates
    
    async def _generate_data_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data profile."""
        profile = {
            "basic_stats": {
                "row_count": len(df),
                "column_count": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "duplicate_rows": len(df) - len(df.drop_duplicates())
            },
            "column_profiles": {},
            "data_types": df.dtypes.astype(str).to_dict(),
            "missing_data": df.isnull().sum().to_dict()
        }
        
        # Generate column-specific profiles
        for column in df.columns:
            col_profile = {
                "data_type": str(df[column].dtype),
                "non_null_count": int(df[column].notna().sum()),
                "null_count": int(df[column].isnull().sum()),
                "unique_count": int(df[column].nunique())
            }
            
            # Add statistics for numeric columns
            if df[column].dtype in ['int64', 'float64']:
                col_profile.update({
                    "min": float(df[column].min()) if pd.notna(df[column].min()) else None,
                    "max": float(df[column].max()) if pd.notna(df[column].max()) else None,
                    "mean": float(df[column].mean()) if pd.notna(df[column].mean()) else None,
                    "median": float(df[column].median()) if pd.notna(df[column].median()) else None
                })
            
            profile["column_profiles"][column] = col_profile
        
        return profile
    
    async def validate_data_for_agent_testing(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        test_type: str = "marketing_campaign"
    ) -> Tuple[bool, DataQualityReport]:
        """
        Validate data quality specifically for agent testing scenarios.
        
        Args:
            data: Data to validate
            test_type: Type of test (affects validation rules)
            
        Returns:
            Tuple of (is_valid, quality_report)
        """
        # Perform quality evaluation
        report = await self.evaluate_data_quality(
            data=data,
            dataset_id=f"{test_type}_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            include_profiling=True
        )
        
        # Determine if data is valid for testing
        is_valid = (
            report.overall_score >= 0.85 and  # Minimum 85% overall quality
            len(report.critical_issues) == 0 and  # No critical issues
            all(
                metric.score >= 0.8 for dimension, metric in report.dimension_scores.items()
                if dimension in [DataQualityDimension.COMPLETENESS, DataQualityDimension.ACCURACY]
            )  # Key dimensions must be high quality
        )
        
        return is_valid, report