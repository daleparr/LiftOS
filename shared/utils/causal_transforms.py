"""
Causal Data Transformation Utilities
Handles transformation of raw marketing data for causal inference
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
import asyncio
from dataclasses import dataclass

from ..models.causal_marketing import (
    CausalMarketingData, ConfounderVariable, ExternalFactor, 
    CalendarDimension, TreatmentType, RandomizationUnit,
    DataQualityAssessment, CausalGraph
)
from ..models.marketing import DataSource, CampaignObjective, AdStatus
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConfounderDetectionResult:
    """Result from confounder detection"""
    confounders: List[ConfounderVariable]
    detection_confidence: float
    detection_method: str
    recommendations: List[str]


@dataclass
class TreatmentAssignmentResult:
    """Result from treatment assignment"""
    treatment_group: str
    treatment_type: TreatmentType
    treatment_intensity: float
    randomization_unit: RandomizationUnit
    experiment_id: Optional[str]
    assignment_confidence: float


class ExternalFactorProvider:
    """Provides external factors that may confound causal relationships"""
    
    def __init__(self):
        self.economic_indicators = {}
        self.market_conditions = {}
        self.competitor_data = {}
        self.seasonal_patterns = {}
    
    async def get_external_factors(self, timestamp: datetime, org_id: str) -> List[ExternalFactor]:
        """Get external factors for a given timestamp"""
        factors = []
        
        # Economic indicators
        economic_factors = await self._get_economic_indicators(timestamp)
        factors.extend(economic_factors)
        
        # Market conditions
        market_factors = await self._get_market_conditions(timestamp, org_id)
        factors.extend(market_factors)
        
        # Competitive landscape
        competitive_factors = await self._get_competitive_factors(timestamp, org_id)
        factors.extend(competitive_factors)
        
        # Seasonal patterns
        seasonal_factors = await self._get_seasonal_factors(timestamp)
        factors.extend(seasonal_factors)
        
        return factors
    
    async def _get_economic_indicators(self, timestamp: datetime) -> List[ExternalFactor]:
        """Get economic indicators for timestamp"""
        # In production, this would fetch from economic data APIs
        return [
            ExternalFactor(
                factor_name="consumer_confidence_index",
                factor_type="economic",
                value=85.2,  # Mock value
                confidence=0.9,
                source="economic_data_api",
                timestamp=timestamp
            ),
            ExternalFactor(
                factor_name="unemployment_rate",
                factor_type="economic",
                value=3.7,  # Mock value
                confidence=0.95,
                source="economic_data_api",
                timestamp=timestamp
            )
        ]
    
    async def _get_market_conditions(self, timestamp: datetime, org_id: str) -> List[ExternalFactor]:
        """Get market conditions for timestamp and organization"""
        return [
            ExternalFactor(
                factor_name="market_volatility",
                factor_type="market",
                value=0.23,  # Mock value
                confidence=0.8,
                source="market_data_api",
                timestamp=timestamp
            )
        ]
    
    async def _get_competitive_factors(self, timestamp: datetime, org_id: str) -> List[ExternalFactor]:
        """Get competitive factors for timestamp and organization"""
        return [
            ExternalFactor(
                factor_name="competitor_ad_spend_index",
                factor_type="competitive",
                value=1.15,  # Mock value - 15% above baseline
                confidence=0.7,
                source="competitive_intelligence",
                timestamp=timestamp
            )
        ]
    
    async def _get_seasonal_factors(self, timestamp: datetime) -> List[ExternalFactor]:
        """Get seasonal factors for timestamp"""
        return [
            ExternalFactor(
                factor_name="holiday_proximity",
                factor_type="seasonal",
                value=self._calculate_holiday_proximity(timestamp),
                confidence=1.0,
                source="calendar_analysis",
                timestamp=timestamp
            )
        ]
    
    def _calculate_holiday_proximity(self, timestamp: datetime) -> float:
        """Calculate proximity to major holidays (0-1 scale)"""
        # Simplified holiday proximity calculation
        major_holidays = [
            (1, 1),   # New Year
            (2, 14),  # Valentine's Day
            (7, 4),   # Independence Day
            (11, 24), # Thanksgiving (approximate)
            (12, 25)  # Christmas
        ]
        
        min_distance = float('inf')
        for month, day in major_holidays:
            holiday_date = date(timestamp.year, month, day)
            distance = abs((timestamp.date() - holiday_date).days)
            min_distance = min(min_distance, distance)
        
        # Convert to proximity score (closer = higher score)
        return max(0, 1 - min_distance / 30)  # 30-day window


class ConfounderDetector:
    """Detects confounding variables in marketing data"""
    
    def __init__(self, data_source: DataSource):
        self.data_source = data_source
        self.detection_rules = self._load_detection_rules()
    
    def _load_detection_rules(self) -> Dict[str, Any]:
        """Load platform-specific confounder detection rules"""
        rules = {
            DataSource.META_BUSINESS: {
                'budget_changes': {
                    'threshold': 0.2,  # 20% change
                    'importance': 0.9
                },
                'audience_overlap': {
                    'threshold': 0.3,  # 30% overlap
                    'importance': 0.8
                },
                'creative_fatigue': {
                    'frequency_threshold': 3.0,
                    'importance': 0.7
                }
            },
            DataSource.GOOGLE_ADS: {
                'quality_score_changes': {
                    'threshold': 1.0,  # 1 point change
                    'importance': 0.85
                },
                'competitor_activity': {
                    'impression_share_threshold': 0.1,
                    'importance': 0.75
                },
                'keyword_seasonality': {
                    'search_volume_threshold': 0.25,
                    'importance': 0.8
                }
            },
            DataSource.KLAVIYO: {
                'list_fatigue': {
                    'send_frequency_threshold': 5,  # emails per week
                    'importance': 0.8
                },
                'deliverability_changes': {
                    'delivery_rate_threshold': 0.05,
                    'importance': 0.9
                }
            }
        }
        return rules.get(self.data_source, {})
    
    async def detect_confounders(
        self, 
        raw_data: Dict[str, Any], 
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> ConfounderDetectionResult:
        """Detect confounding variables in the data"""
        confounders = []
        
        # Platform-specific confounder detection
        if self.data_source == DataSource.META_BUSINESS:
            confounders.extend(await self._detect_meta_confounders(raw_data, historical_data))
        elif self.data_source == DataSource.GOOGLE_ADS:
            confounders.extend(await self._detect_google_confounders(raw_data, historical_data))
        elif self.data_source == DataSource.KLAVIYO:
            confounders.extend(await self._detect_klaviyo_confounders(raw_data, historical_data))
        
        # Universal confounders
        confounders.extend(await self._detect_universal_confounders(raw_data, historical_data))
        
        # Calculate overall detection confidence
        detection_confidence = self._calculate_detection_confidence(confounders)
        
        # Generate recommendations
        recommendations = self._generate_confounder_recommendations(confounders)
        
        return ConfounderDetectionResult(
            confounders=confounders,
            detection_confidence=detection_confidence,
            detection_method=f"{self.data_source.value}_specific_detection",
            recommendations=recommendations
        )
    
    async def _detect_meta_confounders(
        self, 
        raw_data: Dict[str, Any], 
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect Meta Business specific confounders"""
        confounders = []
        
        # Budget changes
        if 'daily_budget' in raw_data and historical_data:
            budget_change = self._calculate_budget_change(raw_data, historical_data)
            if abs(budget_change) > self.detection_rules['budget_changes']['threshold']:
                confounders.append(ConfounderVariable(
                    variable_name="budget_change",
                    variable_type="continuous",
                    value=budget_change,
                    importance_score=self.detection_rules['budget_changes']['importance'],
                    detection_method="budget_change_analysis",
                    control_strategy="include_budget_change_as_covariate"
                ))
        
        # Creative fatigue
        if 'frequency' in raw_data:
            frequency = float(raw_data['frequency'])
            if frequency > self.detection_rules['creative_fatigue']['frequency_threshold']:
                confounders.append(ConfounderVariable(
                    variable_name="creative_fatigue",
                    variable_type="continuous",
                    value=frequency,
                    importance_score=self.detection_rules['creative_fatigue']['importance'],
                    detection_method="frequency_analysis",
                    control_strategy="include_frequency_as_covariate"
                ))
        
        return confounders
    
    async def _detect_google_confounders(
        self, 
        raw_data: Dict[str, Any], 
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect Google Ads specific confounders"""
        confounders = []
        
        # Quality score changes
        if 'quality_score' in raw_data and historical_data:
            quality_change = self._calculate_quality_score_change(raw_data, historical_data)
            if abs(quality_change) > self.detection_rules['quality_score_changes']['threshold']:
                confounders.append(ConfounderVariable(
                    variable_name="quality_score_change",
                    variable_type="continuous",
                    value=quality_change,
                    importance_score=self.detection_rules['quality_score_changes']['importance'],
                    detection_method="quality_score_analysis",
                    control_strategy="include_quality_score_as_covariate"
                ))
        
        return confounders
    
    async def _detect_klaviyo_confounders(
        self, 
        raw_data: Dict[str, Any], 
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect Klaviyo specific confounders"""
        confounders = []
        
        # List fatigue
        if 'send_frequency' in raw_data:
            send_frequency = float(raw_data['send_frequency'])
            if send_frequency > self.detection_rules['list_fatigue']['send_frequency_threshold']:
                confounders.append(ConfounderVariable(
                    variable_name="list_fatigue",
                    variable_type="continuous",
                    value=send_frequency,
                    importance_score=self.detection_rules['list_fatigue']['importance'],
                    detection_method="send_frequency_analysis",
                    control_strategy="include_send_frequency_as_covariate"
                ))
        
        return confounders
    
    async def _detect_universal_confounders(
        self, 
        raw_data: Dict[str, Any], 
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> List[ConfounderVariable]:
        """Detect universal confounders across all platforms"""
        confounders = []
        
        # Day of week effects
        if 'timestamp' in raw_data:
            timestamp = datetime.fromisoformat(raw_data['timestamp'])
            day_of_week = timestamp.weekday()
            confounders.append(ConfounderVariable(
                variable_name="day_of_week",
                variable_type="categorical",
                value=day_of_week,
                importance_score=0.6,
                detection_method="temporal_analysis",
                control_strategy="include_day_of_week_dummies"
            ))
        
        return confounders
    
    def _calculate_budget_change(self, raw_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> float:
        """Calculate budget change from historical data"""
        current_budget = float(raw_data.get('daily_budget', 0))
        if not historical_data:
            return 0.0
        
        historical_budgets = [float(d.get('daily_budget', 0)) for d in historical_data[-7:]]  # Last 7 days
        avg_historical_budget = np.mean(historical_budgets) if historical_budgets else current_budget
        
        if avg_historical_budget == 0:
            return 0.0
        
        return (current_budget - avg_historical_budget) / avg_historical_budget
    
    def _calculate_quality_score_change(self, raw_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> float:
        """Calculate quality score change from historical data"""
        current_quality = float(raw_data.get('quality_score', 0))
        if not historical_data:
            return 0.0
        
        historical_quality = [float(d.get('quality_score', 0)) for d in historical_data[-7:]]
        avg_historical_quality = np.mean(historical_quality) if historical_quality else current_quality
        
        return current_quality - avg_historical_quality
    
    def _calculate_detection_confidence(self, confounders: List[ConfounderVariable]) -> float:
        """Calculate overall confidence in confounder detection"""
        if not confounders:
            return 1.0  # High confidence when no confounders detected
        
        # Weight by importance scores
        weighted_confidence = np.mean([c.importance_score for c in confounders])
        return min(weighted_confidence, 1.0)
    
    def _generate_confounder_recommendations(self, confounders: List[ConfounderVariable]) -> List[str]:
        """Generate recommendations for handling detected confounders"""
        recommendations = []
        
        for confounder in confounders:
            if confounder.importance_score > 0.8:
                recommendations.append(
                    f"High-importance confounder '{confounder.variable_name}' detected. "
                    f"Recommend: {confounder.control_strategy}"
                )
            elif confounder.importance_score > 0.6:
                recommendations.append(
                    f"Medium-importance confounder '{confounder.variable_name}' detected. "
                    f"Consider: {confounder.control_strategy}"
                )
        
        if not recommendations:
            recommendations.append("No significant confounders detected. Data appears suitable for causal analysis.")
        
        return recommendations


class TreatmentAssignmentEngine:
    """Assigns treatment groups and identifies experiments"""
    
    def __init__(self):
        self.assignment_rules = self._load_assignment_rules()
    
    def _load_assignment_rules(self) -> Dict[str, Any]:
        """Load treatment assignment rules"""
        return {
            'budget_change_threshold': 0.15,  # 15% change
            'targeting_change_indicators': ['audience_change', 'location_change'],
            'creative_change_indicators': ['creative_id_change', 'ad_copy_change'],
            'bid_strategy_changes': ['bid_strategy_change']
        }
    
    async def assign_treatment(
        self, 
        raw_data: Dict[str, Any], 
        org_id: str,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> TreatmentAssignmentResult:
        """Assign treatment group and identify experiment"""
        
        # Detect treatment type
        treatment_type = await self._detect_treatment_type(raw_data, historical_data)
        
        # Assign treatment group
        treatment_group = await self._assign_treatment_group(raw_data, treatment_type)
        
        # Calculate treatment intensity
        treatment_intensity = await self._calculate_treatment_intensity(raw_data, treatment_type, historical_data)
        
        # Determine randomization unit
        randomization_unit = await self._determine_randomization_unit(raw_data, treatment_type)
        
        # Check for existing experiment
        experiment_id = await self._identify_experiment(raw_data, org_id, treatment_type)
        
        # Calculate assignment confidence
        assignment_confidence = await self._calculate_assignment_confidence(
            raw_data, treatment_type, historical_data
        )
        
        return TreatmentAssignmentResult(
            treatment_group=treatment_group,
            treatment_type=treatment_type,
            treatment_intensity=treatment_intensity,
            randomization_unit=randomization_unit,
            experiment_id=experiment_id,
            assignment_confidence=assignment_confidence
        )
    
    async def _detect_treatment_type(
        self, 
        raw_data: Dict[str, Any], 
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> TreatmentType:
        """Detect the type of treatment applied"""
        
        # Check for budget changes
        if historical_data:
            budget_change = self._calculate_budget_change_ratio(raw_data, historical_data)
            if budget_change > self.assignment_rules['budget_change_threshold']:
                return TreatmentType.BUDGET_INCREASE
            elif budget_change < -self.assignment_rules['budget_change_threshold']:
                return TreatmentType.BUDGET_DECREASE
        
        # Check for targeting changes
        if any(indicator in raw_data for indicator in self.assignment_rules['targeting_change_indicators']):
            return TreatmentType.TARGETING_CHANGE
        
        # Check for creative changes
        if any(indicator in raw_data for indicator in self.assignment_rules['creative_change_indicators']):
            return TreatmentType.CREATIVE_CHANGE
        
        # Check for bid strategy changes
        if any(indicator in raw_data for indicator in self.assignment_rules['bid_strategy_changes']):
            return TreatmentType.BID_STRATEGY_CHANGE
        
        # Check campaign status
        if raw_data.get('campaign_status') == 'ACTIVE' and not historical_data:
            return TreatmentType.CAMPAIGN_LAUNCH
        elif raw_data.get('campaign_status') == 'PAUSED':
            return TreatmentType.CAMPAIGN_PAUSE
        
        return TreatmentType.CONTROL
    
    async def _assign_treatment_group(self, raw_data: Dict[str, Any], treatment_type: TreatmentType) -> str:
        """Assign treatment group based on treatment type"""
        if treatment_type == TreatmentType.CONTROL:
            return "control"
        else:
            return f"treatment_{treatment_type.value}"
    
    async def _calculate_treatment_intensity(
        self, 
        raw_data: Dict[str, Any], 
        treatment_type: TreatmentType,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Calculate treatment intensity (0-1 scale)"""
        if treatment_type == TreatmentType.CONTROL:
            return 0.0
        
        if treatment_type in [TreatmentType.BUDGET_INCREASE, TreatmentType.BUDGET_DECREASE]:
            if historical_data:
                budget_change = abs(self._calculate_budget_change_ratio(raw_data, historical_data))
                return min(budget_change, 1.0)
        
        # Default intensity for other treatment types
        return 0.5
    
    async def _determine_randomization_unit(
        self, 
        raw_data: Dict[str, Any], 
        treatment_type: TreatmentType
    ) -> RandomizationUnit:
        """Determine the unit of randomization"""
        if 'ad_id' in raw_data:
            return RandomizationUnit.AD_SET
        elif 'campaign_id' in raw_data:
            return RandomizationUnit.CAMPAIGN
        else:
            return RandomizationUnit.CAMPAIGN
    
    async def _identify_experiment(
        self, 
        raw_data: Dict[str, Any], 
        org_id: str, 
        treatment_type: TreatmentType
    ) -> Optional[str]:
        """Identify if this data belongs to an existing experiment"""
        # In production, this would query the experiment database
        # For now, return None (no experiment identified)
        return None
    
    async def _calculate_assignment_confidence(
        self, 
        raw_data: Dict[str, Any], 
        treatment_type: TreatmentType,
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Calculate confidence in treatment assignment"""
        if treatment_type == TreatmentType.CONTROL:
            return 1.0
        
        # Base confidence on data availability and treatment clarity
        confidence = 0.7  # Base confidence
        
        if historical_data and len(historical_data) >= 7:
            confidence += 0.2  # Boost for sufficient historical data
        
        if treatment_type in [TreatmentType.BUDGET_INCREASE, TreatmentType.BUDGET_DECREASE]:
            confidence += 0.1  # Boost for clear budget changes
        
        return min(confidence, 1.0)
    
    def _calculate_budget_change_ratio(self, raw_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> float:
        """Calculate budget change ratio"""
        current_budget = float(raw_data.get('daily_budget', 0))
        if not historical_data:
            return 0.0
        
        historical_budgets = [float(d.get('daily_budget', 0)) for d in historical_data[-7:]]
        avg_historical_budget = np.mean(historical_budgets) if historical_budgets else current_budget
        
        if avg_historical_budget == 0:
            return 0.0
        
        return (current_budget - avg_historical_budget) / avg_historical_budget


class CausalDataQualityAssessor:
    """Assesses data quality for causal inference"""
    
    def __init__(self):
        self.quality_thresholds = {
            'temporal_consistency': 0.95,
            'confounder_coverage': 0.8,
            'treatment_assignment': 0.9,
            'outcome_measurement': 0.85,
            'external_validity': 0.7,
            'missing_data': 0.9,
            'anomaly_detection': 0.8
        }
    
    async def assess_data_quality(
        self, 
        causal_data: CausalMarketingData,
        historical_data: Optional[List[CausalMarketingData]] = None
    ) -> DataQualityAssessment:
        """Assess overall data quality for causal inference"""
        
        # Individual quality scores
        temporal_score = await self._assess_temporal_consistency(causal_data, historical_data)
        confounder_score = await self._assess_confounder_coverage(causal_data)
        treatment_score = await self._assess_treatment_assignment(causal_data)
        outcome_score = await self._assess_outcome_measurement(causal_data)
        external_score = await self._assess_external_validity(causal_data)
        missing_score = await self._assess_missing_data(causal_data)
        anomaly_score = await self._assess_anomaly_detection(causal_data, historical_data)
        
        # Calculate overall score
        scores = {
            'temporal_consistency': temporal_score,
            'confounder_coverage': confounder_score,
            'treatment_assignment_quality': treatment_score,
            'outcome_measurement_quality': outcome_score,
            'external_validity': external_score,
            'missing_data_score': missing_score,
            'anomaly_detection_score': anomaly_score
        }
        
        # Weighted average
        weights = {
            'temporal_consistency': 0.2,
            'confounder_coverage': 0.25,
            'treatment_assignment_quality': 0.2,
            'outcome_measurement_quality': 0.15,
            'external_validity': 0.1,
            'missing_data_score': 0.05,
            'anomaly_detection_score': 0.05
        }
        
        overall_score = sum(scores[metric] * weights[metric] for metric in scores)
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(scores)
        
        return DataQualityAssessment(
            overall_score=overall_score,
            **scores,
            recommendations=recommendations
        )
    
    async def _assess_temporal_consistency(
        self, 
        causal_data: CausalMarketingData,
        historical_data: Optional[List[CausalMarketingData]]
    ) -> float:
        """Assess temporal consistency of the data"""
        # Check if timestamp is reasonable
        now = datetime.utcnow()
        time_diff = abs((now - causal_data.timestamp).total_seconds())
        
        # Data should be recent (within 7 days for real-time analysis)
        if time_diff > 7 * 24 * 3600:  # 7 days
            return 0.5
        
        # Check temporal ordering if historical data available
        if historical_data:
            for hist_data in historical_data:
                if hist_data.timestamp >= causal_data.timestamp:
                    return 0.3  # Temporal ordering violation
        
        return 1.0
    
    async def _assess_confounder_coverage(self, causal_data: CausalMarketingData) -> float:
        """Assess coverage of important confounders"""
        important_confounders = ['budget_change', 'seasonality', 'day_of_week', 'competitor_activity']
        detected_confounders = [c.variable_name for c in causal_data.confounders]
        
        coverage = len(set(detected_confounders) & set(important_confounders)) / len(important_confounders)
        return coverage
    
    async def _assess_treatment_assignment(self, causal_data: CausalMarketingData) -> float:
        """Assess quality of treatment assignment"""
        if causal_data.treatment_group == "control":
            return 1.0
        
        # Check if treatment is well-defined
        if causal_data.treatment_intensity > 0 and causal_data.treatment_type != TreatmentType.CONTROL:
            return 0.9
        
        return 0.5
    
    async def _assess_outcome_measurement(self, causal_data: CausalMarketingData) -> float:
        """Assess quality of outcome measurement"""
        # Check if key metrics are present and reasonable
        if causal_data.conversions < 0 or causal_data.revenue < 0:
            return 0.0
        
        if causal_data.clicks > causal_data.impressions:
            return 0.3  # Impossible scenario
        
        if causal_data.conversions > causal_data.clicks:
            return 0.3  # Impossible scenario
        
        return 1.0
    
    async def _assess_external_validity(self, causal_data: CausalMarketingData) -> float:
        """Assess external validity of the data"""
        # Check if external factors are captured
        if len(causal_data.external_factors) >= 3:
            return 1.0
        elif len(causal_data.external_factors) >= 1:
            return 0.7
        else:
            return 0.3
    
    async def _assess_missing_data(self, causal_data: CausalMarketingData) -> float:
        """Assess missing data quality"""
        required_fields = ['spend', 'impressions', 'clicks', 'conversions']
        missing_count = sum(1 for field in required_fields if getattr(causal_data, field, None) is None)
        
        return 1.0 - (missing_count / len(required_fields))
    
    async def _assess_anomaly_detection(
        self, 
        causal_data: CausalMarketingData,
        historical_data: Optional[List[CausalMarketingData]]
    ) -> float:
        """Assess anomaly detection score"""
        if not historical_data:
            return 0.8  # Default score when no historical data
        
        # Simple anomaly detection based on spend patterns
        current_spend = causal_data.spend
        historical_spends = [d.spend for d in historical_data[-30:]]  # Last 30 records
        
        if not historical_spends:
            return 0.8
        
        mean_spend = np.mean(historical_spends)
        std_spend = np.std(historical_spends)
        
        if std_spend == 0:
            return 1.0 if current_spend == mean_spend else 0.5
        
        z_score = abs((current_spend - mean_spend) / std_spend)
        
        # Higher z-score indicates more anomalous data
        if z_score > 3:
            return 0.2
        elif z_score > 2:
            return 0.5
        else:
            return 1.0
    
    def _generate_quality_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality scores"""
        recommendations = []
        
        for metric, score in scores.items():
            threshold = self.quality_thresholds.get(metric, 0.8)
            if score < threshold:
                if metric == 'temporal_consistency':
                    recommendations.append("Improve temporal consistency by ensuring proper timestamp ordering")
                elif metric == 'confounder_coverage':
                    recommendations.append("Increase confounder detection to capture more potential confounding variables")
                elif metric == 'treatment_assignment_quality':
                    recommendations.append("Improve treatment assignment clarity and documentation")
                elif metric == 'outcome_measurement_quality':
                    recommendations.append("Validate outcome measurements for consistency and accuracy")
                elif metric == 'external_validity':
                    recommendations.append("Capture more external factors that may influence outcomes")
                elif metric == 'missing_data_score':
                    recommendations.append("Reduce missing data in key marketing metrics")
                elif metric == 'anomaly_detection_score':
                    recommendations.append("Investigate potential anomalies in the data")
        
        if not recommendations:
            recommendations.append("Data quality is excellent for causal inference")
        
        return recommendations


class CausalDataTransformer:
    """Main transformer for converting raw marketing data to causal format"""
    
    def __init__(self):
        self.external_factor_provider = ExternalFactorProvider()
        self.quality_assessor = CausalDataQualityAssessor()
        self.confounder_detectors = {
            DataSource.META_BUSINESS: ConfounderDetector(DataSource.META_BUSINESS),
            DataSource.GOOGLE_ADS: ConfounderDetector(DataSource.GOOGLE_ADS),
            DataSource.KLAVIYO: ConfounderDetector(DataSource.KLAVIYO)
        }
        self.treatment_assignment_engine = TreatmentAssignmentEngine()
    
    async def transform_to_causal_format(
        self,
        raw_data: Dict[str, Any],
        data_source: DataSource,
        org_id: str,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> CausalMarketingData:
        """Transform raw marketing data to causal format"""
        
        # 1. Detect confounders
        confounder_detector = self.confounder_detectors[data_source]
        confounder_result = await confounder_detector.detect_confounders(raw_data, historical_data)
        
        # 2. Get external factors
        timestamp = datetime.fromisoformat(raw_data['timestamp'])
        external_factors = await self.external_factor_provider.get_external_factors(timestamp, org_id)
        
        # 3. Assign treatment
        treatment_result = await self.treatment_assignment_engine.assign_treatment(
            raw_data, org_id, historical_data
        )
        
        # 4. Create calendar features
        calendar_features = await self._create_calendar_features(timestamp)
        
        # 5. Calculate lag features
        lag_features = await self._calculate_lag_features(raw_data, historical_data)
        
        # 6. Create causal marketing data
        causal_data = CausalMarketingData(
            id=f"{org_id}_{data_source.value}_{raw_data.get('id', timestamp.isoformat())}",
            org_id=org_id,
            timestamp=timestamp,
            data_source=data_source,
            campaign_id=raw_data.get('campaign_id', ''),
            campaign_name=raw_data.get('campaign_name', ''),
            campaign_objective=self._map_campaign_objective(raw_data.get('objective')),
            campaign_status=self._map_campaign_status(raw_data.get('status')),
            ad_set_id=raw_data.get('ad_set_id'),
            ad_set_name=raw_data.get('ad_set_name'),
            ad_id=raw_data.get('ad_id'),
            ad_name=raw_data.get('ad_name'),
            treatment_group=treatment_result.treatment_group,
            treatment_type=treatment_result.treatment_type,
            treatment_intensity=treatment_result.treatment_intensity,
            randomization_unit=treatment_result.randomization_unit,
            experiment_id=treatment_result.experiment_id,
            spend=float(raw_data.get('spend', 0)),
            impressions=int(raw_data.get('impressions', 0)),
            clicks=int(raw_data.get('clicks', 0)),
            conversions=int(raw_data.get('conversions', 0)),
            revenue=float(raw_data.get('revenue', 0)),
            platform_metrics=self._extract_platform_metrics(raw_data, data_source),
            confounders=confounder_result.confounders,
            external_factors=external_factors,
            calendar_features=calendar_features,
            lag_features=lag_features,
            geographic_data=raw_data.get('geographic_data', {}),
            audience_data=raw_data.get('audience_data', {}),
            causal_metadata={
                'transformation_version': '1.0',
                'data_source': data_source.value,
                'confounder_detection_confidence': confounder_result.detection_confidence,
                'treatment_assignment_confidence': treatment_result.assignment_confidence
            }
        )
        
        # 7. Assess data quality
        quality_assessment = await self.quality_assessor.assess_data_quality(causal_data)
        causal_data.data_quality_score = quality_assessment.overall_score
        
        # 8. Set validation flags
        causal_data.validation_flags = {
            'temporal_consistency': quality_assessment.temporal_consistency > 0.9,
            'confounder_coverage': quality_assessment.confounder_coverage > 0.8,
            'treatment_assignment': quality_assessment.treatment_assignment_quality > 0.8,
            'outcome_measurement': quality_assessment.outcome_measurement_quality > 0.8
        }
        
        # 9. Calculate anomaly score
        causal_data.anomaly_score = 1.0 - quality_assessment.anomaly_detection_score
        
        return causal_data
    
    async def _create_calendar_features(self, timestamp: datetime) -> CalendarDimension:
        """Create calendar dimension features"""
        date_obj = timestamp.date()
        
        # Basic calendar features
        calendar_features = CalendarDimension(
            date=date_obj,
            year=date_obj.year,
            quarter=(date_obj.month - 1) // 3 + 1,
            month=date_obj.month,
            week=date_obj.isocalendar()[1],
            day_of_year=date_obj.timetuple().tm_yday,
            day_of_month=date_obj.day,
            day_of_week=date_obj.weekday(),
            is_weekend=date_obj.weekday() >= 5,
            season=self._get_season(date_obj),
            fiscal_year=date_obj.year if date_obj.month >= 4 else date_obj.year - 1,
            fiscal_quarter=((date_obj.month - 4) % 12) // 3 + 1 if date_obj.month >= 4 else ((date_obj.month + 8) % 12) // 3 + 1
        )
        
        # Add holiday information
        calendar_features.is_holiday, calendar_features.holiday_name = self._check_holiday(date_obj)
        
        # Add economic indicators (mock data)
        calendar_features.economic_indicators = {
            'consumer_confidence': 85.2,
            'unemployment_rate': 3.7,
            'inflation_rate': 2.1
        }
        
        # Add market conditions (mock data)
        calendar_features.market_conditions = {
            'market_volatility': 0.23,
            'sector_performance': 1.05
        }
        
        return calendar_features
    
    async def _calculate_lag_features(
        self,
        raw_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Calculate lagged features for causal analysis"""
        lag_features = {}
        
        if not historical_data:
            return lag_features
        
        # Calculate 7-day and 30-day lags for key metrics
        metrics = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
        
        for metric in metrics:
            # 7-day lag
            lag_7_values = [float(d.get(metric, 0)) for d in historical_data[-7:]]
            if lag_7_values:
                lag_features[f'{metric}_lag_7_avg'] = np.mean(lag_7_values)
                lag_features[f'{metric}_lag_7_trend'] = self._calculate_trend(lag_7_values)
            
            # 30-day lag
            lag_30_values = [float(d.get(metric, 0)) for d in historical_data[-30:]]
            if lag_30_values:
                lag_features[f'{metric}_lag_30_avg'] = np.mean(lag_30_values)
                lag_features[f'{metric}_lag_30_trend'] = self._calculate_trend(lag_30_values)
        
        return lag_features
    
    def _extract_platform_metrics(self, raw_data: Dict[str, Any], data_source: DataSource) -> Dict[str, Any]:
        """Extract platform-specific metrics"""
        platform_metrics = {}
        
        if data_source == DataSource.META_BUSINESS:
            platform_metrics.update({
                'cpm': raw_data.get('cpm'),
                'cpc': raw_data.get('cpc'),
                'ctr': raw_data.get('ctr'),
                'frequency': raw_data.get('frequency'),
                'reach': raw_data.get('reach'),
                'video_views': raw_data.get('video_views')
            })
        elif data_source == DataSource.GOOGLE_ADS:
            platform_metrics.update({
                'quality_score': raw_data.get('quality_score'),
                'search_impression_share': raw_data.get('search_impression_share'),
                'cost_per_conversion': raw_data.get('cost_per_conversion'),
                'conversion_rate': raw_data.get('conversion_rate')
            })
        elif data_source == DataSource.KLAVIYO:
            platform_metrics.update({
                'delivered': raw_data.get('delivered'),
                'opened': raw_data.get('opened'),
                'clicked': raw_data.get('clicked'),
                'unsubscribed': raw_data.get('unsubscribed'),
                'open_rate': raw_data.get('open_rate'),
                'click_rate': raw_data.get('click_rate')
            })
        
        # Remove None values
        return {k: v for k, v in platform_metrics.items() if v is not None}
    
    def _map_campaign_objective(self, objective: Optional[str]) -> CampaignObjective:
        """Map platform-specific objectives to standard objectives"""
        if not objective:
            return CampaignObjective.CONVERSIONS
        
        objective_mapping = {
            'CONVERSIONS': CampaignObjective.CONVERSIONS,
            'TRAFFIC': CampaignObjective.TRAFFIC,
            'AWARENESS': CampaignObjective.AWARENESS,
            'ENGAGEMENT': CampaignObjective.ENGAGEMENT,
            'LEADS': CampaignObjective.LEADS,
            'SALES': CampaignObjective.SALES,
            'APP_INSTALLS': CampaignObjective.APP_INSTALLS,
            'VIDEO_VIEWS': CampaignObjective.VIDEO_VIEWS
        }
        
        return objective_mapping.get(objective.upper(), CampaignObjective.CONVERSIONS)
    
    def _map_campaign_status(self, status: Optional[str]) -> AdStatus:
        """Map platform-specific status to standard status"""
        if not status:
            return AdStatus.ACTIVE
        
        status_mapping = {
            'ACTIVE': AdStatus.ACTIVE,
            'PAUSED': AdStatus.PAUSED,
            'DELETED': AdStatus.DELETED,
            'PENDING': AdStatus.PENDING,
            'DISAPPROVED': AdStatus.DISAPPROVED,
            'LEARNING': AdStatus.LEARNING
        }
        
        return status_mapping.get(status.upper(), AdStatus.ACTIVE)
    
    def _get_season(self, date_obj: date) -> str:
        """Get season for a given date"""
        month = date_obj.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _check_holiday(self, date_obj: date) -> Tuple[bool, Optional[str]]:
        """Check if date is a holiday"""
        # Simplified holiday checking
        holidays = {
            (1, 1): "New Year's Day",
            (2, 14): "Valentine's Day",
            (7, 4): "Independence Day",
            (12, 25): "Christmas Day"
        }
        
        holiday_name = holidays.get((date_obj.month, date_obj.day))
        return holiday_name is not None, holiday_name
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.std(y) == 0:
            return 0.0
        
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0