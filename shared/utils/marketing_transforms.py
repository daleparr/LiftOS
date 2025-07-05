"""
Marketing Data Transformation Utilities
Handles pandas operations and data normalization for marketing data
"""
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Union
import json
import logging

from ..models.marketing import (
    DataSource, MarketingDataEntry, CampaignData, AdSetData, AdData,
    MetaBusinessData, GoogleAdsData, KlaviyoData, CalendarDimension,
    PandasTransformationConfig, DataTransformationRule
)


logger = logging.getLogger(__name__)


class MarketingDataTransformer:
    """Handles transformation of raw marketing data into normalized formats"""
    
    def __init__(self):
        self.transformation_configs = self._load_default_configs()
    
    def _load_default_configs(self) -> Dict[DataSource, PandasTransformationConfig]:
        """Load default transformation configurations for each data source"""
        configs = {}
        
        # Meta Business transformation config
        meta_rules = [
            DataTransformationRule(
                source_field="spend",
                target_field="spend",
                transformation_type="normalize",
                transformation_logic={"type": "float", "default": 0.0}
            ),
            DataTransformationRule(
                source_field="impressions",
                target_field="impressions",
                transformation_type="normalize",
                transformation_logic={"type": "int", "default": 0}
            ),
            DataTransformationRule(
                source_field="clicks",
                target_field="clicks",
                transformation_type="normalize",
                transformation_logic={"type": "int", "default": 0}
            ),
            DataTransformationRule(
                source_field="actions",
                target_field="conversions",
                transformation_type="calculate",
                transformation_logic={
                    "formula": "sum(actions[action_type='purchase'])",
                    "default": 0
                }
            )
        ]
        
        configs[DataSource.META_BUSINESS] = PandasTransformationConfig(
            data_source=DataSource.META_BUSINESS,
            transformation_rules=meta_rules,
            calendar_join=True,
            output_format="normalized"
        )
        
        # Google Ads transformation config
        google_rules = [
            DataTransformationRule(
                source_field="cost_micros",
                target_field="spend",
                transformation_type="calculate",
                transformation_logic={"formula": "cost_micros / 1000000", "default": 0.0}
            ),
            DataTransformationRule(
                source_field="impressions",
                target_field="impressions",
                transformation_type="normalize",
                transformation_logic={"type": "int", "default": 0}
            ),
            DataTransformationRule(
                source_field="clicks",
                target_field="clicks",
                transformation_type="normalize",
                transformation_logic={"type": "int", "default": 0}
            ),
            DataTransformationRule(
                source_field="conversions",
                target_field="conversions",
                transformation_type="normalize",
                transformation_logic={"type": "int", "default": 0}
            )
        ]
        
        configs[DataSource.GOOGLE_ADS] = PandasTransformationConfig(
            data_source=DataSource.GOOGLE_ADS,
            transformation_rules=google_rules,
            calendar_join=True,
            output_format="normalized"
        )
        
        # Klaviyo transformation config
        klaviyo_rules = [
            DataTransformationRule(
                source_field="delivered",
                target_field="impressions",
                transformation_type="map",
                transformation_logic={"mapping": "delivered -> impressions"}
            ),
            DataTransformationRule(
                source_field="clicked",
                target_field="clicks",
                transformation_type="normalize",
                transformation_logic={"type": "int", "default": 0}
            ),
            DataTransformationRule(
                source_field="revenue",
                target_field="revenue",
                transformation_type="normalize",
                transformation_logic={"type": "float", "default": 0.0}
            )
        ]
        
        configs[DataSource.KLAVIYO] = PandasTransformationConfig(
            data_source=DataSource.KLAVIYO,
            transformation_rules=klaviyo_rules,
            calendar_join=True,
            output_format="normalized"
        )
        
        return configs
    
    def transform_raw_data(
        self, 
        raw_data: List[Dict[str, Any]], 
        data_source: DataSource,
        date_range_start: date,
        date_range_end: date,
        org_id: str
    ) -> pd.DataFrame:
        """Transform raw marketing data into normalized pandas DataFrame"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(raw_data)
            
            if df.empty:
                return pd.DataFrame()
            
            # Get transformation config
            config = self.transformation_configs.get(data_source)
            if not config:
                logger.warning(f"No transformation config found for {data_source}")
                return df
            
            # Apply transformation rules
            df_transformed = self._apply_transformation_rules(df, config.transformation_rules)
            
            # Add standard fields
            df_transformed['data_source'] = data_source.value
            df_transformed['org_id'] = org_id
            df_transformed['date_range_start'] = date_range_start
            df_transformed['date_range_end'] = date_range_end
            df_transformed['ingestion_timestamp'] = datetime.utcnow()
            
            # Join with calendar dimensions if enabled
            if config.calendar_join:
                df_transformed = self._join_calendar_dimensions(
                    df_transformed, 
                    date_range_start, 
                    date_range_end
                )
            
            # Calculate derived metrics
            df_transformed = self._calculate_derived_metrics(df_transformed)
            
            logger.info(f"Transformed {len(df_transformed)} records for {data_source}")
            return df_transformed
            
        except Exception as e:
            logger.error(f"Data transformation failed for {data_source}: {str(e)}")
            raise
    
    def _apply_transformation_rules(
        self, 
        df: pd.DataFrame, 
        rules: List[DataTransformationRule]
    ) -> pd.DataFrame:
        """Apply transformation rules to DataFrame"""
        df_result = df.copy()
        
        for rule in rules:
            try:
                if rule.transformation_type == "normalize":
                    df_result = self._normalize_field(df_result, rule)
                elif rule.transformation_type == "calculate":
                    df_result = self._calculate_field(df_result, rule)
                elif rule.transformation_type == "map":
                    df_result = self._map_field(df_result, rule)
                elif rule.transformation_type == "aggregate":
                    df_result = self._aggregate_field(df_result, rule)
                    
            except Exception as e:
                logger.warning(f"Failed to apply rule {rule.source_field} -> {rule.target_field}: {str(e)}")
                continue
        
        return df_result
    
    def _normalize_field(self, df: pd.DataFrame, rule: DataTransformationRule) -> pd.DataFrame:
        """Normalize field based on type and default value"""
        logic = rule.transformation_logic
        field_type = logic.get("type", "str")
        default_value = logic.get("default", None)
        
        if rule.source_field in df.columns:
            if field_type == "float":
                df[rule.target_field] = pd.to_numeric(df[rule.source_field], errors='coerce').fillna(default_value)
            elif field_type == "int":
                df[rule.target_field] = pd.to_numeric(df[rule.source_field], errors='coerce').fillna(default_value).astype(int)
            elif field_type == "str":
                df[rule.target_field] = df[rule.source_field].astype(str).fillna(str(default_value))
            elif field_type == "datetime":
                df[rule.target_field] = pd.to_datetime(df[rule.source_field], errors='coerce')
        else:
            # Create field with default value if source doesn't exist
            df[rule.target_field] = default_value
        
        return df
    
    def _calculate_field(self, df: pd.DataFrame, rule: DataTransformationRule) -> pd.DataFrame:
        """Calculate field using formula"""
        logic = rule.transformation_logic
        formula = logic.get("formula", "")
        default_value = logic.get("default", 0)
        
        try:
            if "cost_micros / 1000000" in formula and "cost_micros" in df.columns:
                df[rule.target_field] = df["cost_micros"] / 1000000
            elif "sum(actions" in formula and "actions" in df.columns:
                # Handle Meta actions array
                df[rule.target_field] = df["actions"].apply(
                    lambda x: self._extract_action_value(x, "purchase") if x else 0
                )
            else:
                df[rule.target_field] = default_value
        except Exception as e:
            logger.warning(f"Formula calculation failed: {str(e)}")
            df[rule.target_field] = default_value
        
        return df
    
    def _map_field(self, df: pd.DataFrame, rule: DataTransformationRule) -> pd.DataFrame:
        """Map field values"""
        if rule.source_field in df.columns:
            df[rule.target_field] = df[rule.source_field]
        
        return df
    
    def _aggregate_field(self, df: pd.DataFrame, rule: DataTransformationRule) -> pd.DataFrame:
        """Aggregate field values"""
        # Implementation for aggregation rules
        return df
    
    def _extract_action_value(self, actions: Any, action_type: str) -> int:
        """Extract value from Meta actions array"""
        try:
            if isinstance(actions, list):
                for action in actions:
                    if isinstance(action, dict) and action.get("action_type") == action_type:
                        return int(action.get("value", 0))
            elif isinstance(actions, dict):
                return int(actions.get("value", 0))
            return 0
        except:
            return 0
    
    def _join_calendar_dimensions(
        self, 
        df: pd.DataFrame, 
        date_start: date, 
        date_end: date
    ) -> pd.DataFrame:
        """Join with calendar dimensions for causal modeling"""
        try:
            # Generate calendar dimensions
            calendar_data = []
            current_date = date_start
            
            while current_date <= date_end:
                day_of_year = current_date.timetuple().tm_yday
                week_number = current_date.isocalendar()[1]
                quarter = (current_date.month - 1) // 3 + 1
                
                # Determine season
                if current_date.month in [12, 1, 2]:
                    season = "winter"
                elif current_date.month in [3, 4, 5]:
                    season = "spring"
                elif current_date.month in [6, 7, 8]:
                    season = "summer"
                else:
                    season = "fall"
                
                calendar_data.append({
                    'date': current_date,
                    'year': current_date.year,
                    'quarter': quarter,
                    'month': current_date.month,
                    'week': week_number,
                    'day_of_year': day_of_year,
                    'day_of_month': current_date.day,
                    'day_of_week': current_date.weekday(),
                    'is_weekend': current_date.weekday() >= 5,
                    'season': season
                })
                
                current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
                current_date = current_date.date()
            
            calendar_df = pd.DataFrame(calendar_data)
            
            # Add date column to main df if not exists
            if 'date' not in df.columns:
                df['date'] = date_start  # Default to start date
            
            # Convert date columns to datetime for joining
            df['date'] = pd.to_datetime(df['date'])
            calendar_df['date'] = pd.to_datetime(calendar_df['date'])
            
            # Join with calendar dimensions
            df_joined = df.merge(calendar_df, on='date', how='left', suffixes=('', '_cal'))
            
            return df_joined
            
        except Exception as e:
            logger.warning(f"Calendar join failed: {str(e)}")
            return df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived marketing metrics"""
        try:
            # Ensure required columns exist with defaults
            required_cols = ['spend', 'impressions', 'clicks', 'conversions', 'revenue']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0
            
            # Calculate CPM (Cost Per Mille)
            df['cpm'] = np.where(
                df['impressions'] > 0,
                (df['spend'] / df['impressions']) * 1000,
                0
            )
            
            # Calculate CPC (Cost Per Click)
            df['cpc'] = np.where(
                df['clicks'] > 0,
                df['spend'] / df['clicks'],
                0
            )
            
            # Calculate CTR (Click Through Rate)
            df['ctr'] = np.where(
                df['impressions'] > 0,
                (df['clicks'] / df['impressions']) * 100,
                0
            )
            
            # Calculate Conversion Rate
            df['conversion_rate'] = np.where(
                df['clicks'] > 0,
                (df['conversions'] / df['clicks']) * 100,
                0
            )
            
            # Calculate ROAS (Return on Ad Spend)
            df['roas'] = np.where(
                df['spend'] > 0,
                df['revenue'] / df['spend'],
                0
            )
            
            # Calculate Cost Per Conversion
            df['cost_per_conversion'] = np.where(
                df['conversions'] > 0,
                df['spend'] / df['conversions'],
                0
            )
            
            return df
            
        except Exception as e:
            logger.warning(f"Derived metrics calculation failed: {str(e)}")
            return df
    
    def aggregate_data(
        self, 
        df: pd.DataFrame, 
        group_by: List[str], 
        metrics: List[str]
    ) -> pd.DataFrame:
        """Aggregate marketing data by specified dimensions"""
        try:
            # Ensure group_by columns exist
            valid_group_by = [col for col in group_by if col in df.columns]
            
            if not valid_group_by:
                logger.warning("No valid group_by columns found")
                return df
            
            # Ensure metrics columns exist
            valid_metrics = [col for col in metrics if col in df.columns]
            
            if not valid_metrics:
                logger.warning("No valid metrics columns found")
                return df
            
            # Perform aggregation
            agg_dict = {}
            for metric in valid_metrics:
                if metric in ['spend', 'revenue', 'cpm', 'cpc', 'cost_per_conversion']:
                    agg_dict[metric] = 'sum'
                elif metric in ['impressions', 'clicks', 'conversions']:
                    agg_dict[metric] = 'sum'
                elif metric in ['ctr', 'conversion_rate', 'roas']:
                    agg_dict[metric] = 'mean'
                else:
                    agg_dict[metric] = 'sum'
            
            df_agg = df.groupby(valid_group_by).agg(agg_dict).reset_index()
            
            # Recalculate derived metrics after aggregation
            df_agg = self._calculate_derived_metrics(df_agg)
            
            return df_agg
            
        except Exception as e:
            logger.error(f"Data aggregation failed: {str(e)}")
            return df
    
    def export_for_causal_analysis(
        self, 
        df: pd.DataFrame, 
        target_metric: str = "conversions"
    ) -> Dict[str, Any]:
        """Export data in format suitable for causal analysis"""
        try:
            # Prepare data for causal modeling
            causal_data = {
                "target_variable": target_metric,
                "features": [],
                "time_series": [],
                "calendar_features": [],
                "marketing_features": []
            }
            
            # Identify feature columns
            marketing_cols = ['spend', 'impressions', 'clicks', 'cpm', 'cpc', 'ctr']
            calendar_cols = ['year', 'quarter', 'month', 'week', 'day_of_week', 'is_weekend', 'season']
            
            causal_data["marketing_features"] = [col for col in marketing_cols if col in df.columns]
            causal_data["calendar_features"] = [col for col in calendar_cols if col in df.columns]
            causal_data["features"] = causal_data["marketing_features"] + causal_data["calendar_features"]
            
            # Prepare time series data
            if 'date' in df.columns:
                df_sorted = df.sort_values('date')
                causal_data["time_series"] = df_sorted.to_dict('records')
            else:
                causal_data["time_series"] = df.to_dict('records')
            
            # Add correlation matrix
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = df[numeric_cols].corr()
                causal_data["correlation_matrix"] = correlation_matrix.to_dict()
            
            return causal_data
            
        except Exception as e:
            logger.error(f"Causal analysis export failed: {str(e)}")
            return {"error": str(e)}


# Utility functions
def create_marketing_data_entry(
    raw_data: Dict[str, Any],
    data_source: DataSource,
    org_id: str,
    date_range_start: date,
    date_range_end: date
) -> MarketingDataEntry:
    """Create appropriate marketing data entry based on source"""
    
    entry_id = str(pd.Timestamp.now().timestamp()).replace('.', '')
    
    if data_source == DataSource.META_BUSINESS:
        return MetaBusinessData(
            id=entry_id,
            org_id=org_id,
            source_id=raw_data.get("id", entry_id),
            raw_data=raw_data,
            processed_data=raw_data,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            account_id=raw_data.get("account_id", ""),
            campaign_id=raw_data.get("campaign_id"),
            adset_id=raw_data.get("adset_id"),
            ad_id=raw_data.get("ad_id")
        )
    elif data_source == DataSource.GOOGLE_ADS:
        return GoogleAdsData(
            id=entry_id,
            org_id=org_id,
            source_id=raw_data.get("id", entry_id),
            raw_data=raw_data,
            processed_data=raw_data,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            customer_id=raw_data.get("customer_id", ""),
            campaign_id=raw_data.get("campaign_id"),
            ad_group_id=raw_data.get("ad_group_id"),
            ad_id=raw_data.get("ad_id")
        )
    elif data_source == DataSource.KLAVIYO:
        return KlaviyoData(
            id=entry_id,
            org_id=org_id,
            source_id=raw_data.get("id", entry_id),
            raw_data=raw_data,
            processed_data=raw_data,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            list_id=raw_data.get("list_id"),
            segment_id=raw_data.get("segment_id"),
            campaign_id=raw_data.get("campaign_id"),
            flow_id=raw_data.get("flow_id")
        )
    else:
        return MarketingDataEntry(
            id=entry_id,
            org_id=org_id,
            data_source=data_source,
            source_id=raw_data.get("id", entry_id),
            raw_data=raw_data,
            processed_data=raw_data,
            date_range_start=date_range_start,
            date_range_end=date_range_end
        )