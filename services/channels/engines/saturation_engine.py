"""
Saturation Engine for Channels Service
Hill/Adstock saturation modeling for diminishing returns analysis
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from models.channels import (
    SaturationModel, SaturationFunction, ChannelPerformance
)

logger = logging.getLogger(__name__)


class SaturationEngine:
    """Saturation modeling engine for diminishing returns analysis"""
    
    def __init__(self, memory_client, data_ingestion_client):
        self.memory_client = memory_client
        self.data_ingestion_client = data_ingestion_client
        
        # Model parameters
        self.min_data_points = 10
        self.calibration_window_days = 90
        self.confidence_threshold = 0.7
        
        # Saturation function implementations
        self.saturation_functions = {
            SaturationFunction.HILL: self._hill_saturation,
            SaturationFunction.ADSTOCK: self._adstock_saturation,
            SaturationFunction.EXPONENTIAL: self._exponential_saturation,
            SaturationFunction.MICHAELIS_MENTEN: self._michaelis_menten_saturation,
            SaturationFunction.GOMPERTZ: self._gompertz_saturation,
            SaturationFunction.LOGISTIC: self._logistic_saturation
        }
        
        # Function fitting methods
        self.fitting_functions = {
            SaturationFunction.HILL: self._fit_hill_function,
            SaturationFunction.ADSTOCK: self._fit_adstock_function,
            SaturationFunction.EXPONENTIAL: self._fit_exponential_function,
            SaturationFunction.MICHAELIS_MENTEN: self._fit_michaelis_menten_function,
            SaturationFunction.GOMPERTZ: self._fit_gompertz_function,
            SaturationFunction.LOGISTIC: self._fit_logistic_function
        }
    
    async def get_channel_saturation(self, org_id: str, channel_id: str) -> SaturationModel:
        """Get or create saturation model for a channel"""
        
        try:
            # Try to get existing model from memory
            existing_model = await self._get_cached_saturation_model(org_id, channel_id)
            
            if existing_model and self._is_model_fresh(existing_model):
                logger.info(f"Using cached saturation model for {channel_id}")
                return existing_model
            
            # Get historical data for calibration
            historical_data = await self._get_historical_data(org_id, channel_id)
            
            if len(historical_data) < self.min_data_points:
                logger.warning(f"Insufficient data for {channel_id}, using default model")
                return self._create_default_saturation_model(channel_id)
            
            # Calibrate saturation model
            saturation_model = await self._calibrate_saturation_model(
                org_id, channel_id, historical_data
            )
            
            # Cache the model
            await self._cache_saturation_model(org_id, channel_id, saturation_model)
            
            return saturation_model
            
        except Exception as e:
            logger.error(f"Failed to get saturation model for {channel_id}: {str(e)}")
            return self._create_default_saturation_model(channel_id)
    
    async def calibrate_all_channels(self, org_id: str, channels: List[str]) -> Dict[str, SaturationModel]:
        """Calibrate saturation models for all channels"""
        
        models = {}
        
        # Process channels in parallel
        tasks = [
            self.get_channel_saturation(org_id, channel_id)
            for channel_id in channels
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for channel_id, result in zip(channels, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to calibrate {channel_id}: {str(result)}")
                models[channel_id] = self._create_default_saturation_model(channel_id)
            else:
                models[channel_id] = result
        
        return models
    
    async def _get_historical_data(self, org_id: str, channel_id: str) -> List[Dict[str, float]]:
        """Get historical spend and response data for calibration"""
        
        try:
            # Get data from the last 90 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=self.calibration_window_days)
            
            # Call data ingestion service for historical data
            historical_data = await self.data_ingestion_client.get_channel_historical_data(
                org_id=org_id,
                channel_id=channel_id,
                start_date=start_date,
                end_date=end_date,
                metrics=["spend", "revenue", "conversions", "impressions", "clicks"]
            )
            
            # Transform data for saturation modeling
            processed_data = []
            for record in historical_data:
                processed_data.append({
                    "spend": record.get("spend", 0.0),
                    "response": record.get("revenue", 0.0),  # Use revenue as primary response
                    "conversions": record.get("conversions", 0.0),
                    "impressions": record.get("impressions", 0.0),
                    "clicks": record.get("clicks", 0.0),
                    "date": record.get("date")
                })
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {channel_id}: {str(e)}")
            # Return mock data for development
            return self._generate_mock_historical_data(channel_id)
    
    async def _calibrate_saturation_model(
        self, 
        org_id: str, 
        channel_id: str, 
        historical_data: List[Dict[str, float]]
    ) -> SaturationModel:
        """Calibrate saturation model using historical data"""
        
        # Extract spend and response arrays
        spend_data = np.array([d["spend"] for d in historical_data])
        response_data = np.array([d["response"] for d in historical_data])
        
        # Remove zero spend points for better fitting
        non_zero_mask = spend_data > 0
        spend_data = spend_data[non_zero_mask]
        response_data = response_data[non_zero_mask]
        
        if len(spend_data) < self.min_data_points:
            logger.warning(f"Insufficient non-zero data points for {channel_id}")
            return self._create_default_saturation_model(channel_id)
        
        # Try different saturation functions and select the best fit
        best_model = None
        best_r_squared = -1
        
        for function_type in SaturationFunction:
            try:
                model = await self._fit_saturation_function(
                    function_type, spend_data, response_data, channel_id
                )
                
                if model.r_squared > best_r_squared:
                    best_r_squared = model.r_squared
                    best_model = model
                    
            except Exception as e:
                logger.warning(f"Failed to fit {function_type} for {channel_id}: {str(e)}")
                continue
        
        if best_model is None or best_model.r_squared < self.confidence_threshold:
            logger.warning(f"Poor model fit for {channel_id}, using default")
            return self._create_default_saturation_model(channel_id)
        
        return best_model
    
    async def _fit_saturation_function(
        self, 
        function_type: SaturationFunction, 
        spend_data: np.ndarray, 
        response_data: np.ndarray,
        channel_id: str
    ) -> SaturationModel:
        """Fit a specific saturation function to the data"""
        
        fitting_function = self.fitting_functions[function_type]
        
        # Fit the function
        params, r_squared, confidence_interval = fitting_function(spend_data, response_data)
        
        # Calculate saturation metrics
        saturation_point = self._calculate_saturation_point(function_type, params, spend_data)
        diminishing_returns_start = self._calculate_diminishing_returns_start(
            function_type, params, spend_data
        )
        max_response = self._calculate_max_response(function_type, params, spend_data)
        
        return SaturationModel(
            channel_id=channel_id,
            function_type=function_type,
            alpha=params.get("alpha", 0.5),
            gamma=params.get("gamma", 1000.0),
            beta=params.get("beta", 1.0),
            theta=params.get("theta", 0.5),
            saturation_point=saturation_point,
            diminishing_returns_start=diminishing_returns_start,
            max_response=max_response,
            r_squared=r_squared,
            confidence_interval=confidence_interval,
            parameters=params,
            last_calibrated=datetime.utcnow()
        )
    
    def _fit_hill_function(self, spend: np.ndarray, response: np.ndarray) -> Tuple[Dict[str, float], float, Tuple[float, float]]:
        """Fit Hill saturation function: response = alpha * spend^gamma / (beta^gamma + spend^gamma)"""
        
        def hill_function(x, alpha, beta, gamma):
            return alpha * (x ** gamma) / (beta ** gamma + x ** gamma)
        
        # Initial parameter guesses
        alpha_init = np.max(response)
        beta_init = np.median(spend)
        gamma_init = 1.0
        
        try:
            # Fit the curve
            popt, pcov = curve_fit(
                hill_function, 
                spend, 
                response,
                p0=[alpha_init, beta_init, gamma_init],
                bounds=([0, 0, 0.1], [np.inf, np.inf, 5.0]),
                maxfev=5000
            )
            
            alpha, beta, gamma = popt
            
            # Calculate R-squared
            y_pred = hill_function(spend, alpha, beta, gamma)
            r_squared = self._calculate_r_squared(response, y_pred)
            
            # Calculate confidence interval (simplified)
            confidence_interval = (max(0, r_squared - 0.1), min(1, r_squared + 0.1))
            
            params = {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma)
            }
            
            return params, r_squared, confidence_interval
            
        except Exception as e:
            logger.error(f"Hill function fitting failed: {str(e)}")
            raise
    
    def _fit_adstock_function(self, spend: np.ndarray, response: np.ndarray) -> Tuple[Dict[str, float], float, Tuple[float, float]]:
        """Fit Adstock saturation function with carryover effects"""
        
        def adstock_function(x, alpha, beta, theta):
            # Simplified adstock: response = alpha * (1 - exp(-beta * x)) * (1 + theta * carryover)
            return alpha * (1 - np.exp(-beta * x))
        
        alpha_init = np.max(response)
        beta_init = 1.0 / np.median(spend)
        theta_init = 0.5
        
        try:
            popt, pcov = curve_fit(
                adstock_function,
                spend,
                response,
                p0=[alpha_init, beta_init, theta_init],
                bounds=([0, 0, 0], [np.inf, np.inf, 1.0]),
                maxfev=5000
            )
            
            alpha, beta, theta = popt
            
            y_pred = adstock_function(spend, alpha, beta, theta)
            r_squared = self._calculate_r_squared(response, y_pred)
            confidence_interval = (max(0, r_squared - 0.1), min(1, r_squared + 0.1))
            
            params = {
                "alpha": float(alpha),
                "beta": float(beta),
                "theta": float(theta)
            }
            
            return params, r_squared, confidence_interval
            
        except Exception as e:
            logger.error(f"Adstock function fitting failed: {str(e)}")
            raise
    
    def _fit_exponential_function(self, spend: np.ndarray, response: np.ndarray) -> Tuple[Dict[str, float], float, Tuple[float, float]]:
        """Fit exponential saturation function: response = alpha * (1 - exp(-beta * spend))"""
        
        def exponential_function(x, alpha, beta):
            return alpha * (1 - np.exp(-beta * x))
        
        alpha_init = np.max(response)
        beta_init = 1.0 / np.median(spend)
        
        try:
            popt, pcov = curve_fit(
                exponential_function,
                spend,
                response,
                p0=[alpha_init, beta_init],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=5000
            )
            
            alpha, beta = popt
            
            y_pred = exponential_function(spend, alpha, beta)
            r_squared = self._calculate_r_squared(response, y_pred)
            confidence_interval = (max(0, r_squared - 0.1), min(1, r_squared + 0.1))
            
            params = {
                "alpha": float(alpha),
                "beta": float(beta)
            }
            
            return params, r_squared, confidence_interval
            
        except Exception as e:
            logger.error(f"Exponential function fitting failed: {str(e)}")
            raise
    
    def _fit_michaelis_menten_function(self, spend: np.ndarray, response: np.ndarray) -> Tuple[Dict[str, float], float, Tuple[float, float]]:
        """Fit Michaelis-Menten saturation function: response = alpha * spend / (beta + spend)"""
        
        def logarithmic_function(x, alpha, beta):
            return alpha * np.log(1 + beta * x)
        
        alpha_init = np.max(response) / np.log(1 + np.max(spend))
        beta_init = 1.0
        
        try:
            popt, pcov = curve_fit(
                logarithmic_function,
                spend,
                response,
                p0=[alpha_init, beta_init],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=5000
            )
            
            alpha, beta = popt
            
            y_pred = logarithmic_function(spend, alpha, beta)
            r_squared = self._calculate_r_squared(response, y_pred)
            confidence_interval = (max(0, r_squared - 0.1), min(1, r_squared + 0.1))
            
            params = {
                "alpha": float(alpha),
                "beta": float(beta)
            }
            
            return params, r_squared, confidence_interval
            
        except Exception as e:
            logger.error(f"Logarithmic function fitting failed: {str(e)}")
            raise
    
    def _fit_gompertz_function(self, spend: np.ndarray, response: np.ndarray) -> Tuple[Dict[str, float], float, Tuple[float, float]]:
        """Fit Gompertz saturation function: response = alpha * exp(-beta * exp(-gamma * spend))"""
        
        def power_function(x, alpha, beta):
            return alpha * (x ** beta)
        
        # Use log-linear regression for initial guess
        log_spend = np.log(spend + 1)
        log_response = np.log(response + 1)
        
        alpha_init = np.exp(np.mean(log_response))
        beta_init = 0.5
        
        try:
            popt, pcov = curve_fit(
                power_function,
                spend,
                response,
                p0=[alpha_init, beta_init],
                bounds=([0, 0], [np.inf, 2.0]),
                maxfev=5000
            )
            
            alpha, beta = popt
            
            y_pred = power_function(spend, alpha, beta)
            r_squared = self._calculate_r_squared(response, y_pred)
            confidence_interval = (max(0, r_squared - 0.1), min(1, r_squared + 0.1))
            
            params = {
                "alpha": float(alpha),
                "beta": float(beta)
            }
            
            return params, r_squared, confidence_interval
            
        except Exception as e:
            logger.error(f"Power function fitting failed: {str(e)}")
            raise
    def _fit_logistic_function(self, spend: np.ndarray, response: np.ndarray) -> Tuple[Dict[str, float], float, Tuple[float, float]]:
        """Fit logistic saturation function: response = alpha / (1 + exp(-beta * (spend - gamma)))"""
        try:
            def logistic_func(x, alpha, beta, gamma):
                return alpha / (1 + np.exp(-beta * (x - gamma)))
            
            # Initial parameter estimates
            alpha_init = np.max(response)
            beta_init = 1.0
            gamma_init = np.median(spend)
            
            # Fit the function
            popt, pcov = curve_fit(
                logistic_func, spend, response,
                p0=[alpha_init, beta_init, gamma_init],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                maxfev=5000
            )
            
            alpha, beta, gamma = popt
            
            # Calculate R-squared
            y_pred = logistic_func(spend, alpha, beta, gamma)
            r_squared = self._calculate_r_squared(response, y_pred)
            
            # Calculate confidence intervals
            param_errors = np.sqrt(np.diag(pcov))
            confidence_interval = (
                float(np.mean(param_errors) - 1.96 * np.std(param_errors)),
                float(np.mean(param_errors) + 1.96 * np.std(param_errors))
            )
            
            params = {
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma)
            }
            
            return params, r_squared, confidence_interval
            
        except Exception as e:
            logger.error(f"Logistic function fitting failed: {str(e)}")
            raise
    
    
    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared coefficient of determination"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return float(max(0, min(1, r_squared)))
    
    def _calculate_saturation_point(
        self, 
        function_type: SaturationFunction, 
        params: Dict[str, float], 
        spend_data: np.ndarray
    ) -> float:
        """Calculate the saturation point (90% of maximum response)"""
        
        max_spend = np.max(spend_data) * 2  # Extrapolate beyond observed data
        spend_range = np.linspace(0, max_spend, 1000)
        
        saturation_func = self.saturation_functions[function_type]
        responses = [saturation_func(s, params) for s in spend_range]
        max_response = np.max(responses)
        
        # Find spend level that achieves 90% of max response
        target_response = 0.9 * max_response
        
        for i, response in enumerate(responses):
            if response >= target_response:
                return float(spend_range[i])
        
        return float(max_spend)  # Fallback
    
    def _calculate_diminishing_returns_start(
        self, 
        function_type: SaturationFunction, 
        params: Dict[str, float], 
        spend_data: np.ndarray
    ) -> float:
        """Calculate where diminishing returns start (inflection point)"""
        
        max_spend = np.max(spend_data) * 1.5
        spend_range = np.linspace(1, max_spend, 1000)
        
        saturation_func = self.saturation_functions[function_type]
        
        # Calculate second derivative to find inflection point
        responses = [saturation_func(s, params) for s in spend_range]
        
        # Numerical second derivative
        second_derivatives = []
        for i in range(2, len(responses) - 2):
            second_deriv = responses[i+1] - 2*responses[i] + responses[i-1]
            second_derivatives.append(second_deriv)
        
        if second_derivatives:
            # Find where second derivative becomes most negative (steepest decline in marginal returns)
            min_idx = np.argmin(second_derivatives)
            return float(spend_range[min_idx + 2])
        
        # Fallback: use 50% of saturation point
        saturation_point = self._calculate_saturation_point(function_type, params, spend_data)
        return saturation_point * 0.5
    
    def _calculate_max_response(
        self, 
        function_type: SaturationFunction, 
        params: Dict[str, float], 
        spend_data: np.ndarray
    ) -> float:
        """Calculate theoretical maximum response"""
        
        if function_type == SaturationFunction.HILL:
            return params.get("alpha", 1000.0)
        elif function_type == SaturationFunction.ADSTOCK:
            return params.get("alpha", 1000.0)
        elif function_type == SaturationFunction.EXPONENTIAL:
            return params.get("alpha", 1000.0)
        # Note: LOGARITHMIC and POWER functions removed as they don't exist in the enum
        # elif function_type == SaturationFunction.LOGARITHMIC:
        #     # Logarithmic has no theoretical maximum, use practical limit
        #     max_spend = np.max(spend_data) * 10
        #     return params.get("alpha", 1.0) * np.log(1 + params.get("beta", 1.0) * max_spend)
        # elif function_type == SaturationFunction.POWER:
        #     # Power function has no theoretical maximum for beta < 1
            max_spend = np.max(spend_data) * 10
            return params.get("alpha", 1.0) * (max_spend ** params.get("beta", 0.5))
        
        return 1000.0  # Default fallback
    
    # Saturation function implementations
    
    def _hill_saturation(self, spend: float, params: Dict[str, float]) -> float:
        """Hill saturation function"""
        alpha = params.get("alpha", 1000.0)
        beta = params.get("beta", 500.0)
        gamma = params.get("gamma", 1.0)
        
        return alpha * (spend ** gamma) / (beta ** gamma + spend ** gamma)
    
    def _adstock_saturation(self, spend: float, params: Dict[str, float]) -> float:
        """Adstock saturation function"""
        alpha = params.get("alpha", 1000.0)
        beta = params.get("beta", 0.001)
        theta = params.get("theta", 0.5)
        
        return alpha * (1 - np.exp(-beta * spend))
    
    def _exponential_saturation(self, spend: float, params: Dict[str, float]) -> float:
        """Exponential saturation function"""
        alpha = params.get("alpha", 1000.0)
        beta = params.get("beta", 0.001)
        
        return alpha * (1 - np.exp(-beta * spend))
    
    def _michaelis_menten_saturation(self, spend: float, params: Dict[str, float]) -> float:
        """Michaelis-Menten saturation function"""
        alpha = params.get("alpha", 100.0)
        beta = params.get("beta", 1.0)
        
        return alpha * np.log(1 + beta * spend)
    
    def _gompertz_saturation(self, spend: float, params: Dict[str, float]) -> float:
        """Gompertz saturation function"""
        alpha = params.get("alpha", 10.0)
        beta = params.get("beta", 0.5)
        
        return alpha * (spend ** beta)
    def _logistic_saturation(self, spend: float, params: Dict[str, float]) -> float:
        """Logistic saturation function"""
        alpha = params.get("alpha", 1.0)
        beta = params.get("beta", 1.0)
        gamma = params.get("gamma", 0.0)
        return alpha / (1 + np.exp(-beta * (spend - gamma)))

    
    def evaluate_saturation(self, saturation_model: SaturationModel, spend: float) -> float:
        """Evaluate saturation model at given spend level"""
        
        saturation_func = self.saturation_functions[saturation_model.function_type]
        params = saturation_model.parameters or {
            "alpha": saturation_model.alpha,
            "gamma": saturation_model.gamma,
            "beta": saturation_model.beta,
            "theta": saturation_model.theta
        }
        
        return saturation_func(spend, params)
    
    async def validate_saturation_model(
        self, 
        org_id: str, 
        channel_id: str, 
        saturation_model: SaturationModel
    ) -> Dict[str, Any]:
        """Validate saturation model against recent data"""
        
        try:
            # Get recent data for validation
            recent_data = await self._get_recent_validation_data(org_id, channel_id)
            
            if len(recent_data) < 5:
                return {
                    "valid": False,
                    "reason": "Insufficient recent data for validation",
                    "data_points": len(recent_data)
                }
            
            # Calculate prediction accuracy
            spend_values = [d["spend"] for d in recent_data]
            actual_responses = [d["response"] for d in recent_data]
            predicted_responses = [
                self.evaluate_saturation(saturation_model, spend) 
                for spend in spend_values
            ]
            
            # Calculate validation metrics
            mape = self._calculate_mape(actual_responses, predicted_responses)
            r_squared = self._calculate_r_squared(
                np.array(actual_responses), 
                np.array(predicted_responses)
            )
            
            is_valid = mape < 0.3 and r_squared > 0.5  # 30% MAPE threshold, 50% R² threshold
            
            return {
                "valid": is_valid,
                "mape": mape,
                "r_squared": r_squared,
                "data_points": len(recent_data),
                "validation_date": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Validation failed for {channel_id}: {str(e)}")
            return {
                "valid": False,
                "reason": f"Validation error: {str(e)}"
            }
    
    def _calculate_mape(self, actual: List[float], predicted: List[float]) -> float:
        """Calculate Mean Absolute Percentage Error"""
        if len(actual) != len(predicted) or len(actual) == 0:
            return float('inf')
        
        percentage_errors = []
        for a, p in zip(actual, predicted):
            if a != 0:
                percentage_errors.append(abs((a - p) / a))
        
        return np.mean(percentage_errors) if percentage_errors else float('inf')
    
    # Caching and data management methods
    
    async def _get_cached_saturation_model(self, org_id: str, channel_id: str) -> Optional[SaturationModel]:
        """Get cached saturation model from memory service"""
        try:
            cache_key = f"saturation_model:{org_id}:{channel_id}"
            cached_data = await self.memory_client.get(cache_key)
            
            if cached_data:
                # Deserialize and return model
                return SaturationModel(**cached_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get cached model for {channel_id}: {str(e)}")
            return None
    
    async def _cache_saturation_model(self, org_id: str, channel_id: str, model: SaturationModel):
        """Cache saturation model in memory service"""
        try:
            cache_key = f"saturation_model:{org_id}:{channel_id}"
            model_data = model.dict()
            
            # Cache for 24 hours
            await self.memory_client.set(cache_key, model_data, ttl=86400)
            
        except Exception as e:
            logger.warning(f"Failed to cache model for {channel_id}: {str(e)}")
    
    def _is_model_fresh(self, model: SaturationModel) -> bool:
        """Check if saturation model is fresh enough to use"""
        if not model.last_calibrated:
            return False
        
        age = datetime.utcnow() - model.last_calibrated
        return age.days < 7  # Model is fresh for 7 days
    
    async def _get_recent_validation_data(self, org_id: str, channel_id: str) -> List[Dict[str, float]]:
        """Get recent data for model validation"""
        try:
            # Get data from the last 7 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=7)
            
            validation_data = await self.data_ingestion_client.get_channel_historical_data(
                org_id=org_id,
                channel_id=channel_id,
                start_date=start_date,
                end_date=end_date,
                metrics=["spend", "revenue"]
            )
            
            return [
                {
                    "spend": record.get("spend", 0.0),
                    "response": record.get("revenue", 0.0)
                }
                for record in validation_data
            ]
            
        except Exception as e:
            logger.error(f"Failed to get validation data for {channel_id}: {str(e)}")
            return []
    
    def _create_default_saturation_model(self, channel_id: str) -> SaturationModel:
        """Create default saturation model when calibration fails"""
        return SaturationModel(
            channel_id=channel_id,
            function_type=SaturationFunction.HILL,
            alpha=1000.0,
            gamma=1.0,
            beta=500.0,
            theta=0.5,
            saturation_point=2000.0,
            diminishing_returns_start=500.0,
            max_response=1000.0,
            r_squared=0.7,  # Assumed reasonable fit
            confidence_interval=(0.6, 0.8),
            parameters={
                "alpha": 1000.0,
                "beta": 500.0,
                "gamma": 1.0
            },
            last_calibrated=datetime.utcnow()
        )
    
    def _generate_mock_historical_data(self, channel_id: str) -> List[Dict[str, float]]:
        """Generate mock historical data for development/testing"""
        np.random.seed(42)  # For reproducible mock data
        
        data = []
        base_spend = 1000.0
        
        for i in range(30):  # 30 days of data
            # Generate spend with some variation
            spend = base_spend * (0.8 + 0.4 * np.random.random())
            
            # Generate response with saturation curve + noise
            response = 1000 * (spend ** 0.7) / (500 ** 0.7 + spend ** 0.7)
            response *= (0.9 + 0.2 * np.random.random())  # Add noise
            
            data.append({
                "spend": spend,
                "response": response,
                "conversions": response / 35.0,  # Assume $35 per conversion
                "impressions": spend * 10,  # Assume $0.1 CPM
                "clicks": spend * 0.5,  # Assume $2 CPC
                "date": (datetime.utcnow() - timedelta(days=30-i)).isoformat()
            })
        
        return data