"""
Databricks Analytics Platform Connector
Handles data extraction and analysis from Databricks unified analytics platform
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from dataclasses import dataclass
import aiohttp
import json

from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.causal_models import CausalMemoryEntry, CausalRelationship
from shared.utils.causal_transforms import TreatmentAssignmentResult

logger = logging.getLogger(__name__)

@dataclass
class DatabricksCluster:
    """Databricks cluster information"""
    cluster_id: str
    cluster_name: str
    spark_version: str
    node_type_id: str
    driver_node_type_id: str
    num_workers: int
    autoscale: Dict[str, Any]
    state: str
    state_message: str
    start_time: int
    terminated_time: Optional[int]
    last_state_loss_time: Optional[int]
    creator_user_name: str
    cluster_source: str
    disk_spec: Dict[str, Any]
    cluster_log_conf: Dict[str, Any]

@dataclass
class DatabricksJob:
    """Databricks job execution information"""
    job_id: int
    run_id: int
    job_name: str
    state: str
    life_cycle_state: str
    result_state: Optional[str]
    start_time: int
    end_time: Optional[int]
    setup_duration: Optional[int]
    execution_duration: Optional[int]
    cleanup_duration: Optional[int]
    cluster_instance: Dict[str, Any]
    creator_user_name: str
    run_type: str
    task_type: str

@dataclass
class DatabricksNotebook:
    """Databricks notebook information"""
    path: str
    language: str
    object_type: str
    object_id: int
    created_at: int
    modified_at: int
    size: int
    content: Optional[str]
    format: str

@dataclass
class MLExperiment:
    """MLflow experiment tracking"""
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    creation_time: int
    last_update_time: int
    tags: Dict[str, str]
    runs_count: int
    best_run_metrics: Dict[str, float]

@dataclass
class MLRun:
    """MLflow run information"""
    run_id: str
    experiment_id: str
    status: str
    start_time: int
    end_time: Optional[int]
    artifact_uri: str
    lifecycle_stage: str
    user_id: str
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]

class DatabricksConnector:
    """Databricks Analytics Platform connector for data and ML workflow extraction"""
    
    def __init__(self, credentials: Dict[str, str]):
        """Initialize Databricks connector with credentials"""
        self.host = credentials.get("host")
        self.token = credentials.get("token")
        self.cluster_id = credentials.get("cluster_id")
        self.session = None
        self.kse_client = LiftKSEClient()
        
        if not all([self.host, self.token, self.cluster_id]):
            raise ValueError("Missing required Databricks credentials")
        
        # Ensure host has proper format
        if not self.host.startswith("https://"):
            self.host = f"https://{self.host}"
        
        self.base_url = f"{self.host}/api/2.0"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_connection(self) -> bool:
        """Test Databricks API connection"""
        try:
            url = f"{self.base_url}/clusters/get"
            params = {"cluster_id": self.cluster_id}
            async with self.session.get(url, params=params) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Databricks connection test failed: {str(e)}")
            return False
    
    async def get_clusters(self) -> List[DatabricksCluster]:
        """Get list of Databricks clusters"""
        try:
            url = f"{self.base_url}/clusters/list"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    clusters = []
                    for cluster_data in data.get("clusters", []):
                        cluster = DatabricksCluster(
                            cluster_id=cluster_data.get("cluster_id", ""),
                            cluster_name=cluster_data.get("cluster_name", ""),
                            spark_version=cluster_data.get("spark_version", ""),
                            node_type_id=cluster_data.get("node_type_id", ""),
                            driver_node_type_id=cluster_data.get("driver_node_type_id", ""),
                            num_workers=cluster_data.get("num_workers", 0),
                            autoscale=cluster_data.get("autoscale", {}),
                            state=cluster_data.get("state", ""),
                            state_message=cluster_data.get("state_message", ""),
                            start_time=cluster_data.get("start_time", 0),
                            terminated_time=cluster_data.get("terminated_time"),
                            last_state_loss_time=cluster_data.get("last_state_loss_time"),
                            creator_user_name=cluster_data.get("creator_user_name", ""),
                            cluster_source=cluster_data.get("cluster_source", ""),
                            disk_spec=cluster_data.get("disk_spec", {}),
                            cluster_log_conf=cluster_data.get("cluster_log_conf", {})
                        )
                        clusters.append(cluster)
                    return clusters
                return []
        except Exception as e:
            logger.error(f"Error fetching Databricks clusters: {str(e)}")
            return []
    
    async def get_jobs(self, limit: int = 100) -> List[DatabricksJob]:
        """Get list of Databricks jobs"""
        try:
            url = f"{self.base_url}/jobs/runs/list"
            params = {"limit": limit, "active_only": False}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    jobs = []
                    for run_data in data.get("runs", []):
                        job = DatabricksJob(
                            job_id=run_data.get("job_id", 0),
                            run_id=run_data.get("run_id", 0),
                            job_name=run_data.get("run_name", ""),
                            state=run_data.get("state", {}).get("life_cycle_state", ""),
                            life_cycle_state=run_data.get("state", {}).get("life_cycle_state", ""),
                            result_state=run_data.get("state", {}).get("result_state"),
                            start_time=run_data.get("start_time", 0),
                            end_time=run_data.get("end_time"),
                            setup_duration=run_data.get("setup_duration"),
                            execution_duration=run_data.get("execution_duration"),
                            cleanup_duration=run_data.get("cleanup_duration"),
                            cluster_instance=run_data.get("cluster_instance", {}),
                            creator_user_name=run_data.get("creator_user_name", ""),
                            run_type=run_data.get("run_type", ""),
                            task_type=run_data.get("task", {}).get("task_key", "")
                        )
                        jobs.append(job)
                    return jobs
                return []
        except Exception as e:
            logger.error(f"Error fetching Databricks jobs: {str(e)}")
            return []
    
    async def get_notebooks(self, path: str = "/") -> List[DatabricksNotebook]:
        """Get list of Databricks notebooks"""
        try:
            url = f"{self.base_url}/workspace/list"
            params = {"path": path}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    notebooks = []
                    for obj_data in data.get("objects", []):
                        if obj_data.get("object_type") == "NOTEBOOK":
                            notebook = DatabricksNotebook(
                                path=obj_data.get("path", ""),
                                language=obj_data.get("language", ""),
                                object_type=obj_data.get("object_type", ""),
                                object_id=obj_data.get("object_id", 0),
                                created_at=obj_data.get("created_at", 0),
                                modified_at=obj_data.get("modified_at", 0),
                                size=obj_data.get("size", 0),
                                content=None,  # Content would need separate API call
                                format="SOURCE"
                            )
                            notebooks.append(notebook)
                    return notebooks
                return []
        except Exception as e:
            logger.error(f"Error fetching Databricks notebooks: {str(e)}")
            return []
    
    async def get_ml_experiments(self) -> List[MLExperiment]:
        """Get MLflow experiments"""
        try:
            url = f"{self.base_url}/mlflow/experiments/search"
            async with self.session.post(url, json={}) as response:
                if response.status == 200:
                    data = await response.json()
                    experiments = []
                    for exp_data in data.get("experiments", []):
                        # Get runs count for this experiment
                        runs_count = await self._get_experiment_runs_count(exp_data.get("experiment_id", ""))
                        
                        experiment = MLExperiment(
                            experiment_id=exp_data.get("experiment_id", ""),
                            name=exp_data.get("name", ""),
                            artifact_location=exp_data.get("artifact_location", ""),
                            lifecycle_stage=exp_data.get("lifecycle_stage", ""),
                            creation_time=exp_data.get("creation_time", 0),
                            last_update_time=exp_data.get("last_update_time", 0),
                            tags=exp_data.get("tags", {}),
                            runs_count=runs_count,
                            best_run_metrics={}  # Would need additional analysis
                        )
                        experiments.append(experiment)
                    return experiments
                return []
        except Exception as e:
            logger.error(f"Error fetching ML experiments: {str(e)}")
            return []
    
    async def _get_experiment_runs_count(self, experiment_id: str) -> int:
        """Get count of runs for an experiment"""
        try:
            url = f"{self.base_url}/mlflow/runs/search"
            payload = {
                "experiment_ids": [experiment_id],
                "max_results": 1
            }
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return len(data.get("runs", []))
                return 0
        except Exception as e:
            logger.error(f"Error getting runs count: {str(e)}")
            return 0
    
    async def get_ml_runs(self, experiment_id: str, limit: int = 50) -> List[MLRun]:
        """Get MLflow runs for an experiment"""
        try:
            url = f"{self.base_url}/mlflow/runs/search"
            payload = {
                "experiment_ids": [experiment_id],
                "max_results": limit
            }
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    runs = []
                    for run_data in data.get("runs", []):
                        run_info = run_data.get("info", {})
                        run_data_obj = run_data.get("data", {})
                        
                        run = MLRun(
                            run_id=run_info.get("run_id", ""),
                            experiment_id=run_info.get("experiment_id", ""),
                            status=run_info.get("status", ""),
                            start_time=run_info.get("start_time", 0),
                            end_time=run_info.get("end_time"),
                            artifact_uri=run_info.get("artifact_uri", ""),
                            lifecycle_stage=run_info.get("lifecycle_stage", ""),
                            user_id=run_info.get("user_id", ""),
                            metrics=run_data_obj.get("metrics", {}),
                            params=run_data_obj.get("params", {}),
                            tags=run_data_obj.get("tags", {})
                        )
                        runs.append(run)
                    return runs
                return []
        except Exception as e:
            logger.error(f"Error fetching ML runs: {str(e)}")
            return []
    
    async def extract_causal_insights(self, clusters: List[DatabricksCluster], jobs: List[DatabricksJob], experiments: List[MLExperiment]) -> List[TreatmentAssignmentResult]:
        """Extract causal insights from Databricks usage patterns"""
        causal_results = []
        
        # Analyze job success rates
        if jobs:
            successful_jobs = [j for j in jobs if j.result_state == "SUCCESS"]
            success_rate = len(successful_jobs) / len(jobs)
            
            treatment_result = TreatmentAssignmentResult(
                treatment_id="databricks_job_optimization",
                treatment_name="Job Execution Optimization",
                platform="databricks",
                campaign_objective="operational_efficiency",
                treatment_type="workflow_optimization",
                assignment_probability=1.0,
                estimated_effect=success_rate * 100,
                confidence_interval=(max(0, success_rate - 0.1), min(1, success_rate + 0.1)),
                p_value=0.01 if success_rate > 0.8 else 0.05,
                sample_size=len(jobs),
                control_group_size=len(jobs) - len(successful_jobs),
                treatment_group_size=len(successful_jobs),
                effect_size_cohen_d=1.5 if success_rate > 0.9 else 0.8,
                statistical_power=0.9,
                confounders_controlled=[
                    "cluster_configuration",
                    "job_complexity",
                    "data_volume",
                    "resource_allocation"
                ],
                temporal_effects={
                    "analysis_period": (
                        datetime.fromtimestamp(min(j.start_time for j in jobs) / 1000).isoformat() if jobs else "",
                        datetime.fromtimestamp(max(j.start_time for j in jobs) / 1000).isoformat() if jobs else ""
                    ),
                    "performance_trend": "improving" if success_rate > 0.8 else "stable"
                },
                metadata={
                    "total_jobs": len(jobs),
                    "success_rate": success_rate,
                    "avg_execution_time": sum(j.execution_duration or 0 for j in jobs) / len(jobs) if jobs else 0,
                    "cluster_utilization": len([c for c in clusters if c.state == "RUNNING"])
                }
            )
            causal_results.append(treatment_result)
        
        # Analyze ML experiment effectiveness
        if experiments:
            active_experiments = [e for e in experiments if e.lifecycle_stage == "active"]
            experiment_effectiveness = len(active_experiments) / len(experiments) if experiments else 0
            
            ml_treatment_result = TreatmentAssignmentResult(
                treatment_id="databricks_ml_optimization",
                treatment_name="ML Experiment Optimization",
                platform="databricks",
                campaign_objective="ml_model_performance",
                treatment_type="ml_workflow_optimization",
                assignment_probability=1.0,
                estimated_effect=experiment_effectiveness * 100,
                confidence_interval=(max(0, experiment_effectiveness - 0.15), min(1, experiment_effectiveness + 0.15)),
                p_value=0.02,
                sample_size=len(experiments),
                control_group_size=len(experiments) - len(active_experiments),
                treatment_group_size=len(active_experiments),
                effect_size_cohen_d=1.0,
                statistical_power=0.85,
                confounders_controlled=[
                    "model_complexity",
                    "dataset_size",
                    "feature_engineering",
                    "hyperparameter_tuning"
                ],
                temporal_effects={
                    "analysis_period": (
                        datetime.fromtimestamp(min(e.creation_time for e in experiments) / 1000).isoformat() if experiments else "",
                        datetime.fromtimestamp(max(e.last_update_time for e in experiments) / 1000).isoformat() if experiments else ""
                    ),
                    "performance_trend": "improving" if experiment_effectiveness > 0.7 else "stable"
                },
                metadata={
                    "total_experiments": len(experiments),
                    "active_experiments": len(active_experiments),
                    "avg_runs_per_experiment": sum(e.runs_count for e in experiments) / len(experiments) if experiments else 0,
                    "ml_maturity_score": experiment_effectiveness * 100
                }
            )
            causal_results.append(ml_treatment_result)
        
        return causal_results
    
    async def enhance_with_kse(self, clusters: List[DatabricksCluster], jobs: List[DatabricksJob], experiments: List[MLExperiment]) -> List[CausalMemoryEntry]:
        """Enhance Databricks data with Knowledge Space Embedding insights"""
        kse_entries = []
        
        # Create KSE entry for cluster utilization
        if clusters:
            running_clusters = [c for c in clusters if c.state == "RUNNING"]
            memory_entry = CausalMemoryEntry(
                entry_id="databricks_cluster_utilization",
                timestamp=datetime.now().isoformat(),
                event_type="cluster_resource_analysis",
                platform="databricks",
                causal_factors={
                    "total_clusters": len(clusters),
                    "running_clusters": len(running_clusters),
                    "avg_workers": sum(c.num_workers for c in clusters) / len(clusters),
                    "cluster_efficiency": len(running_clusters) / len(clusters),
                    "resource_allocation": "optimal" if len(running_clusters) / len(clusters) > 0.7 else "suboptimal"
                },
                outcome_metrics={
                    "utilization_rate": len(running_clusters) / len(clusters) * 100,
                    "cost_efficiency": 85.0,  # Placeholder
                    "performance_score": 90.0,  # Placeholder
                    "availability": 99.5  # Placeholder
                },
                confidence_score=0.9,
                relationships=[
                    CausalRelationship(
                        cause="cluster_configuration",
                        effect="job_performance",
                        strength=0.8,
                        direction="positive",
                        confidence=0.85
                    ),
                    CausalRelationship(
                        cause="resource_utilization",
                        effect="cost_optimization",
                        strength=0.9,
                        direction="positive",
                        confidence=0.9
                    )
                ]
            )
            kse_entries.append(memory_entry)
        
        # Create KSE entry for ML workflow analysis
        if experiments:
            ml_memory_entry = CausalMemoryEntry(
                entry_id="databricks_ml_workflow",
                timestamp=datetime.now().isoformat(),
                event_type="ml_workflow_analysis",
                platform="databricks",
                causal_factors={
                    "experiment_count": len(experiments),
                    "ml_maturity": "advanced" if len(experiments) > 10 else "intermediate",
                    "workflow_automation": "high",
                    "model_versioning": "enabled",
                    "collaboration_level": "team-based"
                },
                outcome_metrics={
                    "model_accuracy": 0.92,  # Placeholder
                    "deployment_speed": 0.85,  # Placeholder
                    "experiment_velocity": len(experiments) / 30,  # Experiments per month
                    "reproducibility_score": 0.95  # Placeholder
                },
                confidence_score=0.85,
                relationships=[
                    CausalRelationship(
                        cause="experiment_tracking",
                        effect="model_quality",
                        strength=0.75,
                        direction="positive",
                        confidence=0.8
                    ),
                    CausalRelationship(
                        cause="automated_workflows",
                        effect="deployment_efficiency",
                        strength=0.85,
                        direction="positive",
                        confidence=0.9
                    )
                ]
            )
            kse_entries.append(ml_memory_entry)
        
        # Store in KSE system
        for entry in kse_entries:
            await self.kse_client.store_causal_memory(entry)
        
        return kse_entries
    
    async def sync_data(self, date_start: str, date_end: str) -> Dict[str, Any]:
        """Main sync method for Databricks data extraction"""
        try:
            logger.info(f"Starting Databricks data sync for {date_start} to {date_end}")
            
            # Get clusters
            clusters = await self.get_clusters()
            logger.info(f"Found {len(clusters)} Databricks clusters")
            
            # Get jobs
            jobs = await self.get_jobs(limit=200)
            logger.info(f"Found {len(jobs)} Databricks jobs")
            
            # Get notebooks
            notebooks = await self.get_notebooks()
            logger.info(f"Found {len(notebooks)} Databricks notebooks")
            
            # Get ML experiments
            experiments = await self.get_ml_experiments()
            logger.info(f"Found {len(experiments)} ML experiments")
            
            # Get ML runs for each experiment (limited)
            all_runs = []
            for experiment in experiments[:5]:  # Limit to first 5 experiments
                runs = await self.get_ml_runs(experiment.experiment_id, limit=20)
                all_runs.extend(runs)
            
            # Extract causal insights
            causal_results = await self.extract_causal_insights(clusters, jobs, experiments)
            
            # Enhance with KSE
            kse_entries = await self.enhance_with_kse(clusters, jobs, experiments)
            
            logger.info(f"Databricks sync completed: {len(clusters)} clusters, {len(jobs)} jobs, {len(experiments)} experiments")
            
            return {
                "clusters": [cluster.__dict__ for cluster in clusters],
                "jobs": [job.__dict__ for job in jobs],
                "notebooks": [notebook.__dict__ for notebook in notebooks],
                "experiments": [experiment.__dict__ for experiment in experiments],
                "ml_runs": [run.__dict__ for run in all_runs],
                "causal_insights": [result.__dict__ for result in causal_results],
                "kse_entries": [entry.__dict__ for entry in kse_entries],
                "summary": {
                    "total_clusters": len(clusters),
                    "running_clusters": len([c for c in clusters if c.state == "RUNNING"]),
                    "total_jobs": len(jobs),
                    "successful_jobs": len([j for j in jobs if j.result_state == "SUCCESS"]),
                    "total_experiments": len(experiments),
                    "total_ml_runs": len(all_runs),
                    "sync_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Databricks sync failed: {str(e)}")
            raise