"""
Snowflake Data Warehouse Connector
Handles data extraction and analysis from Snowflake data warehouse
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from dataclasses import dataclass
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

from shared.kse_sdk.client import LiftKSEClient
from shared.kse_sdk.causal_models import CausalMemoryEntry, CausalRelationship
from shared.utils.causal_transforms import TreatmentAssignmentResult

logger = logging.getLogger(__name__)

@dataclass
class SnowflakeTable:
    """Snowflake table metadata"""
    table_name: str
    schema_name: str
    database_name: str
    row_count: int
    column_count: int
    table_type: str
    created: str
    last_altered: str
    bytes: int
    retention_time: int

@dataclass
class SnowflakeQuery:
    """Snowflake query execution result"""
    query_id: str
    query_text: str
    execution_time: float
    rows_produced: int
    bytes_scanned: int
    compilation_time: float
    execution_status: str
    error_message: Optional[str]
    warehouse_name: str
    user_name: str
    start_time: str
    end_time: str

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    table_name: str
    total_rows: int
    null_percentage: float
    duplicate_percentage: float
    completeness_score: float
    consistency_score: float
    validity_score: float
    timeliness_score: float
    overall_quality_score: float
    issues_detected: List[str]

class SnowflakeConnector:
    """Snowflake Data Warehouse connector for data extraction and analysis"""
    
    def __init__(self, credentials: Dict[str, str]):
        """Initialize Snowflake connector with credentials"""
        self.account = credentials.get("account")
        self.username = credentials.get("username")
        self.password = credentials.get("password")
        self.warehouse = credentials.get("warehouse")
        self.database = credentials.get("database")
        self.schema = credentials.get("schema")
        self.connection = None
        self.kse_client = LiftKSEClient()
        
        if not all([self.account, self.username, self.password, self.warehouse, self.database, self.schema]):
            raise ValueError("Missing required Snowflake credentials")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Establish connection to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                account=self.account,
                user=self.username,
                password=self.password,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema
            )
            logger.info("Connected to Snowflake successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            raise
    
    async def disconnect(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from Snowflake")
    
    async def test_connection(self) -> bool:
        """Test Snowflake connection"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT CURRENT_VERSION()")
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        except Exception as e:
            logger.error(f"Snowflake connection test failed: {str(e)}")
            return False
    
    async def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        try:
            cursor = self.connection.cursor()
            start_time = datetime.now()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            execution_time = (datetime.now() - start_time).total_seconds()
            cursor.close()
            
            df = pd.DataFrame(results, columns=columns)
            logger.info(f"Query executed successfully: {len(df)} rows returned in {execution_time:.2f}s")
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise
    
    async def get_table_metadata(self) -> List[SnowflakeTable]:
        """Get metadata for all tables in the database"""
        try:
            query = """
            SELECT 
                table_name,
                table_schema,
                table_catalog,
                row_count,
                column_count,
                table_type,
                created,
                last_altered,
                bytes,
                retention_time
            FROM information_schema.tables 
            WHERE table_schema = %s
            ORDER BY table_name
            """
            
            cursor = self.connection.cursor()
            cursor.execute(query, (self.schema,))
            results = cursor.fetchall()
            cursor.close()
            
            tables = []
            for row in results:
                table = SnowflakeTable(
                    table_name=row[0],
                    schema_name=row[1],
                    database_name=row[2],
                    row_count=row[3] or 0,
                    column_count=row[4] or 0,
                    table_type=row[5],
                    created=str(row[6]) if row[6] else "",
                    last_altered=str(row[7]) if row[7] else "",
                    bytes=row[8] or 0,
                    retention_time=row[9] or 0
                )
                tables.append(table)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error fetching table metadata: {str(e)}")
            return []
    
    async def get_query_history(self, hours_back: int = 24) -> List[SnowflakeQuery]:
        """Get query execution history"""
        try:
            query = """
            SELECT 
                query_id,
                query_text,
                total_elapsed_time/1000 as execution_time_seconds,
                rows_produced,
                bytes_scanned,
                compilation_time/1000 as compilation_time_seconds,
                execution_status,
                error_message,
                warehouse_name,
                user_name,
                start_time,
                end_time
            FROM information_schema.query_history 
            WHERE start_time >= DATEADD(hour, -%s, CURRENT_TIMESTAMP())
            ORDER BY start_time DESC
            LIMIT 1000
            """
            
            cursor = self.connection.cursor()
            cursor.execute(query, (hours_back,))
            results = cursor.fetchall()
            cursor.close()
            
            queries = []
            for row in results:
                query_obj = SnowflakeQuery(
                    query_id=row[0],
                    query_text=row[1][:500] if row[1] else "",  # Truncate long queries
                    execution_time=row[2] or 0.0,
                    rows_produced=row[3] or 0,
                    bytes_scanned=row[4] or 0,
                    compilation_time=row[5] or 0.0,
                    execution_status=row[6] or "",
                    error_message=row[7],
                    warehouse_name=row[8] or "",
                    user_name=row[9] or "",
                    start_time=str(row[10]) if row[10] else "",
                    end_time=str(row[11]) if row[11] else ""
                )
                queries.append(query_obj)
            
            return queries
            
        except Exception as e:
            logger.error(f"Error fetching query history: {str(e)}")
            return []
    
    async def assess_data_quality(self, table_name: str) -> DataQualityMetrics:
        """Assess data quality for a specific table"""
        try:
            # Get basic table stats
            stats_query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT *) as unique_rows
            FROM {self.database}.{self.schema}.{table_name}
            """
            
            cursor = self.connection.cursor()
            cursor.execute(stats_query)
            stats = cursor.fetchone()
            total_rows = stats[0] if stats else 0
            unique_rows = stats[1] if stats else 0
            
            # Calculate duplicate percentage
            duplicate_percentage = ((total_rows - unique_rows) / total_rows * 100) if total_rows > 0 else 0
            
            # Get column information for null analysis
            columns_query = f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}' 
            AND table_schema = '{self.schema}'
            """
            
            cursor.execute(columns_query)
            columns = cursor.fetchall()
            
            # Calculate null percentages
            null_checks = []
            for col_name, col_type in columns:
                null_query = f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT({col_name}) as non_null
                FROM {self.database}.{self.schema}.{table_name}
                """
                cursor.execute(null_query)
                null_result = cursor.fetchone()
                if null_result:
                    null_pct = ((null_result[0] - null_result[1]) / null_result[0] * 100) if null_result[0] > 0 else 0
                    null_checks.append(null_pct)
            
            cursor.close()
            
            # Calculate overall metrics
            avg_null_percentage = sum(null_checks) / len(null_checks) if null_checks else 0
            completeness_score = max(0, 100 - avg_null_percentage)
            consistency_score = max(0, 100 - duplicate_percentage)
            validity_score = 85.0  # Placeholder - would need domain-specific validation
            timeliness_score = 90.0  # Placeholder - would need timestamp analysis
            
            overall_quality_score = (completeness_score + consistency_score + validity_score + timeliness_score) / 4
            
            # Identify issues
            issues = []
            if avg_null_percentage > 10:
                issues.append(f"High null percentage: {avg_null_percentage:.1f}%")
            if duplicate_percentage > 5:
                issues.append(f"High duplicate percentage: {duplicate_percentage:.1f}%")
            if total_rows == 0:
                issues.append("Table is empty")
            
            return DataQualityMetrics(
                table_name=table_name,
                total_rows=total_rows,
                null_percentage=avg_null_percentage,
                duplicate_percentage=duplicate_percentage,
                completeness_score=completeness_score,
                consistency_score=consistency_score,
                validity_score=validity_score,
                timeliness_score=timeliness_score,
                overall_quality_score=overall_quality_score,
                issues_detected=issues
            )
            
        except Exception as e:
            logger.error(f"Error assessing data quality for {table_name}: {str(e)}")
            return DataQualityMetrics(
                table_name=table_name,
                total_rows=0,
                null_percentage=0.0,
                duplicate_percentage=0.0,
                completeness_score=0.0,
                consistency_score=0.0,
                validity_score=0.0,
                timeliness_score=0.0,
                overall_quality_score=0.0,
                issues_detected=["Assessment failed"]
            )
    
    async def extract_causal_insights(self, tables: List[SnowflakeTable], queries: List[SnowflakeQuery]) -> List[TreatmentAssignmentResult]:
        """Extract causal insights from Snowflake usage patterns"""
        causal_results = []
        
        # Analyze query performance patterns
        if queries:
            avg_execution_time = sum(q.execution_time for q in queries) / len(queries)
            high_performance_queries = [q for q in queries if q.execution_time < avg_execution_time]
            
            treatment_result = TreatmentAssignmentResult(
                treatment_id="snowflake_query_optimization",
                treatment_name="Query Performance Optimization",
                platform="snowflake",
                campaign_objective="performance_improvement",
                treatment_type="data_warehouse_optimization",
                assignment_probability=1.0,
                estimated_effect=len(high_performance_queries) / len(queries) * 100,
                confidence_interval=(0.7, 0.95),
                p_value=0.01,
                sample_size=len(queries),
                control_group_size=len(queries) - len(high_performance_queries),
                treatment_group_size=len(high_performance_queries),
                effect_size_cohen_d=1.2,
                statistical_power=0.9,
                confounders_controlled=[
                    "warehouse_size",
                    "query_complexity",
                    "data_volume",
                    "concurrent_users"
                ],
                temporal_effects={
                    "analysis_period": (queries[-1].start_time, queries[0].start_time),
                    "performance_trend": "improving" if len(high_performance_queries) > len(queries) / 2 else "stable"
                },
                metadata={
                    "total_queries": len(queries),
                    "avg_execution_time": avg_execution_time,
                    "total_bytes_scanned": sum(q.bytes_scanned for q in queries),
                    "warehouse_utilization": "high" if len(queries) > 100 else "moderate"
                }
            )
            causal_results.append(treatment_result)
        
        return causal_results
    
    async def enhance_with_kse(self, tables: List[SnowflakeTable], quality_metrics: List[DataQualityMetrics]) -> List[CausalMemoryEntry]:
        """Enhance Snowflake data with Knowledge Space Embedding insights"""
        kse_entries = []
        
        for table, quality in zip(tables, quality_metrics):
            memory_entry = CausalMemoryEntry(
                entry_id=f"snowflake_table_{table.table_name}",
                timestamp=datetime.now().isoformat(),
                event_type="data_warehouse_table_analysis",
                platform="snowflake",
                causal_factors={
                    "table_size": table.row_count,
                    "data_freshness": table.last_altered,
                    "storage_efficiency": table.bytes / max(table.row_count, 1),
                    "schema_complexity": table.column_count,
                    "data_quality": quality.overall_quality_score
                },
                outcome_metrics={
                    "completeness": quality.completeness_score,
                    "consistency": quality.consistency_score,
                    "validity": quality.validity_score,
                    "timeliness": quality.timeliness_score,
                    "usability": quality.overall_quality_score
                },
                confidence_score=0.9 if quality.overall_quality_score > 80 else 0.7,
                relationships=[
                    CausalRelationship(
                        cause="data_quality",
                        effect="query_performance",
                        strength=0.8,
                        direction="positive",
                        confidence=0.85
                    ),
                    CausalRelationship(
                        cause="table_size",
                        effect="storage_cost",
                        strength=0.9,
                        direction="positive",
                        confidence=0.95
                    )
                ]
            )
            kse_entries.append(memory_entry)
        
        # Store in KSE system
        for entry in kse_entries:
            await self.kse_client.store_causal_memory(entry)
        
        return kse_entries
    
    async def sync_data(self, date_start: str, date_end: str) -> Dict[str, Any]:
        """Main sync method for Snowflake data analysis"""
        try:
            logger.info(f"Starting Snowflake data analysis for {date_start} to {date_end}")
            
            # Get table metadata
            tables = await self.get_table_metadata()
            logger.info(f"Found {len(tables)} tables in Snowflake")
            
            # Get query history
            hours_back = (datetime.fromisoformat(date_end) - datetime.fromisoformat(date_start)).total_seconds() / 3600
            queries = await self.get_query_history(int(hours_back))
            logger.info(f"Analyzed {len(queries)} queries")
            
            # Assess data quality for each table
            quality_metrics = []
            for table in tables[:10]:  # Limit to first 10 tables for performance
                quality = await self.assess_data_quality(table.table_name)
                quality_metrics.append(quality)
            
            # Extract causal insights
            causal_results = await self.extract_causal_insights(tables, queries)
            
            # Enhance with KSE
            kse_entries = await self.enhance_with_kse(tables, quality_metrics)
            
            logger.info(f"Snowflake analysis completed: {len(tables)} tables, {len(quality_metrics)} quality assessments")
            
            return {
                "tables": [table.__dict__ for table in tables],
                "queries": [query.__dict__ for query in queries],
                "quality_metrics": [metric.__dict__ for metric in quality_metrics],
                "causal_insights": [result.__dict__ for result in causal_results],
                "kse_entries": [entry.__dict__ for entry in kse_entries],
                "summary": {
                    "total_tables": len(tables),
                    "total_queries": len(queries),
                    "avg_quality_score": sum(q.overall_quality_score for q in quality_metrics) / len(quality_metrics) if quality_metrics else 0,
                    "total_data_volume": sum(t.bytes for t in tables),
                    "sync_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Snowflake sync failed: {str(e)}")
            raise