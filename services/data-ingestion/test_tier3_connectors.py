"""
Test Suite for Tier 3 Connectors (Social/Analytics/Data)
Tests TikTok, Snowflake, and Databricks connectors
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

# Import shared modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from connectors.tiktok_connector import TikTokConnector
from connectors.snowflake_connector import SnowflakeConnector
from connectors.databricks_connector import DatabricksConnector


class TestTikTokConnector:
    """Test TikTok for Business connector"""
    
    @pytest.fixture
    def tiktok_credentials(self):
        return {
            "access_token": "test_access_token",
            "app_id": "test_app_id",
            "app_secret": "test_app_secret"
        }
    
    @pytest.fixture
    def mock_tiktok_response(self):
        return {
            "code": 0,
            "data": {
                "list": [
                    {
                        "campaign_id": "123456789",
                        "campaign_name": "Test Campaign",
                        "objective_type": "CONVERSIONS",
                        "status": "ENABLE",
                        "budget": 1000.0,
                        "create_time": "2025-01-01T00:00:00Z",
                        "modify_time": "2025-01-07T00:00:00Z"
                    }
                ]
            }
        }
    
    @pytest.mark.asyncio
    async def test_tiktok_connector_initialization(self, tiktok_credentials):
        """Test TikTok connector initialization"""
        connector = TikTokConnector(tiktok_credentials)
        assert connector.access_token == "test_access_token"
        assert connector.app_id == "test_app_id"
        assert connector.app_secret == "test_app_secret"
        assert connector.base_url == "https://business-api.tiktok.com/open_api/v1.3"
    
    @pytest.mark.asyncio
    async def test_tiktok_connector_invalid_credentials(self):
        """Test TikTok connector with invalid credentials"""
        with pytest.raises(ValueError, match="Missing required TikTok credentials"):
            TikTokConnector({"access_token": "test"})
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_tiktok_test_connection(self, mock_get, tiktok_credentials, mock_tiktok_response):
        """Test TikTok connection test"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_tiktok_response
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with TikTokConnector(tiktok_credentials) as connector:
            result = await connector.test_connection()
            assert result is True
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_tiktok_get_campaigns(self, mock_get, tiktok_credentials, mock_tiktok_response):
        """Test TikTok campaign retrieval"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_tiktok_response
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with TikTokConnector(tiktok_credentials) as connector:
            campaigns = await connector.get_campaigns("test_advertiser", "2025-01-01", "2025-01-07")
            assert len(campaigns) == 1
            assert campaigns[0].campaign_id == "123456789"
            assert campaigns[0].campaign_name == "Test Campaign"


class TestSnowflakeConnector:
    """Test Snowflake Data Warehouse connector"""
    
    @pytest.fixture
    def snowflake_credentials(self):
        return {
            "account": "test_account",
            "username": "test_user",
            "password": "test_password",
            "warehouse": "test_warehouse",
            "database": "test_database",
            "schema": "test_schema"
        }
    
    @pytest.fixture
    def mock_snowflake_tables(self):
        return [
            ("customers", "public", "test_db", 1000, 5, "BASE TABLE", 
             "2025-01-01", "2025-01-07", 50000, 1),
            ("orders", "public", "test_db", 5000, 8, "BASE TABLE",
             "2025-01-01", "2025-01-07", 250000, 1)
        ]
    
    @pytest.mark.asyncio
    async def test_snowflake_connector_initialization(self, snowflake_credentials):
        """Test Snowflake connector initialization"""
        connector = SnowflakeConnector(snowflake_credentials)
        assert connector.account == "test_account"
        assert connector.username == "test_user"
        assert connector.database == "test_database"
        assert connector.schema == "test_schema"
    
    @pytest.mark.asyncio
    async def test_snowflake_connector_invalid_credentials(self):
        """Test Snowflake connector with invalid credentials"""
        with pytest.raises(ValueError, match="Missing required Snowflake credentials"):
            SnowflakeConnector({"account": "test"})
    
    @pytest.mark.asyncio
    @patch('snowflake.connector.connect')
    async def test_snowflake_connect(self, mock_connect, snowflake_credentials):
        """Test Snowflake connection"""
        mock_connection = Mock()
        mock_connect.return_value = mock_connection
        
        connector = SnowflakeConnector(snowflake_credentials)
        await connector.connect()
        
        assert connector.connection == mock_connection
        mock_connect.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('snowflake.connector.connect')
    async def test_snowflake_get_table_metadata(self, mock_connect, snowflake_credentials, mock_snowflake_tables):
        """Test Snowflake table metadata retrieval"""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = mock_snowflake_tables
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        connector = SnowflakeConnector(snowflake_credentials)
        await connector.connect()
        
        tables = await connector.get_table_metadata()
        assert len(tables) == 2
        assert tables[0].table_name == "customers"
        assert tables[0].row_count == 1000
        assert tables[1].table_name == "orders"
        assert tables[1].row_count == 5000


class TestDatabricksConnector:
    """Test Databricks Analytics Platform connector"""
    
    @pytest.fixture
    def databricks_credentials(self):
        return {
            "host": "test-workspace.cloud.databricks.com",
            "token": "test_token",
            "cluster_id": "test_cluster_id"
        }
    
    @pytest.fixture
    def mock_databricks_clusters(self):
        return {
            "clusters": [
                {
                    "cluster_id": "test_cluster_id",
                    "cluster_name": "Test Cluster",
                    "spark_version": "11.3.x-scala2.12",
                    "node_type_id": "i3.xlarge",
                    "driver_node_type_id": "i3.xlarge",
                    "num_workers": 2,
                    "autoscale": {},
                    "state": "RUNNING",
                    "state_message": "",
                    "start_time": 1640995200000,
                    "creator_user_name": "test@example.com",
                    "cluster_source": "UI",
                    "disk_spec": {},
                    "cluster_log_conf": {}
                }
            ]
        }
    
    @pytest.fixture
    def mock_databricks_jobs(self):
        return {
            "runs": [
                {
                    "job_id": 123,
                    "run_id": 456,
                    "run_name": "Test Job",
                    "state": {
                        "life_cycle_state": "TERMINATED",
                        "result_state": "SUCCESS"
                    },
                    "start_time": 1640995200000,
                    "end_time": 1640995800000,
                    "setup_duration": 30000,
                    "execution_duration": 570000,
                    "cleanup_duration": 10000,
                    "cluster_instance": {},
                    "creator_user_name": "test@example.com",
                    "run_type": "JOB_RUN",
                    "task": {"task_key": "main_task"}
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_databricks_connector_initialization(self, databricks_credentials):
        """Test Databricks connector initialization"""
        connector = DatabricksConnector(databricks_credentials)
        assert connector.host == "https://test-workspace.cloud.databricks.com"
        assert connector.token == "test_token"
        assert connector.cluster_id == "test_cluster_id"
        assert connector.base_url == "https://test-workspace.cloud.databricks.com/api/2.0"
    
    @pytest.mark.asyncio
    async def test_databricks_connector_invalid_credentials(self):
        """Test Databricks connector with invalid credentials"""
        with pytest.raises(ValueError, match="Missing required Databricks credentials"):
            DatabricksConnector({"host": "test"})
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_databricks_test_connection(self, mock_get, databricks_credentials):
        """Test Databricks connection test"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with DatabricksConnector(databricks_credentials) as connector:
            result = await connector.test_connection()
            assert result is True
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_databricks_get_clusters(self, mock_get, databricks_credentials, mock_databricks_clusters):
        """Test Databricks cluster retrieval"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_databricks_clusters
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with DatabricksConnector(databricks_credentials) as connector:
            clusters = await connector.get_clusters()
            assert len(clusters) == 1
            assert clusters[0].cluster_id == "test_cluster_id"
            assert clusters[0].cluster_name == "Test Cluster"
            assert clusters[0].state == "RUNNING"
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_databricks_get_jobs(self, mock_get, databricks_credentials, mock_databricks_jobs):
        """Test Databricks job retrieval"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_databricks_jobs
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with DatabricksConnector(databricks_credentials) as connector:
            jobs = await connector.get_jobs()
            assert len(jobs) == 1
            assert jobs[0].job_id == 123
            assert jobs[0].run_id == 456
            assert jobs[0].result_state == "SUCCESS"


class TestTier3Integration:
    """Integration tests for Tier 3 connectors"""
    
    @pytest.mark.asyncio
    async def test_all_tier3_connectors_initialization(self):
        """Test that all Tier 3 connectors can be initialized"""
        tiktok_creds = {
            "access_token": "test", "app_id": "test", "app_secret": "test"
        }
        snowflake_creds = {
            "account": "test", "username": "test", "password": "test",
            "warehouse": "test", "database": "test", "schema": "test"
        }
        databricks_creds = {
            "host": "test", "token": "test", "cluster_id": "test"
        }
        
        # Test initialization without errors
        tiktok_connector = TikTokConnector(tiktok_creds)
        snowflake_connector = SnowflakeConnector(snowflake_creds)
        databricks_connector = DatabricksConnector(databricks_creds)
        
        assert tiktok_connector is not None
        assert snowflake_connector is not None
        assert databricks_connector is not None
    
    @pytest.mark.asyncio
    async def test_tier3_causal_analysis_integration(self):
        """Test that Tier 3 connectors support causal analysis"""
        tiktok_creds = {
            "access_token": "test", "app_id": "test", "app_secret": "test"
        }
        
        connector = TikTokConnector(tiktok_creds)
        
        # Test that causal analysis methods exist
        assert hasattr(connector, 'extract_causal_marketing_data')
        assert hasattr(connector, 'enhance_with_kse')
        assert hasattr(connector, 'sync_data')


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(TestTier3Integration().test_all_tier3_connectors_initialization())
    print("[SUCCESS] All Tier 3 connectors initialized successfully")
    
    asyncio.run(TestTier3Integration().test_tier3_causal_analysis_integration())
    print("[SUCCESS] Tier 3 connectors support causal analysis")
    
    print("\n[COMPLETE] Tier 3 connectors test suite completed successfully!")