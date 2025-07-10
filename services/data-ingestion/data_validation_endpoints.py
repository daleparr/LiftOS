"""
Data Validation API Endpoints
Provides REST API for data source validation and quality monitoring
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
import logging

from shared.models.platform_connections import (
    DataQualityReportResponse,
    ValidationResultResponse
)
from shared.services.data_source_validator import DataQualityLevel
from shared.services.data_source_validator import get_data_source_validator
from shared.auth.jwt_auth import get_current_user, require_permissions
from shared.utils.logging import setup_logging

logger = setup_logging("data_validation_endpoints")

router = APIRouter(prefix="/api/v1/data-validation", tags=["data-validation"])

@router.get("/connections/{connection_id}/validate", response_model=DataQualityReportResponse)
async def validate_connection(
    connection_id: str,
    current_user: dict = Depends(get_current_user),
    validator = Depends(get_data_source_validator)
):
    """Validate a specific platform connection"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate connection access
        await require_permissions(current_user, ["data:read"])
        
        # Run validation
        report = await validator.validate_connection(user_id, org_id, connection_id)
        
        # Convert to response model
        validation_results = [
            ValidationResultResponse(
                rule_id=result.rule_id,
                status=result.status.value,
                score=result.score,
                message=result.message,
                details=result.details,
                recommendations=result.recommendations
            )
            for result in report.validation_results
        ]
        
        return DataQualityReportResponse(
            connection_id=report.connection_id,
            platform=report.platform,
            overall_score=report.overall_score,
            quality_level=report.quality_level.value,
            validation_results=validation_results,
            data_freshness=report.data_freshness,
            completeness_metrics=report.completeness_metrics,
            consistency_metrics=report.consistency_metrics,
            reliability_metrics=report.reliability_metrics,
            recommendations=report.recommendations,
            generated_at=report.generated_at
        )
        
    except ValueError as e:
        logger.error(f"Validation error for connection {connection_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to validate connection {connection_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Validation failed")

@router.get("/connections/validate-all", response_model=List[DataQualityReportResponse])
async def validate_all_connections(
    current_user: dict = Depends(get_current_user),
    validator = Depends(get_data_source_validator)
):
    """Validate all user connections"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        # Run validation for all connections
        reports = await validator.validate_all_connections(user_id, org_id)
        
        # Convert to response models
        response_reports = []
        for report in reports:
            validation_results = [
                ValidationResultResponse(
                    rule_id=result.rule_id,
                    status=result.status.value,
                    score=result.score,
                    message=result.message,
                    details=result.details,
                    recommendations=result.recommendations
                )
                for result in report.validation_results
            ]
            
            response_reports.append(DataQualityReportResponse(
                connection_id=report.connection_id,
                platform=report.platform,
                overall_score=report.overall_score,
                quality_level=report.quality_level.value,
                validation_results=validation_results,
                data_freshness=report.data_freshness,
                completeness_metrics=report.completeness_metrics,
                consistency_metrics=report.consistency_metrics,
                reliability_metrics=report.reliability_metrics,
                recommendations=report.recommendations,
                generated_at=report.generated_at
            ))
        
        return response_reports
        
    except Exception as e:
        logger.error(f"Failed to validate all connections: {str(e)}")
        raise HTTPException(status_code=500, detail="Validation failed")

@router.post("/connections/{connection_id}/validate-async")
async def validate_connection_async(
    connection_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    validator = Depends(get_data_source_validator)
):
    """Start asynchronous validation of a connection"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        # Add validation task to background
        background_tasks.add_task(
            _run_background_validation,
            validator,
            user_id,
            org_id,
            connection_id
        )
        
        return {
            "message": "Validation started",
            "connection_id": connection_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to start async validation for {connection_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start validation")

@router.get("/quality-summary")
async def get_quality_summary(
    current_user: dict = Depends(get_current_user),
    validator = Depends(get_data_source_validator)
):
    """Get overall data quality summary for user"""
    try:
        user_id = current_user["user_id"]
        org_id = current_user["org_id"]
        
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        # Get all validation reports
        reports = await validator.validate_all_connections(user_id, org_id)
        
        if not reports:
            return {
                "total_connections": 0,
                "average_score": 0.0,
                "quality_distribution": {},
                "top_issues": [],
                "recommendations": []
            }
        
        # Calculate summary metrics
        total_connections = len(reports)
        average_score = sum(report.overall_score for report in reports) / total_connections
        
        # Quality level distribution
        quality_distribution = {}
        for report in reports:
            level = report.quality_level.value
            quality_distribution[level] = quality_distribution.get(level, 0) + 1
        
        # Top issues (most common failed validations)
        issue_counts = {}
        all_recommendations = []
        
        for report in reports:
            all_recommendations.extend(report.recommendations)
            for result in report.validation_results:
                if result.status.value in ["failed", "warning"]:
                    issue_counts[result.rule_id] = issue_counts.get(result.rule_id, 0) + 1
        
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Most common recommendations
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        top_recommendations = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_connections": total_connections,
            "average_score": round(average_score, 1),
            "quality_distribution": quality_distribution,
            "top_issues": [{"rule_id": rule_id, "count": count} for rule_id, count in top_issues],
            "recommendations": [rec for rec, count in top_recommendations]
        }
        
    except Exception as e:
        logger.error(f"Failed to get quality summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get quality summary")

@router.get("/validation-rules")
async def get_validation_rules(
    platform: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    validator = Depends(get_data_source_validator)
):
    """Get available validation rules"""
    try:
        # Validate permissions
        await require_permissions(current_user, ["data:read"])
        
        rules = validator.validation_rules
        
        # Filter by platform if specified
        if platform:
            rules = [
                rule for rule in rules
                if not rule.platform_specific or platform in (rule.applicable_platforms or [])
            ]
        
        return [
            {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "severity": rule.severity,
                "platform_specific": rule.platform_specific,
                "applicable_platforms": rule.applicable_platforms
            }
            for rule in rules
        ]
        
    except Exception as e:
        logger.error(f"Failed to get validation rules: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get validation rules")

async def _run_background_validation(validator, user_id: str, org_id: str, connection_id: str):
    """Run validation in background"""
    try:
        report = await validator.validate_connection(user_id, org_id, connection_id)
        logger.info(f"Background validation completed for connection {connection_id}")
        
        # Here you could store the report in a cache or database
        # and/or send notifications to the user
        
    except Exception as e:
        logger.error(f"Background validation failed for connection {connection_id}: {str(e)}")