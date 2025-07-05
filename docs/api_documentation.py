#!/usr/bin/env python3
"""
API Documentation Generator
Generates comprehensive OpenAPI/Swagger documentation for all services
"""

import json
import yaml
from typing import Dict, Any, List
import os

class LiftOSAPIDocumentationGenerator:
    """Generate comprehensive API documentation for Lift OS Core"""
    
    def __init__(self):
        self.base_info = {
            "openapi": "3.0.3",
            "info": {
                "title": "Lift OS Core API",
                "description": "Unified API documentation for the Lift OS Core platform - a modular, scalable operating system for orchestrating all Lift products",
                "version": "1.0.0",
                "contact": {
                    "name": "Lift OS Core Team",
                    "email": "support@liftos.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development Gateway Server"
                },
                {
                    "url": "https://api.liftos.com",
                    "description": "Production Gateway Server"
                }
            ]
        }
    
    def generate_complete_api_spec(self) -> Dict[str, Any]:
        """Generate complete API specification for all services"""
        spec = self.base_info.copy()
        
        # Combine all service paths
        all_paths = {}
        all_paths.update(self.generate_gateway_paths())
        all_paths.update(self.generate_auth_paths())
        all_paths.update(self.generate_memory_paths())
        all_paths.update(self.generate_registry_paths())
        
        spec["paths"] = all_paths
        spec.update(self.generate_schemas())
        
        return spec
    
    def generate_gateway_paths(self) -> Dict[str, Any]:
        """Generate Gateway service paths"""
        return {
            "/health": {
                "get": {
                    "tags": ["Gateway Health"],
                    "summary": "Gateway Health Check",
                    "description": "Check the health status of the Gateway service",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/ready": {
                "get": {
                    "tags": ["Gateway Health"],
                    "summary": "Gateway Readiness Check",
                    "description": "Check if the Gateway service is ready to handle requests",
                    "responses": {
                        "200": {
                            "description": "Service is ready",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ReadinessResponse"}
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def generate_auth_paths(self) -> Dict[str, Any]:
        """Generate Authentication service paths"""
        return {
            "/auth/health": {
                "get": {
                    "tags": ["Auth Health"],
                    "summary": "Auth Service Health Check",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/auth/register": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "User Registration",
                    "description": "Register a new user in the system",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/UserRegistration"}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "User registered successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UserResponse"}
                                }
                            }
                        },
                        "400": {"description": "Invalid registration data"},
                        "409": {"description": "User already exists"}
                    }
                }
            },
            "/auth/login": {
                "post": {
                    "tags": ["Authentication"],
                    "summary": "User Login",
                    "description": "Authenticate user and return JWT token",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/UserLogin"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Login successful",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/LoginResponse"}
                                }
                            }
                        },
                        "401": {"description": "Invalid credentials"}
                    }
                }
            }
        }
    
    def generate_memory_paths(self) -> Dict[str, Any]:
        """Generate Memory service paths"""
        return {
            "/memory/health": {
                "get": {
                    "tags": ["Memory Health"],
                    "summary": "Memory Service Health Check",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/memory/store": {
                "post": {
                    "tags": ["Memory Operations"],
                    "summary": "Store Memory",
                    "description": "Store a memory item in the Knowledge Storage Engine",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MemoryStoreRequest"}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Memory stored successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MemoryStoreResponse"}
                                }
                            }
                        },
                        "400": {"description": "Invalid memory data"},
                        "401": {"description": "Unauthorized"}
                    }
                }
            },
            "/memory/search": {
                "post": {
                    "tags": ["KSE Operations"],
                    "summary": "Semantic Search",
                    "description": "Perform semantic search using the Knowledge Storage Engine",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SearchRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Search completed successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SearchResponse"}
                                }
                            }
                        },
                        "400": {"description": "Invalid search query"},
                        "401": {"description": "Unauthorized"}
                    }
                }
            }
        }
    
    def generate_registry_paths(self) -> Dict[str, Any]:
        """Generate Registry service paths"""
        return {
            "/registry/health": {
                "get": {
                    "tags": ["Registry Health"],
                    "summary": "Registry Service Health Check",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/registry/modules": {
                "get": {
                    "tags": ["Module Management"],
                    "summary": "List Modules",
                    "description": "Get list of all registered modules",
                    "security": [{"BearerAuth": []}],
                    "responses": {
                        "200": {
                            "description": "Modules retrieved successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ModuleListResponse"}
                                }
                            }
                        },
                        "401": {"description": "Unauthorized"}
                    }
                },
                "post": {
                    "tags": ["Module Management"],
                    "summary": "Register Module",
                    "description": "Register a new module in the system",
                    "security": [{"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ModuleRegistration"}
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "description": "Module registered successfully",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/ModuleResponse"}
                                }
                            }
                        },
                        "400": {"description": "Invalid module data"},
                        "401": {"description": "Unauthorized"},
                        "409": {"description": "Module already exists"}
                    }
                }
            }
        }
    
    def generate_schemas(self) -> Dict[str, Any]:
        """Generate common schemas for API documentation"""
        return {
            "components": {
                "securitySchemes": {
                    "BearerAuth": {
                        "type": "http",
                        "scheme": "bearer",
                        "bearerFormat": "JWT"
                    }
                },
                "schemas": {
                    "HealthResponse": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "example": "healthy"},
                            "service": {"type": "string", "example": "gateway"},
                            "timestamp": {"type": "string", "format": "date-time"},
                            "uptime": {"type": "number", "example": 3600.5},
                            "version": {"type": "string", "example": "1.0.0"}
                        },
                        "required": ["status", "service", "timestamp"]
                    },
                    "ReadinessResponse": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "example": "ready"},
                            "service": {"type": "string", "example": "gateway"},
                            "timestamp": {"type": "string", "format": "date-time"},
                            "checks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "status": {"type": "string"},
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        },
                        "required": ["status", "service", "timestamp", "checks"]
                    },
                    "UserRegistration": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "format": "email", "example": "user@example.com"},
                            "password": {"type": "string", "minLength": 8, "example": "securepassword123"},
                            "org_id": {"type": "string", "example": "org_123"}
                        },
                        "required": ["email", "password", "org_id"]
                    },
                    "UserLogin": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "format": "email", "example": "user@example.com"},
                            "password": {"type": "string", "example": "securepassword123"}
                        },
                        "required": ["email", "password"]
                    },
                    "LoginResponse": {
                        "type": "object",
                        "properties": {
                            "access_token": {"type": "string", "example": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."},
                            "token_type": {"type": "string", "example": "bearer"},
                            "expires_in": {"type": "integer", "example": 3600},
                            "user": {"$ref": "#/components/schemas/UserResponse"}
                        },
                        "required": ["access_token", "token_type", "expires_in", "user"]
                    },
                    "UserResponse": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "example": "user_123"},
                            "email": {"type": "string", "example": "user@example.com"},
                            "org_id": {"type": "string", "example": "org_123"},
                            "created_at": {"type": "string", "format": "date-time"}
                        },
                        "required": ["user_id", "email", "org_id"]
                    },
                    "MemoryStoreRequest": {
                        "type": "object",
                        "properties": {
                            "key": {"type": "string", "example": "user_preference_theme"},
                            "value": {
                                "type": "object",
                                "example": {"theme": "dark", "language": "en"}
                            },
                            "context": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string", "example": "user_123"},
                                    "org_id": {"type": "string", "example": "org_123"}
                                }
                            }
                        },
                        "required": ["key", "value", "context"]
                    },
                    "MemoryStoreResponse": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "example": "mem_123456"},
                            "key": {"type": "string", "example": "user_preference_theme"},
                            "stored_at": {"type": "string", "format": "date-time"}
                        },
                        "required": ["memory_id", "key", "stored_at"]
                    },
                    "SearchRequest": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "example": "user preferences for dark theme"},
                            "context": {
                                "type": "object",
                                "properties": {
                                    "user_id": {"type": "string"},
                                    "org_id": {"type": "string"}
                                }
                            },
                            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10}
                        },
                        "required": ["query", "context"]
                    },
                    "SearchResponse": {
                        "type": "object",
                        "properties": {
                            "results": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "memory_id": {"type": "string"},
                                        "score": {"type": "number", "minimum": 0, "maximum": 1}
                                    }
                                }
                            },
                            "total": {"type": "integer", "example": 25},
                            "query_time": {"type": "number", "example": 0.045}
                        },
                        "required": ["results", "total", "query_time"]
                    },
                    "ModuleRegistration": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "example": "analytics-module"},
                            "version": {"type": "string", "example": "1.2.0"},
                            "endpoint": {"type": "string", "format": "uri", "example": "http://localhost:9001"}
                        },
                        "required": ["name", "version", "endpoint"]
                    },
                    "ModuleResponse": {
                        "type": "object",
                        "properties": {
                            "module_id": {"type": "string", "example": "mod_123456"},
                            "name": {"type": "string", "example": "analytics-module"},
                            "version": {"type": "string", "example": "1.2.0"},
                            "status": {"type": "string", "example": "active"}
                        },
                        "required": ["module_id", "name", "version", "status"]
                    },
                    "ModuleListResponse": {
                        "type": "object",
                        "properties": {
                            "modules": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/ModuleResponse"}
                            },
                            "total": {"type": "integer", "example": 5}
                        },
                        "required": ["modules", "total"]
                    }
                }
            }
        }
    
    def save_documentation(self, output_dir: str = "docs"):
        """Save API documentation in multiple formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate complete specification
        spec = self.generate_complete_api_spec()
        
        # Save as JSON
        with open(os.path.join(output_dir, "api_spec.json"), "w") as f:
            json.dump(spec, f, indent=2)
        
        # Save as YAML
        with open(os.path.join(output_dir, "api_spec.yaml"), "w") as f:
            yaml.dump(spec, f, default_flow_style=False, indent=2)
        
        print(f"API documentation saved to {output_dir}/")
        print(f"- JSON: {output_dir}/api_spec.json")
        print(f"- YAML: {output_dir}/api_spec.yaml")

if __name__ == "__main__":
    generator = LiftOSAPIDocumentationGenerator()
    generator.save_documentation()