"""Configuration management for Financial Services Advisor."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemoryStoreSettings(BaseSettings):
    """Memory Store (OpenSearch) configuration."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_STORE_")

    endpoint: str = Field(
        default="https://localhost:9200", description="Memory Store REST endpoint URL"
    )


class BedrockSettings(BaseSettings):
    """Amazon Bedrock configuration."""

    model_config = SettingsConfigDict(env_prefix="BEDROCK_")

    model_id: str = Field(
        default="anthropic.claude-sonnet-4-20250514-v1:0",
        description="Bedrock model ID for LLM",
    )
    embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Bedrock model ID for embeddings",
    )
    region: str = Field(default="us-west-2", description="AWS region for Bedrock")


class AWSSettings(BaseSettings):
    """AWS general configuration."""

    model_config = SettingsConfigDict(env_prefix="AWS_")

    region: str = Field(default="us-west-2", description="AWS region")
    profile: str | None = Field(default=None, description="AWS profile name")
    access_key_id: str | None = Field(default=None, description="AWS access key ID")
    secret_access_key: str | None = Field(
        default=None, description="AWS secret access key"
    )


class CognitoSettings(BaseSettings):
    """Amazon Cognito configuration."""

    model_config = SettingsConfigDict(env_prefix="COGNITO_")

    user_pool_id: str | None = Field(default=None, description="Cognito User Pool ID")
    client_id: str | None = Field(default=None, description="Cognito Client ID")


class S3Settings(BaseSettings):
    """Amazon S3 configuration."""

    model_config = SettingsConfigDict(env_prefix="S3_")

    bucket_name: str = Field(
        default="financial-advisor-documents",
        description="S3 bucket for document storage",
    )
    region: str = Field(default="us-west-2", description="S3 bucket region")


class AppSettings(BaseSettings):
    """Application-level settings."""

    log_level: str = Field(default="INFO", description="Logging level")
    cors_origins: str = Field(
        default="http://localhost:5173,http://localhost:3000",
        description="Comma-separated list of allowed CORS origins",
    )
    debug: bool = Field(default=False, description="Enable debug mode")

    # Feature flags
    enable_sanctions_check: bool = Field(
        default=True, description="Enable sanctions checking"
    )
    enable_pep_check: bool = Field(default=True, description="Enable PEP checking")
    enable_adverse_media: bool = Field(
        default=True, description="Enable adverse media screening"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


class Settings(BaseSettings):
    """Main settings container."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    memory_store: MemoryStoreSettings = Field(default_factory=MemoryStoreSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)
    aws: AWSSettings = Field(default_factory=AWSSettings)
    cognito: CognitoSettings = Field(default_factory=CognitoSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    app: AppSettings = Field(default_factory=AppSettings)

    def to_strands_config_dict(self) -> dict[str, Any]:
        """Convert settings to Strands integration config format."""
        return {
            "backend": "memory_store",
            "memory_store_endpoint": self.memory_store.endpoint,
            "embedding_provider": "bedrock",
            "embedding_model": self.bedrock.embedding_model_id,
            "aws_region": self.aws.region,
        }

    def to_memory_settings_dict(self) -> dict[str, Any]:
        """Convert settings to MemorySettings format."""
        return {
            "backend": "memory_store",
            "memory_store": {
                "endpoint": self.memory_store.endpoint,
            },
            "embedding": {
                "provider": "bedrock",
                "model": self.bedrock.embedding_model_id,
                "aws_region": self.aws.region,
            },
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
