"""Configuration settings for the Memory Store backend."""

from pydantic import BaseModel, Field, SecretStr


class MemoryStoreConfig(BaseModel):
    """OpenSearch Graph Memory Store connection configuration.

    Used when ``backend="memory_store"`` is selected in MemorySettings.
    """

    endpoint: str = Field(
        default="https://localhost:9200",
        description="Memory Store REST endpoint URL",
    )
    database: str = Field(
        default="memory",
        description="Database / index name within the Memory Store",
    )
    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenant isolation",
    )
    user_id: str = Field(
        default="default",
        description="User identifier scoping memory operations",
    )
    auth_token: SecretStr | None = Field(
        default=None,
        description="Bearer token for Memory Store authentication",
    )
    username: str | None = Field(
        default=None,
        description="Username for basic authentication",
    )
    password: SecretStr | None = Field(
        default=None,
        description="Password for basic authentication",
    )
    connect_timeout: float = Field(
        default=10.0,
        gt=0,
        description="Connection timeout in seconds",
    )
    read_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Read timeout in seconds",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Whether to verify SSL certificates",
    )
