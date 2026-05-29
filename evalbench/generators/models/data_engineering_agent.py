import asyncio
import logging
from typing import Any

from a2a.client import ClientCallContext
from a2a.client.auth import AuthInterceptor, CredentialService
import google.auth
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import Request

from .generator import QueryGenerator


class GcpAdcCredentialService(CredentialService):
    """GCP Application Default Credentials (ADC) service for A2A SDK.

    This provider only services OAuth/OAuth2 schemes.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.credentials = None
        self._lock = None

    async def get_credentials(
        self,
        security_scheme_name: str,
        context: ClientCallContext | None,
    ) -> str:
        if security_scheme_name.lower() not in ("oauth", "oauth2"):
            raise ValueError(
                f"GcpAdcCredentialService only services 'oauth' or 'oauth2' "
                f"schemes, got '{security_scheme_name}'"
            )

        if self._lock is None:
            self._lock = asyncio.Lock()

        try:
            async with self._lock:
                if self.credentials is None:
                    credentials, _ = await asyncio.to_thread(
                        google.auth.default,
                        scopes=[
                            "https://www.googleapis.com/auth/cloud-platform"
                        ]
                    )
                    self.credentials = credentials

                if not self.credentials.valid:
                    await asyncio.to_thread(
                        self.credentials.refresh, Request()
                    )

                self.logger.debug("Retrieved GCP ADC token successfully.")
                return self.credentials.token

        except (DefaultCredentialsError, RefreshError) as e:
            self.logger.error(
                "Failed to retrieve or refresh GCP Application Default "
                "Credentials: %s",
                e,
            )
            raise
        except Exception as e:
            self.logger.exception(
                "Unexpected error while fetching GCP ADC credentials: %s", e
            )
            raise


class DataEngineeringAgentGenerator(QueryGenerator):
    """Data Engineering Agent (DEA) Query Generator using the A2A SDK."""

    def __init__(self, querygenerator_config: dict[str, Any]):
        super().__init__(querygenerator_config)
        self.name = "data_engineering_agent"
        self.endpoint = querygenerator_config.get("endpoint", "")
        self.target_workspace = querygenerator_config.get(
            "target_workspace", ""
        )

        if not self.endpoint:
            raise ValueError(
                "Configuration key 'endpoint' is required for "
                "DataEngineeringAgentGenerator."
            )
        if not self.target_workspace:
            raise ValueError(
                "Configuration key 'target_workspace' is required for "
                "DataEngineeringAgentGenerator."
            )

        self.logger = logging.getLogger(__name__)

        self.auth_interceptor = AuthInterceptor(GcpAdcCredentialService())
        self.logger.info(
            "A2A AuthInterceptor successfully configured with "
            "GcpAdcCredentialService."
        )

    def generate_internal(self, prompt: str) -> Any:
        """Stubbed messaging logic for WIP scaffolding (Task 1.3)."""
        raise NotImplementedError(
            "Task 1.3 DEA A2A messaging logic in generate_internal is "
            "not yet implemented."
        )
