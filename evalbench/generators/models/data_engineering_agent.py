import logging
from typing import Any, Dict

from a2a.client import create_client, ClientConfig
from a2a.client.auth import AuthInterceptor, CredentialService
from a2a.client.client import ClientCallContext
from a2a.types import a2a_pb2 as pb
import google.auth
from google.auth.transport.requests import Request

from .generator import QueryGenerator


class GcpAdcCredentialService(CredentialService):
    """GCP Application Default Credentials (ADC) service for A2A SDK."""

    async def get_credentials(
        self,
        security_scheme_name: str,
        context: ClientCallContext | None,
    ) -> str | None:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        token = credentials.token
        print(f"🔑 [A2A Credential Service] Successfully retrieved GCP ADC token: {token[:10]}...")
        return token


class DataEngineeringAgentGenerator(QueryGenerator):
    """Data Engineering Agent (DEA) Query Generator."""

    def __init__(self, querygenerator_config: Dict[str, Any]):
        super().__init__(querygenerator_config)
        self.name = "data_engineering_agent"
        self.logger = logging.getLogger(__name__)
        
        # Task 1.2: Configure authentication
        self.auth_interceptor = AuthInterceptor(GcpAdcCredentialService())
        print("✅ A2A AuthInterceptor successfully configured with GCP ADC Credential Service!")

    def generate_internal(self, prompt: str) -> str:
        """Generates a response for the given prompt (A2A logic stub)."""
        raise NotImplementedError("Task 1.3 A2A messaging logic is not yet implemented.")


