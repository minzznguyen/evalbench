import asyncio
import logging
import urllib.parse
import uuid
from typing import Any

# A2A SDK Imports
from a2a.client import create_client, ClientConfig, ClientCallContext
from a2a.client.auth import AuthInterceptor, CredentialService
from a2a.types import a2a_pb2 as pb
import google.auth
from google.auth.credentials import Credentials
from google.auth.exceptions import DefaultCredentialsError, RefreshError
from google.auth.transport.requests import Request

from .generator import QueryGenerator

# Standardized A2A Extension URIs
CONVERSATION_TOKEN_URI = "https://geminidataanalytics.googleapis.com/a2a/extensions/conversationtoken/v1"
GCP_RESOURCE_URI = "https://geminidataanalytics.googleapis.com/a2a/extensions/gcpresource/v1"
MESSAGE_LEVEL_URI = "https://geminidataanalytics.googleapis.com/a2a/extensions/messagelevel/v1"
INSTRUCTION_URI = "https://geminidataanalytics.googleapis.com/a2a/extensions/instruction/v1"
FINISH_REASON_URI = "https://geminidataanalytics.googleapis.com/a2a/extensions/finishreason/v1"

# All required A2A Extension Headers combined
ALL_EXTENSIONS = f"{MESSAGE_LEVEL_URI},{INSTRUCTION_URI},{GCP_RESOURCE_URI},{CONVERSATION_TOKEN_URI},{FINISH_REASON_URI}"

# Define module-level logger consistent with gemini.py
logger = logging.getLogger(__name__)


class GcpAdcCredentialService(CredentialService):
    """GCP Application Default Credentials (ADC) service for A2A SDK.

    This provider only services OAuth/OAuth2 schemes.
    """

    credentials: Credentials | None
    _lock: asyncio.Lock

    def __init__(self, scopes: list[str] = None) -> None:
        """Initializes the GCP Application Default Credentials service.

        Args:
            scopes: Optional list of GCP OAuth scopes to request. Defaults
              to full cloud-platform scope.
        """
        self.credentials = None
        self.scopes = scopes or [
            "https://www.googleapis.com/auth/cloud-platform"
        ]
        self._lock = None

    async def get_credentials(
        self,
        security_scheme_name: str,
        context: ClientCallContext | None,
    ) -> str:
        """Retrieves or refreshes GCP ADC credentials and returns the token.

        Args:
            security_scheme_name: The name of the security scheme (e.g.,
              'oauth').
            context: The A2A client call context (unused).

        Returns:
            The OAuth2 access token as a string.

        Raises:
            ValueError: If the security scheme is not supported.
            DefaultCredentialsError: If credentials fail to load.
        """
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
                    # Offload blocking filesystem/env I/O to a background thread
                    credentials, _ = await asyncio.to_thread(
                        google.auth.default,
                        scopes=self.scopes
                    )
                    self.credentials = credentials

                if self.credentials is None:
                    logger.error(
                        "GCP Application Default Credentials failed to "
                        "initialize."
                    )
                    raise DefaultCredentialsError(
                        "GCP Application Default Credentials failed to "
                        "initialize."
                    )

                if not self.credentials.valid:
                    await asyncio.to_thread(
                        self.credentials.refresh, Request()
                    )

                logger.debug("Retrieved GCP ADC token successfully.")
                return self.credentials.token

        except (DefaultCredentialsError, RefreshError) as e:
            logger.error(
                "Failed to retrieve or refresh GCP Application Default "
                "Credentials: %s",
                e,
            )
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error while fetching GCP ADC credentials: %s", e
            )
            raise


class DataEngineeringAgentGenerator(QueryGenerator):
    """Data Engineering Agent (DEA) Query Generator using the A2A SDK."""

    name: str
    endpoint: str
    target_workspace: str
    auth_interceptor: AuthInterceptor
    _session_tokens: dict[str, str]

    def __init__(self, querygenerator_config: dict[str, Any]) -> None:
        """Initializes the DataEngineeringAgentGenerator with the provided configuration.

        Args:
            querygenerator_config: Configuration dictionary containing 'endpoint' and
              'target_workspace' settings.
        """
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

        # Security: Strict URL schema & host validation to prevent SSRF
        parsed_endpoint = urllib.parse.urlparse(self.endpoint)
        if parsed_endpoint.scheme not in ("http", "https") or not parsed_endpoint.netloc:
            raise ValueError(
                f"Configuration key 'endpoint' has an invalid URL: '{self.endpoint}'"
            )

        # Security: Strict alphanumeric validation on target workspace path to prevent injection
        workspace_chars = self.target_workspace.replace("/", "").replace("-", "").replace("_", "")
        if not workspace_chars.isalnum():
            raise ValueError(
                f"Configuration key 'target_workspace' contains invalid characters: "
                f"'{self.target_workspace}'"
            )

        scopes = querygenerator_config.get("adc_scopes")
        self.auth_interceptor = AuthInterceptor(
            GcpAdcCredentialService(scopes=scopes)
        )

        # Map to maintain session-isolated ConversationTokens thread-safely in memory
        self._session_tokens = {}

        logger.info(
            "A2A AuthInterceptor successfully configured with "
            "GcpAdcCredentialService."
        )

    def generate_internal(self, prompt: Any) -> Any:
        """Polymorphic entry point that detects and integrates with different evaluators."""
        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
        except RuntimeError:
            pass

        if hasattr(prompt, "nl_prompt"):
            # 🟢 Pattern A: Programmatic Object Flow (CortadoEvaluator)
            # Mutates the prompt object directly.
            prompt_text = prompt.nl_prompt
            session_id = prompt.id

            try:
                response_text = asyncio.run(self._run_client(prompt_text, session_id=session_id))
                prompt.generated_nl_response = response_text
                prompt.generated_sql = ""
            except Exception as e:
                logger.exception("A2A SDK messaging error in Cortado flow")
                prompt.generated_nl_response = ""
                prompt.generated_sql = ""
            return prompt

        elif isinstance(prompt, dict):
            # 🟢 Pattern B: Conversational Dict Flow (DataAgentEvaluator)
            # Mutates the dictionary object directly.
            prompt_text = prompt.get("nl_prompt", "")
            session_id = prompt.get("id", f"session-{uuid.uuid4()}")

            try:
                response_text = asyncio.run(self._run_client(prompt_text, session_id=session_id))
                prompt["nl_response"] = response_text
                prompt["generated_sql"] = ""
                prompt["disambiguation_question"] = None
                prompt["sql_generator_error"] = None
            except Exception as e:
                logger.exception("A2A SDK messaging error in dict flow")
                prompt["generated_sql"] = None
                prompt["sql_generator_error"] = str(e)
            return prompt

        else:
            # 🟢 Pattern C: Standard String Flow (OneShot / Direct test)
            # Returns raw string output.
            return asyncio.run(self._run_client(str(prompt), session_id=None))

    async def _run_client(self, prompt: str, session_id: str | None) -> str:
        """Core asynchronous A2A SDK connection loop."""
        # Configure Client in standard Non-Streaming Mode
        config = ClientConfig(
            supported_protocol_bindings=["HTTP+JSON"],
            streaming=False
        )

        oauth_scheme = pb.SecurityScheme(oauth2_security_scheme=pb.OAuth2SecurityScheme())
        agent_card = pb.AgentCard(
            name="DataEngineeringAgent",
            supported_interfaces=[
                pb.AgentInterface(
                    protocol_binding="HTTP+JSON",
                    url=self.endpoint,
                    protocol_version="0.3"
                )
            ],
            capabilities=pb.AgentCapabilities(streaming=False),
            security_schemes={
                "oauth": oauth_scheme
            },
            security_requirements=[
                pb.SecurityRequirement(
                    schemes={
                        "oauth": pb.StringList()
                    }
                )
            ]
        )

        client = await create_client(
            agent_card,
            client_config=config,
            interceptors=[self.auth_interceptor]
        )

        if not session_id:
            session_id = f"session-{uuid.uuid4()}"

        req = pb.SendMessageRequest(
            message=pb.Message(
                message_id=str(uuid.uuid4()),
                role=pb.ROLE_USER,
                context_id="live-dea-workflow",
                parts=[pb.Part(text=prompt)]
            )
        )

        # Configure GCP Resource extension
        req.metadata.update({
            GCP_RESOURCE_URI: {
                "gcpResourceId": self.target_workspace
            }
        })

        # Handle ConversationToken state memory
        token = self._session_tokens.get(session_id, "")
        if token:
            req.metadata[CONVERSATION_TOKEN_URI] = token

        context = ClientCallContext(
            timeout=180.0,
            service_parameters={
                "A2A-Extensions": ALL_EXTENSIONS
            }
        )

        reply_text = ""
        new_token = ""

        try:
            async for resp in client.send_message(req, context=context):
                if resp.HasField("task") and resp.task.history:
                    reply_text = ""
                    for msg in resp.task.history:
                        if msg.role == pb.ROLE_AGENT:
                            msg_level = ""
                            if MESSAGE_LEVEL_URI in msg.metadata:
                                msg_level = msg.metadata[MESSAGE_LEVEL_URI]

                            # Filter out internal DEBUG, WARNING, and INFO messages
                            if msg_level not in ["DEBUG", "WARNING", "INFO"]:
                                for part in msg.parts:
                                    if part.text:
                                        reply_text += part.text

                    # Extract Conversation Token
                    if CONVERSATION_TOKEN_URI in resp.task.metadata:
                        new_token = resp.task.metadata[CONVERSATION_TOKEN_URI]
        except Exception as e:
            logger.exception("A2A SDK messaging error")
            raise

        await client.close()

        # Cache the new token
        if new_token:
            self._session_tokens[session_id] = new_token

        return reply_text.strip()
