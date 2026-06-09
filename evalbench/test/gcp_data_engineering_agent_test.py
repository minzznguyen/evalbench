import asyncio
import concurrent.futures
import concurrent.futures.thread
import logging
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from google.auth.exceptions import DefaultCredentialsError, RefreshError

from a2a.client.base_client import BaseClient
from a2a.client.transports.base import ClientTransport
from a2a.types import a2a_pb2 as pb
from a2a.utils import TransportProtocol

# Add generators path to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataengineeringagentinput import EvalDeaRequest  # noqa: E402
from evaluator.dataengineeringagentorchestrator import (  # noqa: E402
    DataEngineeringAgentOrchestrator,
)
from generators.models import get_generator  # noqa: E402
from generators.models.gcp_data_engineering_agent import (  # noqa: E402
    DataEngineeringAgentGenerator,
    GcpAdcCredentialService,
)


def test_data_engineering_agent_generator_setup():
    config = {
        "generator": "data_engineering_agent",
        "gcp_project_id": "test-project-123",
        "gcp_region": "us-east1",
        "target_workspace": (
            "projects/diff-project-abc/locations/diff-region-xyz/repositories/"
            "test-repo/workspaces/test-workspace"
        ),
    }

    # Mock google.auth.default during initialization
    with patch("google.auth.default") as mock_auth_default:
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_auth_default.return_value = (mock_creds, "test-project")

        generator = DataEngineeringAgentGenerator(config)

        assert generator.name == "data_engineering_agent"
        expected_endpoint = (
            "https://geminidataanalytics.googleapis.com/v1/a2a/projects/"
            "test-project-123/locations/us-east1/agents/dataengineeringagent"
        )
        assert generator.endpoint == expected_endpoint
        assert generator.target_workspace == config["target_workspace"]
        assert generator.auth_interceptor is not None


@pytest.mark.anyio
async def test_get_credentials_invalid_scheme():
    service = GcpAdcCredentialService()

    with pytest.raises(ValueError) as excinfo:
        await service.get_credentials("basic", None)

    assert "only services 'oauth' or 'oauth2'" in str(excinfo.value)


def test_generator_setup_missing_project_id():
    config = {
        "generator": "data_engineering_agent",
        "gcp_region": "us-west4",
        "target_workspace": "projects/test-workspace",
    }
    with pytest.raises(ValueError) as excinfo:
        DataEngineeringAgentGenerator(config)
    assert "gcp_project_id' is required" in str(excinfo.value)


def test_generator_setup_missing_workspace():
    config = {
        "generator": "data_engineering_agent",
        "gcp_project_id": "test",
        "gcp_region": "us-west4",
    }
    with pytest.raises(ValueError) as excinfo:
        DataEngineeringAgentGenerator(config)
    assert "target_workspace' is required" in str(excinfo.value)


@pytest.mark.anyio
@patch("google.auth.default")
async def test_get_credentials_error_resiliency_default(mock_auth_default):
    mock_auth_default.side_effect = DefaultCredentialsError(
        "Credentials missing."
    )
    service = GcpAdcCredentialService()

    with pytest.raises(DefaultCredentialsError):
        await service.get_credentials("oauth", None)


@pytest.mark.anyio
@patch("google.auth.default")
async def test_get_credentials_error_resiliency_refresh(mock_auth_default):
    mock_creds = MagicMock()
    mock_creds.valid = False
    mock_creds.refresh.side_effect = RefreshError("Network timed out.")
    mock_auth_default.return_value = (mock_creds, "test-project")

    service = GcpAdcCredentialService()

    with pytest.raises(RefreshError):
        await service.get_credentials("oauth", None)


def test_generator_setup_invalid_workspace_characters():
    config = {
        "generator": "data_engineering_agent",
        "gcp_project_id": "test-project-123",
        "gcp_region": "us-east1",
        "target_workspace": (
            "projects/test-project/locations/us-east1/repositories/test-repo/"
            "workspaces/test-workspace; rm -rf /"
        ),
    }
    with patch("google.auth.default") as mock_auth_default:
        mock_creds = MagicMock()
        mock_auth_default.return_value = (mock_creds, "test-project")

        with pytest.raises(ValueError) as excinfo:
            DataEngineeringAgentGenerator(config)
        assert "target_workspace' contains invalid characters" in str(
            excinfo.value
        )


@pytest.mark.anyio
async def test_gcp_adc_credential_service_concurrency():
    service = GcpAdcCredentialService()
    mock_creds = MagicMock()
    mock_creds.valid = False
    mock_creds.token = "stub-token"

    def slow_refresh(*args, **kwargs):
        time.sleep(0.2)
        mock_creds.valid = True

    mock_creds.refresh.side_effect = slow_refresh

    with patch("google.auth.default") as mock_auth_default:
        mock_auth_default.return_value = (mock_creds, "test-project")

        results = []
        errors = []

        def run_in_thread():
            try:
                token = asyncio.run(service.get_credentials("oauth", None))
                results.append(token)
            except Exception as e:
                errors.append(e)

        num_threads = 3
        threads = [
            threading.Thread(target=run_in_thread, daemon=True)
            for _ in range(num_threads)
        ]
        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=3.0)
            if t.is_alive():
                pytest.fail("Deadlock detected in credential service")

        if errors:
            pytest.fail(f"Threads raised errors: {errors}")

        assert len(results) == num_threads
        assert all(t == "stub-token" for t in results)
        assert mock_creds.refresh.call_count == 1


@pytest.mark.anyio
@patch("google.auth.default")
@patch("generators.models.gcp_data_engineering_agent.create_client")
async def test_generate_internal_uses_minimal_agent_card(
    mock_create_client, mock_auth_default
):
    mock_creds = MagicMock()
    mock_creds.valid = True
    mock_auth_default.return_value = (mock_creds, "test-project")
    mock_client = MagicMock()

    async def mock_send_message(*args, **kwargs):
        resp = pb.SendMessageResponse()
        resp.task.metadata[
            "https://geminidataanalytics.googleapis.com/a2a/extensions/"
            "conversationtoken/v1"
        ] = "stub-token-123"
        msg = pb.Message(role=pb.ROLE_AGENT)
        msg.metadata[
            "https://geminidataanalytics.googleapis.com/a2a/extensions/"
            "messagelevel/v1"
        ] = "USER"
        msg.parts.append(pb.Part(text="Analysis complete."))
        resp.task.history.append(msg)
        yield resp

    mock_client.send_message.side_effect = mock_send_message

    async def mock_close():
        pass
    mock_client.close.side_effect = mock_close
    mock_create_client.return_value = mock_client

    config = {
        "generator": "data_engineering_agent",
        "gcp_project_id": "test",
        "gcp_region": "us-west4",
        "target_workspace": (
            "projects/test/locations/us-west4/repositories/"
            "test-repo/workspaces/test-workspace"
        ),
    }

    generator = DataEngineeringAgentGenerator(config)
    req = EvalDeaRequest({
        "starting_prompt": "Please analyze table schema.",
        "id": "test-conv-id",
    })
    result = generator.generate_internal(req)

    assert result.generated_nl_response == "Analysis complete."
    mock_create_client.assert_called_once()
    called_card = mock_create_client.call_args[0][0]
    assert len(called_card.supported_interfaces) == 1
    expected_endpoint = (
        "https://geminidataanalytics.googleapis.com/v1/a2a/projects/"
        "test/locations/us-west4/agents/dataengineeringagent"
    )
    assert called_card.supported_interfaces[0].url == expected_endpoint
    assert (
        called_card.supported_interfaces[0].protocol_binding
        == TransportProtocol.HTTP_JSON
    )
    assert called_card.capabilities.extended_agent_card is True


@patch("evaluator.dataengineeringagentevaluator.SimulatedUser")
@patch("evaluator.dataengineeringagentevaluator.AgentScoreWork")
@patch("google.auth.default")
@patch("generators.models.gcp_data_engineering_agent.create_client")
def test_parallel_runners_deadlock(
    mock_create_client, mock_auth_default, mock_score_work, mock_simulated_user
):
    mock_creds = MagicMock()
    mock_creds.valid = False
    mock_creds.token = "stub-token"
    mock_creds.expiry = None

    def slow_refresh(*args, **kwargs):
        time.sleep(0.5)
        mock_creds.valid = True

    mock_creds.refresh.side_effect = slow_refresh
    mock_auth_default.return_value = (mock_creds, "test-project")
    mock_transport = MagicMock(spec=ClientTransport)

    async def mock_send_message(*args, **kwargs):
        resp = pb.SendMessageResponse()
        resp.message.role = pb.ROLE_AGENT
        resp.message.parts.append(pb.Part(text="Mocked Response"))
        return resp

    mock_transport.send_message.side_effect = mock_send_message

    async def mock_close():
        pass
    mock_transport.close.side_effect = mock_close

    async def fake_create_client(
        agent_card, client_config, interceptors, **kwargs
    ):
        return BaseClient(
            card=agent_card,
            config=client_config,
            transport=mock_transport,
            interceptors=interceptors or [],
        )

    mock_create_client.side_effect = fake_create_client

    config = {
        "generator": "data_engineering_agent",
        "gcp_project_id": "test-project",
        "gcp_region": "us-west4",
        "target_workspace": (
            "projects/test/locations/us-west4/repositories/"
            "test-repo/workspaces/test-workspace"
        ),
        "runners": {"agent_runners": 2},
    }

    dataset = [
        EvalDeaRequest({
            "id": "scenario-0",
            "starting_prompt": "Prompt 0",
            "max_turns": 1,
            "conversation_plan": ["Verify 0"],
            "binary_rubric": ["Rubric 0"],
        }),
        EvalDeaRequest({
            "id": "scenario-1",
            "starting_prompt": "Prompt 1",
            "max_turns": 1,
            "conversation_plan": ["Verify 1"],
            "binary_rubric": ["Rubric 1"],
        }),
    ]

    orchestrator = DataEngineeringAgentOrchestrator(config)
    test_logger = logging.getLogger("test_debug")
    original_thread = threading.Thread

    def daemon_thread_factory(*args, **kwargs):
        name = kwargs.get('name', 'Unnamed')
        target = kwargs.get('target', None)
        test_logger.info(
            f"Creating thread: name={name}, target={target}, "
            f"daemon={kwargs.get('daemon', None)}"
        )
        if 'daemon' not in kwargs:
            kwargs['daemon'] = True
            test_logger.info(f"Forced daemon=True for {name}")
        return original_thread(*args, **kwargs)

    def run_evaluate():
        with patch("threading.Thread", side_effect=daemon_thread_factory):
            orchestrator.evaluate(dataset)

    t = threading.Thread(target=run_evaluate, daemon=True)
    t.start()
    t.join(timeout=3.0)

    if t.is_alive():
        for item in list(threading._threading_atexits):
            if item.__closure__:
                for cell in item.__closure__:
                    if (
                        cell.cell_contents
                        is concurrent.futures.thread._python_exit
                    ):
                        threading._threading_atexits.remove(item)
                        break
        assert False, "Orchestrator hung (concurrency lock issue detected)"


@patch("generators.models.gcp_data_engineering_agent.logger")
def test_log_api_error_details_httpx_json(mock_logger):
    from generators.models.gcp_data_engineering_agent import (
        _log_api_error_details,
    )

    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = '{"error": {"message": "GCP Permission Denied"}}'

    mock_exception = Exception("HTTP Error")
    mock_exception.response = mock_response

    _log_api_error_details(mock_exception)

    mock_logger.exception.assert_called_once_with(
        "API Error (Status 403): GCP Permission Denied"
    )


@patch("generators.models.gcp_data_engineering_agent.logger")
def test_log_api_error_details_grpc(mock_logger):
    from generators.models.gcp_data_engineering_agent import (
        _log_api_error_details,
    )

    mock_exception = Exception("gRPC Error")
    mock_exception.code = MagicMock(return_value="RESOURCE_EXHAUSTED")
    mock_exception.details = MagicMock(return_value="Quota Exceeded")

    _log_api_error_details(mock_exception)

    mock_logger.exception.assert_called_once_with(
        "API gRPC Error (Code %s): %s", "RESOURCE_EXHAUSTED", "Quota Exceeded"
    )


def test_extract_reply_text_heuristic_standard_node():
    from generators.models.gcp_data_engineering_agent import (
        _extract_reply_text,
    )

    resp = pb.SendMessageResponse()
    resp.message.role = pb.ROLE_AGENT
    resp.message.metadata[
        "https://geminidataanalytics.googleapis.com/a2a/extensions/"
        "messagelevel/v1"
    ] = "USER"
    resp.message.parts.append(pb.Part(text="Direct message text"))

    assert _extract_reply_text(resp) == "Direct message text"


def test_extract_reply_text_heuristic_recursive_fallback():
    from generators.models.gcp_data_engineering_agent import (
        _find_agent_text_recursive,
    )

    nested_msg = pb.Message(role=pb.ROLE_AGENT)
    nested_msg.metadata[
        "https://geminidataanalytics.googleapis.com/a2a/extensions/"
        "messagelevel/v1"
    ] = "USER"
    nested_msg.parts.append(pb.Part(text="Nested message text"))

    container = [{"some_key": "some_value"}, [nested_msg]]
    assert _find_agent_text_recursive(container) == "Nested message text"
