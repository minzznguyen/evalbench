import concurrent.futures
import datetime
import json
import logging
from typing import Any, List

from dataset.dataengineeringagentinput import EvalDeaRequest
from generators.models.gcp_data_engineering_agent import (
    DataEngineeringAgentGenerator,
)
from mp import mprunner
from work.agentgenwork import AgentGenWork
from evaluator.simulateduser import SimulatedUser
from work.agentscorework import AgentScoreWork
from util.config import load_yaml_config
from util.dataform import DataformHelper

# Module-level logger
logger = logging.getLogger(__name__)


class DataEngineeringAgentEvaluator:
    """Evaluator designed specifically for pure conversational DEA evaluations.

    Coordinates turn-by-turn natural language dialogue between the Simulated
    User and the generator, completely bypassing all SQL execution and
    database dependencies.
    """

    generator: DataEngineeringAgentGenerator
    agentrunner: mprunner.MPRunner

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

        # Resolve and parse model config
        model_config = config
        if (
            "model_config" in config
            and isinstance(config["model_config"], str)
        ):
            loaded_config = load_yaml_config(config["model_config"])
            model_config = loaded_config.copy()
            model_config.update(config)

        self.generator = DataEngineeringAgentGenerator(model_config)
        self.project_id = model_config.get("gcp_project_id", "")
        self.location = model_config.get("gcp_region", "")

        runner_config = self.config.get("runners", {})
        self.agent_runners = runner_config.get("agent_runners", 10)
        self.agentrunner = mprunner.MPRunner(self.agent_runners)

        reporting_config = self.config.get("reporting") or {}
        self.gcs_config = {}
        if isinstance(reporting_config, dict):
            self.gcs_config = reporting_config.get("gcs_export") or {}

    def evaluate(
        self,
        dataset: List[EvalDeaRequest],
        job_id: str,
        run_time: datetime.datetime,
    ):
        """Runs the conversational scenarios in a parallel thread pool."""
        eval_outputs: List[dict[str, Any]] = []
        if not self.gcs_config.get("bucket_name"):
            logger.error(
                "GCS export is required but 'bucket_name' is missing from "
                "configuration."
            )
            raise ValueError("GCS export 'bucket_name' is required.")

        scoring_results: List[dict[str, Any]] = []
        logger.info("Running pure conversational DEA evaluation")

        job_uuid = job_id.removeprefix("dea-job-")
        repo_id = f"evalbench-{job_uuid}"

        # Derive workspace_id from the dataset config file name
        dataset_path = self.config.get("dataset_config", "")
        workspace_id = (
            dataset_path.split("/")[-1]
            .split(".evalset.json")[0]
            .split(".json")[0]
            .lower()
            .replace("_", "-")
            .replace(" ", "-")
        )

        helper = DataformHelper(self.project_id, self.location)

        created_repo_name = None

        try:
            created_repo_name = helper.create_repository(repo_id)
            workspace_name = helper.create_workspace(
                repo_id, workspace_id
            )

            # Validate workspace exists before running.
            if not workspace_name:
                raise ValueError(
                    "GCP Resource Workspace URI must be dynamically provisioned "
                    "and present on the evaluation item."
                )

            for item in dataset:
                item.gcp_resource_id = workspace_name

            self.agentrunner.futures.clear()

            metadata = {
                "dialects": self.config.get("dialects", []),
                "database": self.config.get("database", "unknown"),
                "scorers": self.config.get("scorers", {}),
            }

            # Submit scenarios concurrently to parallel worker threads
            for item in dataset:
                simulated_user = SimulatedUser(self.config)
                work = AgentGenWork(
                    processor=self.process_scenario,
                    eval_result=item,
                    job_id=job_id,
                    metadata=metadata,
                    simulated_user=simulated_user,
                )
                self.agentrunner.execute_work(work)

            futures = self.agentrunner.futures
            for future in concurrent.futures.as_completed(futures):
                try:
                    modified_item = future.result()
                    if hasattr(modified_item, "agent_results"):
                        eval_outputs.extend(modified_item.agent_results)
                    if hasattr(modified_item, "scoring_results"):
                        scoring_results.extend(modified_item.scoring_results)
                except Exception as e:
                    logger.exception(f"Error getting result from future: {e}")

        except Exception as err:
            logger.exception("Failed to initialize dynamic cloud sandbox: %s", err)
            raise
        finally:
            bucket_name = self.gcs_config.get("bucket_name")
            if bucket_name and created_repo_name:
                try:
                    logger.info(
                        "Exporting workspace artifacts to GCS bucket: %s...",
                        bucket_name,
                    )
                    gcs_prefix = f"runs/{job_id}"
                    helper.upload_archive_to_gcs(
                        repository_id=repo_id,
                        workspace_id=workspace_id,
                        bucket_name=bucket_name,
                        gcs_prefix=gcs_prefix,
                    )
                    logger.info("Successfully exported all artifacts to GCS!")
                except Exception as export_err:
                    logger.exception("Failed to export workspace artifacts to GCS")

            if created_repo_name:
                try:
                    logger.info("Cleaning up temporary cloud resources...")
                    helper.delete_repository(repo_id)
                except Exception as cleanup_err:
                    logger.error(
                        "Failed to clean up Dataform resources: %s",
                        cleanup_err,
                    )

        return eval_outputs, scoring_results

    def process_scenario(
        self,
        scenario: dict[str, Any],
        eval_result: EvalDeaRequest,
        job_id: str,
        metadata: dict[str, Any],
        simulated_user: SimulatedUser | None = None,
    ) -> EvalDeaRequest:
        """Manages the multi-turn conversational dialogue turn-by-turn."""

        current_prompt = scenario.get("starting_prompt", "")
        max_turns = scenario.get("max_turns", 1)
        conversation_plan = scenario.get("conversation_plan", [])
        conversation_history: List[dict[str, str]] = []
        last_agent_text = ""

        for turn in range(max_turns):
            logger.info(
                "Turn %d/%d - Prompt: %s",
                turn + 1,
                max_turns,
                current_prompt
            )

            # Pass prompt to programmatic request object
            eval_result.nl_prompt = current_prompt

            agent_text = ""
            try:
                # Native API call: mutates eval_result in-place
                self.generator.generate(eval_result)
                agent_text = getattr(eval_result, "generated_nl_response", "")
            except Exception as e:
                logger.exception(
                    "A2A SDK generation failed: %s", type(e).__name__
                )
                agent_text = f"Error: {e}"

            last_agent_text = agent_text
            logger.info(
                "Turn %d/%d - Agent Reply: %s",
                turn + 1,
                max_turns,
                agent_text
            )

            conversation_history.append({
                "user": current_prompt,
                "agent": agent_text,
            })

            # Simulated User checks conversation plan and generates next prompt
            if turn < max_turns - 1 and simulated_user:
                next_response = simulated_user.get_next_response(
                    conversation_plan, conversation_history, agent_text
                )
                if "TERMINATE" in next_response:
                    logger.info("Simulated user met the goal and terminated.")
                    break
                current_prompt = next_response
            else:
                break

        self._finalize_scenario(
            scenario,
            last_agent_text,
            conversation_history,
            eval_result,
            job_id,
            metadata,
        )
        return eval_result

    def _finalize_scenario(
        self,
        scenario: dict[str, Any],
        last_response: str,
        conversation_history: List[dict[str, str]],
        eval_result: EvalDeaRequest,
        job_id: str,
        metadata: dict[str, Any],
    ) -> None:
        """Packages conversation and invokes scoring engine."""
        eval_output_data = {
            "eval_id": scenario["id"],
            "stdout": last_response,
            "stderr": "",
            "returncode": 0 if not last_response.startswith("Error") else 1,
            "prompt_generator_error": None,
            "generated_error": None,
            "sql_generator_error": None,
            "golden_error": None,
            # Non-SQL conversational runs skip SQL evaluation
            "generated_sql": "skipped",
            "prompt": scenario["starting_prompt"],
            "conversation_history": json.dumps(conversation_history, indent=2),
            "scenario": scenario,
            "accumulated_tools": [],
            "accumulated_skills": [],
            "job_id": job_id,
            "metadata": metadata,
            "gcp_resource_id": getattr(eval_result, "gcp_resource_id", None),
        }

        score_work = AgentScoreWork(
            config=self.config,
            eval_output=eval_output_data,
            scoring_results=eval_result.scoring_results,
        )
        score_work.run()
        eval_result.agent_results.append(eval_output_data)
