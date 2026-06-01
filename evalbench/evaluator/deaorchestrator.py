import datetime
import json
import tempfile
import uuid
from typing import Any, List

from dataset.deainput import EvalDeaRequest
from evaluator.orchestrator import Orchestrator
from evaluator.deaevaluator import DeaEvaluator


class DeaOrchestrator(Orchestrator):
    """Orchestrator designed specifically for pure conversational, non-database DEA evaluations.

    Bypasses all legacy database connection handshakes, dialect checks, and connection pool setups.
    """

    def __init__(
        self,
        config: dict[str, Any],
        db_configs: dict[str, Any] | None = None,
        setup_config: dict[str, Any] | None = None,
        report_progress: bool = False,
    ) -> None:
        super().__init__(config, db_configs, setup_config, report_progress)
        self.config = config
        self.job_id = f"dea-job-{uuid.uuid4()}"
        self.run_time = datetime.datetime.now()
        self.total_eval_outputs: List[dict[str, Any]] = []
        self.total_scoring_results: List[dict[str, Any]] = []

    def evaluate(self, dataset: List[EvalDeaRequest]) -> None:
        """Orchestrates pure conversational evaluations by delegating straight to DeaEvaluator."""
        evaluator = DeaEvaluator(self.config)
        eval_outputs, scoring_results = evaluator.evaluate(
            dataset, self.job_id, self.run_time
        )
        self.total_eval_outputs.extend(eval_outputs)
        self.total_scoring_results.extend(scoring_results)

    def process(self) -> tuple[str, datetime.datetime, str, str]:
        """Packages and writes the final completed scores and transcripts to JSON files."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(
                self.total_eval_outputs, f, sort_keys=True, indent=4, default=str
            )
            results_tf = f.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(
                self.total_scoring_results, f, sort_keys=True, indent=4, default=str
            )
            scores_tf = f.name

        return self.job_id, self.run_time, results_tf, scores_tf
