"""
TrajectoryMatcher

Compares the expected tool usage trajectory with the actual executed tools.

Tool names on both sides are expected to already be in canonical form -- MCP
tools as ``<server>__<tool>`` and native tools as their bare names. Each
harness adapter performs that normalization at the boundary (see
``generators/models/tool_naming.py``), so this scorer can stay
generator-agnostic and do a plain string comparison.

By default the matcher drops native/harness-internal tools (``Read``,
``Bash``, ``update_topic``, ``run_shell_command``, ``ToolSearch``, ...)
from both expected and actual trajectories before scoring, so dataset
authors can focus ``expected_trajectory`` on user-visible MCP intent.
Set ``filter_native_tools: false`` in the scorer config to compare raw
trajectories instead -- useful when an evalset cares about how often the
agent reaches for a native tool.
"""

from typing import Tuple, Any, List
from scorers import comparator
from generators.models.tool_naming import looks_like_canonical_mcp_name


class TrajectoryMatcher(comparator.Comparator):
    """
    TrajectoryMatcher class implements the Comparator base class for checking tool execution trajectories.

    It checks if the sequence of executed tools matches the expected trajectory using
    Jaccard Similarity for flexible ordering or Levenshtein distance for strict order enforcement.
    """

    def __init__(self, config: dict):
        self.name = "trajectory_matcher"
        self.config = config
        self.enforce_order = config.get("enforce_order", False)
        self.filter_native_tools = config.get("filter_native_tools", True)

    def _levenshtein_distance(self, seq1: List[str], seq2: List[str]) -> int:
        n, m = len(seq1), len(seq2)
        if n == 0:
            return m
        if m == 0:
            return n

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + cost  # Substitution
                )

        return dp[n][m]

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 1.0  # Both are empty
        return intersection / union

    def compare(
        self,
        nl_prompt: str,
        golden_query: str,
        query_type: str,
        golden_execution_result: list,
        golden_eval_result: str,
        golden_error: str,
        generated_query: str,
        generated_execution_result: list,
        generated_eval_result: str,
        generated_error: str,
    ) -> Tuple[float, str]:
        """
        Compares expected trajectory (golden) with actual executed tools (generated).

        Args:
            golden_execution_result: List of expected tool names (strings).
            generated_execution_result: List of actually executed tool names (strings).

        Returns:
            Tuple (score, explanation)
        """
        if generated_error:
            return 0.0, f"Generation error: {generated_error}"

        expected = golden_execution_result or []
        actual = generated_execution_result or []

        if not isinstance(expected, list) or not isinstance(actual, list):
            return 0.0, "Trajectory data must be lists."

        filter_note = ""
        if self.filter_native_tools:
            filtered_expected = [t for t in expected if looks_like_canonical_mcp_name(t)]
            filtered_actual = [t for t in actual if looks_like_canonical_mcp_name(t)]
            dropped_expected = len(expected) - len(filtered_expected)
            dropped_actual = len(actual) - len(filtered_actual)
            if dropped_expected or dropped_actual:
                filter_note = (
                    f" (filter_native_tools=True dropped "
                    f"{dropped_expected} expected, {dropped_actual} actual)"
                )
            expected, actual = filtered_expected, filtered_actual

        if not expected and not actual:
            return 100.0, (
                "Both expected and actual trajectories are empty." + filter_note
            )

        score = 0.0
        explanation = ""

        if self.enforce_order:
            # Ordered comparison (Levenshtein distance)
            distance = self._levenshtein_distance(expected, actual)
            max_len = max(len(expected), len(actual))

            # Normalize to 0-100 score
            normalized_score = max(
                0.0, 1.0 - (distance / max_len)) if max_len > 0 else 1.0
            score = normalized_score * 100.0
            explanation = (
                f"Sequence Alignment Score: {score:.2f} (Distance: {distance}, "
                f"Max Length: {max_len}). Expected: {expected}, Actual: {actual}"
                + filter_note
            )

        else:
            # Flexible ordering (Jaccard Similarity)
            similarity = self._jaccard_similarity(set(expected), set(actual))
            score = similarity * 100.0
            explanation = (
                f"Jaccard Similarity Score: {score:.2f} (Intersection over Union). "
                f"Expected Set: {set(expected)}, Actual Set: {set(actual)}"
                + filter_note
            )

        return score, explanation
