"""Unit tests for the trajectory matcher.

The matcher is generator-agnostic: it expects tool names on both sides to
already be in canonical ``<server>__<tool>`` form (or bare for native
tools). Per-harness normalization lives in the adapters; see
``generators/models/tool_naming.py`` for the canonical-naming helper and
``test/tool_naming_test.py`` for its tests.
"""

import os
import sys
import unittest

# Make the ``scorers`` package importable when the test is run directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scorers.trajectorymatcher import TrajectoryMatcher


def _compare(matcher, expected, actual):
    """Convenience wrapper around the matcher's positional ``compare`` API."""
    return matcher.compare(
        None, None, None, expected, None, None, None, actual, None, None
    )


class TrajectoryMatcherTest(unittest.TestCase):

    def test_exact_match_returns_full_score(self):
        matcher = TrajectoryMatcher({})

        expected = ["cloud-sql__list_instances", "cloud-sql__get_instance"]
        actual = ["cloud-sql__list_instances", "cloud-sql__get_instance"]

        score, explanation = _compare(matcher, expected, actual)
        self.assertEqual(score, 100.0)
        self.assertIn("Jaccard Similarity Score: 100.00", explanation)

    def test_jaccard_ignores_order_by_default(self):
        matcher = TrajectoryMatcher({})

        expected = ["cloud-sql__list_instances", "cloud-sql__get_instance"]
        actual = ["cloud-sql__get_instance", "cloud-sql__list_instances"]

        score, _ = _compare(matcher, expected, actual)
        self.assertEqual(score, 100.0)

    def test_strict_ordering_penalizes_swaps(self):
        matcher = TrajectoryMatcher({"enforce_order": True})

        expected = ["cloud-sql__list_instances", "cloud-sql__get_instance"]
        actual = ["cloud-sql__get_instance", "cloud-sql__list_instances"]

        score, _ = _compare(matcher, expected, actual)
        self.assertLess(score, 100.0)

    def test_server_qualifier_distinguishes_same_tool_across_servers(self):
        # Without the server prefix, both calls would collide with the
        # expected ``list_instances``. The canonical form keeps them
        # distinct: alloydb's call should not satisfy a cloud-sql expectation.
        matcher = TrajectoryMatcher({})

        expected = ["cloud-sql__list_instances"]
        actual = ["alloydb__list_instances"]

        score, _ = _compare(matcher, expected, actual)
        self.assertEqual(score, 0.0)

    def test_native_tools_pass_through_when_filter_disabled(self):
        # With filtering off, native tool names are compared verbatim.
        matcher = TrajectoryMatcher({"filter_native_tools": False})

        expected = ["Read", "Bash"]
        actual = ["Read", "Bash"]

        score, _ = _compare(matcher, expected, actual)
        self.assertEqual(score, 100.0)

    def test_filter_native_tools_drops_native_on_actual_by_default(self):
        # Default-on filter: native tools in actual must not drag Jaccard down
        # when expected contains only MCP intent.
        matcher = TrajectoryMatcher({})

        expected = ["cloud-sql__list_instances"]
        actual = ["cloud-sql__list_instances", "Read", "Bash", "update_topic"]

        score, explanation = _compare(matcher, expected, actual)
        self.assertEqual(score, 100.0)
        self.assertIn("filter_native_tools=True", explanation)

    def test_filter_disabled_keeps_native_tools(self):
        # With filtering off, the same native leakage drags Jaccard down.
        matcher = TrajectoryMatcher({"filter_native_tools": False})

        expected = ["cloud-sql__list_instances"]
        actual = ["cloud-sql__list_instances", "Read", "Bash", "update_topic"]

        score, _ = _compare(matcher, expected, actual)
        self.assertLess(score, 100.0)

    def test_filter_applies_to_expected_too(self):
        # Symmetric filtering: native tools in expected are also dropped so
        # an evalset author can't accidentally pin behavior on a native tool
        # while filtering is on.
        matcher = TrajectoryMatcher({})

        expected = ["cloud-sql__list_instances", "Bash"]
        actual = ["cloud-sql__list_instances"]

        score, _ = _compare(matcher, expected, actual)
        self.assertEqual(score, 100.0)

    def test_filter_removes_all_tools_scores_empty(self):
        # If filtering wipes both sides clean, the matcher should report the
        # standard "both empty" success rather than divide-by-zero.
        matcher = TrajectoryMatcher({})

        score, explanation = _compare(matcher, ["Read"], ["Bash"])
        self.assertEqual(score, 100.0)
        self.assertIn("empty", explanation)
        self.assertIn("filter_native_tools=True", explanation)

    def test_both_empty_is_full_score(self):
        matcher = TrajectoryMatcher({})
        score, explanation = _compare(matcher, [], [])
        self.assertEqual(score, 100.0)
        self.assertIn("empty", explanation)

    def test_generation_error_returns_zero(self):
        matcher = TrajectoryMatcher({})
        score, explanation = matcher.compare(
            None, None, None, ["x"], None, None, None, ["x"], None, "boom"
        )
        self.assertEqual(score, 0.0)
        self.assertIn("boom", explanation)


if __name__ == "__main__":
    unittest.main()
