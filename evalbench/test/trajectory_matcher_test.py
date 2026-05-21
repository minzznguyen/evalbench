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

    def test_native_tools_pass_through(self):
        matcher = TrajectoryMatcher({})

        expected = ["Read", "Bash"]
        actual = ["Read", "Bash"]

        score, _ = _compare(matcher, expected, actual)
        self.assertEqual(score, 100.0)

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
