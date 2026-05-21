"""Unit tests for the canonical MCP tool-naming helper."""

import os
import sys
import unittest

# Make the ``generators`` package importable when the test is run directly.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generators.models.tool_naming import (
    canonical_tool_name,
    canonicalize_claude_tool_name,
    canonicalize_gemini_tool_name,
    parse_claude_mcp_tool_name,
    parse_gemini_mcp_tool_name,
)


class CanonicalToolNameTest(unittest.TestCase):

    def test_joins_server_and_tool(self):
        self.assertEqual(
            canonical_tool_name("cloud-sql", "list_instances"),
            "cloud-sql__list_instances",
        )

    def test_returns_bare_tool_when_no_server(self):
        self.assertEqual(canonical_tool_name("", "Read"), "Read")
        self.assertEqual(canonical_tool_name(None, "Read"), "Read")

    def test_empty_tool_returns_empty(self):
        self.assertEqual(canonical_tool_name("cloud-sql", ""), "")


class ParseClaudeMcpToolNameTest(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(
            parse_claude_mcp_tool_name("mcp__cloud-sql__list_instances"),
            ("cloud-sql", "list_instances"),
        )

    def test_server_with_underscore(self):
        # Claude allows underscores in server names; only the first ``__``
        # separates server from tool.
        self.assertEqual(
            parse_claude_mcp_tool_name("mcp__my_server__do_thing"),
            ("my_server", "do_thing"),
        )

    def test_tool_with_double_underscore_preserved(self):
        self.assertEqual(
            parse_claude_mcp_tool_name("mcp__srv__odd__tool"),
            ("srv", "odd__tool"),
        )

    def test_rejects_missing_prefix(self):
        self.assertIsNone(parse_claude_mcp_tool_name("list_instances"))

    def test_rejects_empty_server(self):
        self.assertIsNone(parse_claude_mcp_tool_name("mcp____tool"))

    def test_rejects_no_tool(self):
        self.assertIsNone(parse_claude_mcp_tool_name("mcp__server__"))


class ParseGeminiMcpToolNameTest(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(
            parse_gemini_mcp_tool_name("mcp_cloud-sql_list_instances"),
            ("cloud-sql", "list_instances"),
        )

    def test_tool_with_underscores(self):
        # Server has no underscore by upstream contract; tool may contain
        # several.
        self.assertEqual(
            parse_gemini_mcp_tool_name("mcp_alloydb_create_user_password"),
            ("alloydb", "create_user_password"),
        )

    def test_rejects_missing_prefix(self):
        self.assertIsNone(parse_gemini_mcp_tool_name("list_instances"))

    def test_rejects_server_only(self):
        self.assertIsNone(parse_gemini_mcp_tool_name("mcp_cloudsql"))


class CanonicalizeAdapterFormsTest(unittest.TestCase):

    def test_claude_mcp_becomes_canonical(self):
        self.assertEqual(
            canonicalize_claude_tool_name("mcp__cloud-sql__list_instances"),
            "cloud-sql__list_instances",
        )

    def test_claude_native_tool_passthrough(self):
        self.assertEqual(canonicalize_claude_tool_name("Read"), "Read")
        self.assertEqual(canonicalize_claude_tool_name("Bash"), "Bash")

    def test_claude_malformed_mcp_returned_as_is(self):
        # Falls back to the raw name so callers can debug unexpected inputs
        # instead of silently producing a misleading canonical form.
        self.assertEqual(
            canonicalize_claude_tool_name("mcp__only-server"),
            "mcp__only-server",
        )

    def test_gemini_mcp_becomes_canonical(self):
        self.assertEqual(
            canonicalize_gemini_tool_name("mcp_cloud-sql_list_instances"),
            "cloud-sql__list_instances",
        )

    def test_gemini_native_tool_passthrough(self):
        self.assertEqual(canonicalize_gemini_tool_name("write_file"), "write_file")

    def test_gemini_malformed_mcp_returned_as_is(self):
        self.assertEqual(
            canonicalize_gemini_tool_name("mcp_lonely"),
            "mcp_lonely",
        )


if __name__ == "__main__":
    unittest.main()
