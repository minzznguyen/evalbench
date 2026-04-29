from .generator import QueryGenerator
import subprocess
import os
import json
import logging
import sys


class CLICommand:
    def __init__(self, cli, prompt, env=None, resume=False, session_id=None):
        self.cli = cli
        self.prompt = prompt
        self.env = env if env else {}
        self.resume = resume
        self.session_id = session_id


class CodexCliGenerator(QueryGenerator):
    """Generator queries using OpenAI Codex CLI (`codex exec`)."""

    def __init__(self, querygenerator_config):
        super().__init__(querygenerator_config)
        self.name = "codex_cli"

        self.real_home = os.environ.get("HOME", os.path.expanduser("~"))

        if sys.argv[0].endswith("eval_server.py"):
            session_id = querygenerator_config.get("session_id", "default")
            self.fake_home = os.path.join(
                "/tmp_sessions", session_id, "fake_home")
        else:
            self.fake_home = os.path.abspath(
                os.path.join(".venv", "fake_home_codex"))

        self.codex_config_dir = os.path.join(self.fake_home, ".codex")

        os.makedirs(self.fake_home, exist_ok=True)
        os.makedirs(self.codex_config_dir, exist_ok=True)

        self.env = querygenerator_config.get("env", {})
        self.env["HOME"] = self.fake_home

        api_key = self.env.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.env["OPENAI_API_KEY"] = api_key

        # Copy any existing Codex auth (e.g. auth.json) from the real home so the
        # CLI can authenticate inside the sandboxed environment.
        real_codex_dir = os.path.join(self.real_home, ".codex")
        if os.path.exists(real_codex_dir):
            import shutil
            for fname in ("auth.json",):
                src = os.path.join(real_codex_dir, fname)
                dst = os.path.join(self.codex_config_dir, fname)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)

        self.codex_cli_version = querygenerator_config.get(
            "codex_cli_version", "codex"
        )
        self.model = querygenerator_config.get("model")
        self.sandbox_mode = querygenerator_config.get(
            "sandbox_mode", "danger-full-access")
        self.approval_mode = querygenerator_config.get(
            "approval_mode", "never")
        self.profile = querygenerator_config.get("profile")
        # Codex emits NDJSON events when invoked with --json. Older versions
        # only support --experimental-json; this flag controls which is used.
        self.json_flag = querygenerator_config.get("json_flag", "--json")

        self.setup_config = querygenerator_config.get("setup", {})
        self.config_path = os.path.join(self.codex_config_dir, "config.toml")
        if self.setup_config:
            self._setup()

    def _setup(self):
        """Performs initial setup for Codex CLI (writes ~/.codex/config.toml)."""
        mcp_servers_config = self.setup_config.get("mcp_servers", {})
        extra_config = self.setup_config.get("config", {})
        self._write_config_toml(mcp_servers_config, extra_config)

    def _write_config_toml(self, mcp_servers_config: dict, extra_config: dict):
        """Writes Codex CLI's `config.toml` with MCP server declarations.

        Accepts the same Gemini-style MCP shape the rest of evalbench uses
        (`httpUrl`, `authProviderType: google_credentials`, `headers`) and
        translates it into Codex's TOML schema:

          [mcp_servers.NAME]              # stdio
          command = "..."
          args    = [...]
          env     = { KEY = "VALUE" }

          [mcp_servers.NAME]              # streamable HTTP
          url          = "..."
          http_headers = { KEY = "VALUE" }
        """
        lines: list[str] = []

        for key, value in extra_config.items():
            lines.append(f"{key} = {self._toml_value(value)}")
        if extra_config:
            lines.append("")

        for server_name, config in mcp_servers_config.items():
            translated = self._translate_mcp_config(server_name, dict(config))
            lines.append(f"[mcp_servers.{self._toml_key(server_name)}]")
            for key, value in translated.items():
                lines.append(f"{key} = {self._toml_value(value)}")
            lines.append("")

        with open(self.config_path, "w") as f:
            f.write("\n".join(lines).rstrip() + "\n")

        logging.info(f"Codex CLI config written to {self.config_path}")

    def _translate_mcp_config(self, server_name: str, config: dict) -> dict:
        """Translates a Gemini-style MCP server config into Codex's TOML shape."""
        # stdio server (command + args): pass through fields Codex understands
        if "command" in config:
            out = {"command": config["command"]}
            if "args" in config:
                out["args"] = config["args"]
            if "env" in config:
                out["env"] = config["env"]
            if "cwd" in config:
                out["cwd"] = config["cwd"]
            return out

        # HTTP/streamable server: translate Gemini-style `httpUrl` -> `url`
        url = config.get("url") or config.get("httpUrl")
        if not url:
            logging.warning(
                f"MCP server '{server_name}' has no command or url; skipping translation"
            )
            return config

        out: dict = {"url": url}
        headers = dict(config.get("headers") or {})

        auth_provider = config.get("authProviderType")
        if auth_provider == "google_credentials" and "Authorization" not in headers:
            token = self._fetch_gcloud_access_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            else:
                logging.warning(
                    f"MCP server '{server_name}' requires google_credentials but "
                    "failed to fetch access token via `gcloud auth print-access-token`."
                )
        if headers:
            out["http_headers"] = headers
        return out

    def _fetch_gcloud_access_token(self) -> str:
        try:
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"Failed to retrieve gcloud access token: {e}")
            return ""

    @staticmethod
    def _toml_key(key: str) -> str:
        # Bare TOML keys allow [A-Za-z0-9_-]; quote anything else.
        if all(c.isalnum() or c in "_-" for c in key) and key:
            return key
        return json.dumps(key)

    @classmethod
    def _toml_value(cls, value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return json.dumps(value)
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, list):
            return "[" + ", ".join(cls._toml_value(v) for v in value) + "]"
        if isinstance(value, dict):
            inner = ", ".join(
                f"{cls._toml_key(k)} = {cls._toml_value(v)}"
                for k, v in value.items()
            )
            return "{ " + inner + " }"
        return json.dumps(str(value))

    def generate_internal(self, cli_cmd):
        if not isinstance(cli_cmd, CLICommand):
            cli_cmd = CLICommand(self.codex_cli_version, str(cli_cmd))
        return self._run_codex_cli(cli_cmd)

    def _execute_cli_command(
        self, command: list[str], env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                command, capture_output=True, text=True, check=False, env=env,
                stdin=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return subprocess.CompletedProcess(
                command, 127, "", f"Error: Command not found: {command[0]}"
            )
        except Exception as e:
            return subprocess.CompletedProcess(
                command, 1, "", f"An unexpected error occurred: {e}"
            )

    def _run_codex_cli(self, cli_cmd: CLICommand):
        env = os.environ.copy()
        env.update(self.env)
        env.update(cli_cmd.env)

        # Pin a specific npm version when the spec looks like an npm package.
        cli = cli_cmd.cli
        if cli.startswith("@") or "/" in cli:
            command = ["npm", "exec", "--yes", cli, "--"]
        else:
            command = [cli]

        # `codex exec` runs a single non-interactive turn; `codex exec resume`
        # continues a prior session by id.
        if cli_cmd.resume and cli_cmd.session_id:
            command.extend(["exec", "resume", cli_cmd.session_id])
        else:
            command.append("exec")

        # NDJSON streamed events.
        command.append(self.json_flag)

        # Don't refuse to run outside a git repo.
        command.append("--skip-git-repo-check")

        # Disable approvals + sandbox so MCP/tool calls run unattended. The
        # caller controls the strength via sandbox_mode/approval_mode in the
        # model config; the default is full bypass (matches Gemini's --yolo
        # and Claude Code's --dangerously-skip-permissions).
        if self.sandbox_mode == "danger-full-access":
            command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            command.extend(["--sandbox", self.sandbox_mode])
            if self.approval_mode:
                command.extend(["--ask-for-approval", self.approval_mode])

        if self.model:
            command.extend(["-m", self.model])

        if self.profile:
            command.extend(["--profile", self.profile])

        # Prompt is positional and must come last.
        command.append(cli_cmd.prompt)

        logging.info(f"Running Codex CLI: {' '.join(command)}")

        result = self._execute_cli_command(command, env=env)
        if result.stdout:
            result.stdout = self._parse_stream_json(result.stdout)
        return result

    def _parse_stream_json(self, stream_output: str) -> str:
        """Parses Codex CLI ThreadEvent NDJSON into the eval pipeline's
        normalized {session_id, response, stats} shape."""

        final_obj = {"session_id": "", "response": "", "stats": {}}
        tool_uses: dict[str, dict] = {}
        tool_results: dict[str, dict] = {}
        usage: dict = {}
        model_name = self.model or "unknown"

        def item_kind(item: dict) -> str:
            # Codex's ThreadItem may serialize the variant under "type",
            # "item_type", or as a nested "details": {"type": ...}. Be lenient.
            return (
                item.get("type")
                or item.get("item_type")
                or (item.get("details") or {}).get("type")
                or ""
            )

        def item_payload(item: dict) -> dict:
            # When the variant is nested under `details`, the payload lives there.
            details = item.get("details")
            if isinstance(details, dict) and details.get("type"):
                return details
            return item

        for line in stream_output.split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")

            if event_type == "thread.started":
                final_obj["session_id"] = event.get("thread_id", "") or final_obj["session_id"]
                continue

            if event_type == "turn.completed":
                usage = event.get("usage", {}) or usage
                continue

            if event_type == "error":
                # surface the error text into the response field
                msg = event.get("message", "")
                if msg:
                    final_obj["response"] += (
                        ("\n" if final_obj["response"] else "") + f"[error] {msg}"
                    )
                continue

            if event_type not in ("item.started", "item.updated", "item.completed"):
                continue

            item = event.get("item") or {}
            kind = item_kind(item)
            payload = item_payload(item)
            item_id = item.get("id") or payload.get("id") or ""

            if kind == "agent_message":
                if event_type == "item.completed":
                    text = payload.get("text", "")
                    if text:
                        final_obj["response"] += text

            elif kind == "mcp_tool_call":
                # Record on first sight; refresh on completion.
                tool_uses[item_id] = {
                    "tool_name": payload.get("tool", "unknown"),
                    "server": payload.get("server", ""),
                    "parameters": self._coerce_json(payload.get("arguments", {})),
                }
                if event_type == "item.completed":
                    status = payload.get("status", "")
                    is_error = bool(payload.get("error")) or status not in (
                        "", "completed", "success", "ok",
                    )
                    tool_results[item_id] = {
                        "status": "error" if is_error else "success",
                        "content": payload.get("result", ""),
                    }

            elif kind == "command_execution":
                tool_uses[item_id] = {
                    "tool_name": "shell",
                    "parameters": {"command": payload.get("command", "")},
                }
                if event_type == "item.completed":
                    exit_code = payload.get("exit_code")
                    is_error = bool(exit_code) and exit_code != 0
                    tool_results[item_id] = {
                        "status": "error" if is_error else "success",
                        "content": payload.get("aggregated_output", ""),
                    }

            elif kind == "web_search":
                tool_uses[item_id] = {
                    "tool_name": "web_search",
                    "parameters": {"query": payload.get("query", "")},
                }
                if event_type == "item.completed":
                    tool_results[item_id] = {"status": "success", "content": ""}

            elif kind == "file_change":
                tool_uses[item_id] = {
                    "tool_name": "file_change",
                    "parameters": {"changes": payload.get("changes", [])},
                }
                if event_type == "item.completed":
                    status = payload.get("status", "")
                    is_error = status not in ("", "completed", "success", "ok")
                    tool_results[item_id] = {
                        "status": "error" if is_error else "success",
                        "content": "",
                    }

        # Build aggregate stats so downstream scorers (token_consumption,
        # tool_call_latency, etc.) see the same shape they get from the
        # Claude Code generator.
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cached_tokens = int(usage.get("cached_input_tokens", 0) or 0)
        total_tokens = input_tokens + output_tokens

        models = {
            model_name: {
                "api": {
                    "totalRequests": 1,
                    "totalErrors": 0,
                    "totalLatencyMs": 0,
                },
                "tokens": {
                    "input": input_tokens,
                    "prompt": input_tokens,
                    "candidates": output_tokens,
                    "total": total_tokens,
                    "cached": cached_tokens,
                    "cache_creation": 0,
                    "thoughts": 0,
                    "tool": 0,
                },
                "cost_usd": 0,
                "roles": {
                    "main": {
                        "totalRequests": 1,
                        "totalErrors": 0,
                        "totalLatencyMs": 0,
                        "tokens": {
                            "input": input_tokens,
                            "prompt": input_tokens,
                            "candidates": output_tokens,
                            "total": total_tokens,
                            "cached": cached_tokens,
                            "thoughts": 0,
                            "tool": 0,
                        },
                    }
                },
            }
        }
        final_obj["stats"]["models"] = models

        tools_stats = {
            "totalCalls": len(tool_uses),
            "totalSuccess": sum(
                1 for tr in tool_results.values() if tr.get("status") == "success"
            ),
            "totalFail": sum(
                1 for tr in tool_results.values() if tr.get("status") != "success"
            ),
            "totalDurationMs": 0,
            "decisions": {
                "accept": len(tool_uses),
                "reject": 0,
                "modify": 0,
                "auto_accept": len(tool_uses),
            },
            "byName": {},
        }

        for tid, tu in tool_uses.items():
            tname = tu.get("tool_name", "unknown")
            bucket = tools_stats["byName"].setdefault(tname, {
                "count": 0,
                "success": 0,
                "fail": 0,
                "durationMs": 0,
                "parameters": [],
                "decisions": {
                    "accept": 0, "reject": 0, "modify": 0, "auto_accept": 0,
                },
            })
            bucket["count"] += 1
            bucket["parameters"].append(tu.get("parameters", {}))
            bucket["decisions"]["accept"] += 1
            bucket["decisions"]["auto_accept"] += 1

            tr = tool_results.get(tid)
            if tr:
                if tr.get("status") == "success":
                    bucket["success"] += 1
                else:
                    bucket["fail"] += 1

        final_obj["stats"]["tools"] = tools_stats

        return json.dumps(final_obj, indent=2)

    @staticmethod
    def _coerce_json(value):
        """Codex serializes tool arguments as a JSON-encoded string. Parse it
        back into a dict when possible so scorers see real fields."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return {"raw": value}
        return value or {}

    def parse_response(self, stdout: str) -> dict:
        if not stdout:
            return {}
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            logging.error(f"Failed to parse JSON response: {stdout[:100]}...")
            return {}

    def extract_tools(self, stdout: str) -> list[str]:
        output_json = self.parse_response(stdout)
        if (
            "stats" in output_json
            and "tools" in output_json["stats"]
            and "byName" in output_json["stats"]["tools"]
        ):
            return list(output_json["stats"]["tools"]["byName"].keys())
        return []

    def safe_generate(self, cli_cmd: CLICommand) -> subprocess.CompletedProcess:
        result = self.generate_internal(cli_cmd)
        if isinstance(result, str):
            return subprocess.CompletedProcess(args=[], returncode=0, stdout=result)
        if not result.stdout and result.returncode != 0:
            result.stderr += "\nError: Generator returned empty response."
        return result

    def create_command(
        self, cli: str, prompt: str, env: dict = None, resume: bool = False,
        session_id: str = None,
    ) -> CLICommand:
        merged_env = self.env.copy()
        if env:
            merged_env.update(env)
        return CLICommand(
            cli=cli, prompt=prompt, env=merged_env,
            resume=resume, session_id=session_id,
        )
