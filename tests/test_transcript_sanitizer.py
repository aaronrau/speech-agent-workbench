import json
import os
import tempfile
import time
import unittest
from unittest import mock

from app import (
    append_transcript_history,
    arm_auto_trigger_session,
    auto_shell_command_has_explicit_audio,
    build_command_text_aliases,
    build_transcript_correction_messages,
    build_auto_tmux_switch_commands,
    build_sherpa_vad,
    correct_common_coding_terms,
    correct_transcript_details,
    correct_transcript_text,
    extract_text_after_trigger_word,
    format_colored_detection_words,
    get_transcript_correction_backend,
    get_transcript_history_path,
    is_auto_trigger_session_armed,
    is_likely_bad_transcript,
    make_auto_trigger_session,
    match_auto_shell_command,
    match_auto_shell_command_prefix,
    normalize_voice_command_text,
    parse_word_list,
    reset_auto_trigger_session,
    sanitize_transcript_text,
    split_trailing_submit_command,
)
import app


def load_example_transcript_correction_prompt():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(
        os.path.join(repo_root, "config.example.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)["transcript_correction_prompt"]


class SanitizeTranscriptTextTests(unittest.TestCase):
    def test_keeps_alphanumeric_period_plus_and_spaces(self):
        self.assertEqual(
            sanitize_transcript_text("Call me at 555.123+45"),
            "Call me at 555.123+45",
        )

    def test_filters_other_symbols(self):
        self.assertEqual(
            sanitize_transcript_text("alpha@beta#gamma$delta"),
            "alpha beta gamma delta",
        )

    def test_collapses_whitespace_after_filtering(self):
        self.assertEqual(
            sanitize_transcript_text("foo,\n\tbar! baz?"),
            "foo bar baz",
        )

    def test_split_trailing_submit_command_removes_final_word(self):
        self.assertEqual(
            split_trailing_submit_command("run this send"),
            ("run this", True),
        )

    def test_split_trailing_submit_command_handles_enter_only(self):
        self.assertEqual(split_trailing_submit_command("Send"), ("", True))

    def test_split_trailing_submit_command_ignores_non_final_trigger(self):
        self.assertEqual(
            split_trailing_submit_command("send later"),
            ("send later", False),
        )

    def test_split_trailing_submit_command_ignores_embedded_suffix(self):
        self.assertEqual(
            split_trailing_submit_command("resend"),
            ("resend", False),
        )

    def test_split_trailing_submit_command_allows_sanitized_period(self):
        self.assertEqual(
            split_trailing_submit_command("submit this send."),
            ("submit this", True),
        )

    def test_trim_auto_tmux_console_log_drops_old_records(self):
        now = int(time.time())
        old = now - 7200
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            path = handle.name
            handle.write(f"{old}\t[tmux][Flux] stale response\n")
            handle.write(f"{now}\t[tmux][Flux] current response\n")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        app.trim_auto_tmux_console_log(
            path,
            retention_seconds=3600,
            max_bytes=1024,
        )

        with open(path, "r", encoding="utf-8") as handle:
            log_text = handle.read()
        self.assertNotIn("stale response", log_text)
        self.assertIn("current response", log_text)

    def test_split_tmux_console_log_record_accepts_chunk_record(self):
        line = json.dumps(
            {
                "ts": 123.5,
                "data": "[tmux][Flux] partial response",
            },
            separators=(",", ":"),
        )

        timestamp, payload = app.split_tmux_console_log_record(line)

        self.assertEqual(timestamp, 123.5)
        self.assertEqual(payload, "[tmux][Flux] partial response")

    def test_extract_tmux_console_agent_payload_removes_prefixes(self):
        label, payload = app.extract_tmux_console_agent_payload(
            "[tmux][Brock] line one\n[tmux][Brock] line two\n",
        )

        self.assertEqual(label, "Brock")
        self.assertEqual(payload, "line one\nline two\n")

    def test_tmux_console_buffer_flushes_only_after_idle(self):
        buffer = ["[tmux][Flux] first", " second"]
        self.assertFalse(
            app.tmux_console_buffer_should_flush(
                buffer,
                last_update_at=10.0,
                idle_seconds=5.0,
                now=14.9,
            )
        )
        self.assertTrue(
            app.tmux_console_buffer_should_flush(
                buffer,
                last_update_at=10.0,
                idle_seconds=5.0,
                now=15.0,
            )
        )

    def test_tmux_recent_command_matches_display_label(self):
        app.TMUX_RECENT_COMMANDS.clear()
        self.addCleanup(app.TMUX_RECENT_COMMANDS.clear)

        app.record_tmux_sent_command("brock", "%2", "add focused tests")

        self.assertEqual(
            app.get_tmux_sent_command("Brock")["text"],
            "add focused tests",
        )
        self.assertEqual(
            app.get_tmux_sent_command(target="%2")["text"],
            "add focused tests",
        )

    def test_build_tmux_summary_messages_include_command_and_lines(self):
        messages = app.build_tmux_summary_messages(
            "Brock",
            "add focused tests",
            ["edited app.py", "Ran 72 tests OK"],
        )

        prompt = messages[-1]["content"]
        self.assertIn("Agent: Brock", prompt)
        self.assertIn("Original command: add focused tests", prompt)
        self.assertIn("edited app.py", prompt)
        self.assertIn("Ran 72 tests OK", prompt)

    def test_agent_completion_record_formats_console_line(self):
        record = app.parse_agent_completion_record(
            json.dumps(
                {
                    "agent": "Brock",
                    "status": "done",
                    "message": "tests passed",
                }
            )
        )

        self.assertEqual(
            app.format_agent_completion_record(record),
            "[agent-complete][Brock] done: tests passed",
        )

    def test_parse_api_message_payload_accepts_agent_prefix(self):
        agent, message, error = app.parse_api_message_payload(
            b'{"message":"flux: pull the latest"}',
            "application/json",
        )

        self.assertIsNone(error)
        self.assertEqual(agent, "flux")
        self.assertEqual(message, "pull the latest")

    def test_parse_api_message_payload_accepts_plain_text_without_content_type(self):
        agent, message, error = app.parse_api_message_payload(
            b"flux: pull the latest",
            "",
        )

        self.assertIsNone(error)
        self.assertEqual(agent, "flux")
        self.assertEqual(message, "pull the latest")

    def test_route_api_message_to_tmux_sends_known_agent(self):
        commands = {
            "flux": {
                "label": "flux",
                "tmux_send_target": "%1",
                "argv": ["tmux", "select-pane", "-t", "%1"],
            }
        }

        with mock.patch.object(app, "send_text_to_tmux_target", return_value=True) as sent:
            result = app.route_api_message_to_tmux(
                "Flux",
                "pull the latest",
                commands,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["agent"], "flux")
        self.assertEqual(result["message"], "pull the latest")
        sent.assert_called_once_with(commands["flux"], "pull the latest")

    def test_route_api_message_to_tmux_reports_unknown_agent(self):
        commands = {
            "flux": {
                "label": "flux",
                "tmux_send_target": "%1",
                "argv": ["tmux", "select-pane", "-t", "%1"],
            }
        }

        result = app.route_api_message_to_tmux(
            "Unknown",
            "pull the latest",
            commands,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "unknown_agent")
        self.assertEqual(result["available_agents"], ["flux"])

    def test_post_tmux_summary_webhook_sends_json_payload(self):
        config = {
            "tmux_summary_webhook_url": "http://127.0.0.1:9999/hook",
            "tmux_summary_webhook_token": "secret",
            "tmux_summary_webhook_timeout": 1.0,
        }
        requests = []

        class Response:
            status = 204

            def __enter__(self):
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                return False

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return Response()

        with mock.patch.object(app.urllib.request, "urlopen", side_effect=fake_urlopen):
            self.assertTrue(
                app.post_tmux_summary_webhook(
                    config,
                    "Flux",
                    "pull the latest",
                    "Flux pulled the latest changes.",
                )
            )

        request, timeout = requests[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(timeout, 1.0)
        self.assertEqual(request.full_url, "http://127.0.0.1:9999/hook")
        self.assertEqual(request.headers["Authorization"], "Bearer secret")
        self.assertEqual(payload["agent"], "Flux")
        self.assertEqual(payload["command"], "pull the latest")
        self.assertEqual(payload["summary"], "Flux pulled the latest changes.")

    def test_redact_url_for_log_hides_webhook_secrets(self):
        url = "https://user:pass@example.test/hook?token=abc&api_key=xyz&mode=dev"
        self.assertEqual(
            app.redact_url_for_log(url),
            "https://***@example.test/hook?token=***&api_key=***&mode=dev",
        )

    def test_build_voice_api_post_url_uses_loopback_for_wildcard_bind(self):
        self.assertEqual(
            app.build_voice_api_post_url("0.0.0.0", 8787),
            "http://127.0.0.1:8787/messages",
        )

    def test_trim_auto_tmux_console_log_caps_size(self):
        now = int(time.time())
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
            path = handle.name
            for index in range(20):
                handle.write(f"{now}\t[tmux][Pike] response line {index:02d}\n")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))

        app.trim_auto_tmux_console_log(
            path,
            retention_seconds=0,
            max_bytes=120,
        )

        with open(path, "r", encoding="utf-8") as handle:
            log_text = handle.read()
        self.assertLessEqual(os.path.getsize(path), 120)
        self.assertIn("response line 19", log_text)

    def test_extract_text_after_trigger_word_requires_trigger_at_start(self):
        self.assertIsNone(
            extract_text_after_trigger_word("ignore this agent create issue", "agent")
        )

    def test_extract_text_after_trigger_word_keeps_words_after_start_trigger(self):
        self.assertEqual(
            extract_text_after_trigger_word("agent create issue", "agent"),
            "create issue",
        )

    def test_extract_text_after_trigger_word_handles_trigger_at_start(self):
        self.assertEqual(
            extract_text_after_trigger_word("Agent create issue", "agent"),
            "create issue",
        )

    def test_extract_text_after_trigger_word_ignores_missing_trigger(self):
        self.assertIsNone(
            extract_text_after_trigger_word("create issue", "agent")
        )

    def test_extract_text_after_trigger_word_ignores_embedded_trigger(self):
        self.assertIsNone(
            extract_text_after_trigger_word("reagent create issue", "agent")
        )

    def test_extract_text_after_trigger_word_allows_no_queued_text(self):
        self.assertEqual(extract_text_after_trigger_word("agent.", "agent"), "")

    def test_extract_text_after_trigger_word_accepts_aliases(self):
        self.assertEqual(
            extract_text_after_trigger_word("assistant create issue", "agent", ["assistant"]),
            "create issue",
        )

    def test_parse_word_list_accepts_comma_separated_aliases(self):
        self.assertEqual(parse_word_list("assistant, helper, assistant"), ["assistant", "helper"])

    def test_auto_trigger_session_expires_stale_armed_state(self):
        session = make_auto_trigger_session()

        arm_auto_trigger_session(session, "trigger_only", now=10.0)

        self.assertFalse(is_auto_trigger_session_armed(session, 8.0, now=18.1))
        self.assertFalse(session["armed"])
        self.assertFalse(session["clicked"])
        self.assertFalse(session["focus_failed"])
        self.assertIsNone(session["source"])

    def test_auto_trigger_session_keeps_recent_armed_state(self):
        session = make_auto_trigger_session()

        arm_auto_trigger_session(session, "trigger_only", now=10.0)

        self.assertTrue(is_auto_trigger_session_armed(session, 8.0, now=17.9))
        self.assertTrue(session["armed"])
        self.assertEqual(session["source"], "trigger_only")

    def test_auto_trigger_session_tracks_focus_failure(self):
        session = make_auto_trigger_session()

        arm_auto_trigger_session(
            session,
            "probe",
            now=10.0,
            focus_success=False,
        )

        self.assertTrue(session["armed"])
        self.assertFalse(session["clicked"])
        self.assertTrue(session["focus_failed"])
        self.assertEqual(session["source"], "probe")

    def test_auto_trigger_session_timeout_zero_disarms(self):
        session = make_auto_trigger_session()

        arm_auto_trigger_session(session, "trigger_only", now=10.0)

        self.assertFalse(is_auto_trigger_session_armed(session, 0.0, now=10.1))
        self.assertFalse(session["armed"])
        self.assertFalse(session["focus_failed"])

    def test_reset_auto_trigger_session_clears_metadata(self):
        session = make_auto_trigger_session()
        arm_auto_trigger_session(session, "probe", now=10.0)

        reset_auto_trigger_session(session)

        self.assertEqual(session, make_auto_trigger_session())

    def test_normalize_voice_command_text_keeps_command_words(self):
        self.assertEqual(
            normalize_voice_command_text("Switch to Agent Two."),
            "switch to agent two",
        )

    def test_format_colored_detection_words_can_disable_color(self):
        with mock.patch.dict("os.environ", {"VOICE_AUTO_COLOR_NAMES": "0"}):
            self.assertEqual(
                format_colored_detection_words(),
                "agent, agent two, agent three, voice",
            )

    def test_auto_tmux_switch_commands_build_from_environment(self):
        with mock.patch.dict(
            "os.environ",
            {
                "VOICE_AUTO_TMUX_SESSION": "speech-agent-workbench",
                "VOICE_AUTO_TMUX_SWITCHES": "agent=Agent 1,agent two=Agent 2",
            },
            clear=True,
        ):
            commands = build_auto_tmux_switch_commands({})

        self.assertEqual(
            commands["agent two"]["argv"],
            ["tmux", "select-window", "-t", "speech-agent-workbench:Agent 2"],
        )
        self.assertEqual(
            commands["agent two"]["tmux_send_target"],
            "speech-agent-workbench:Agent 2",
        )

    def test_auto_tmux_switch_commands_accept_pane_targets(self):
        with mock.patch.dict(
            "os.environ",
            {
                "VOICE_AUTO_TMUX_SESSION": "speech-agent-workbench",
                "VOICE_AUTO_TMUX_SWITCHES": "agent three=pane:speech-agent-workbench:Workbench.2",
            },
            clear=True,
        ):
            commands = build_auto_tmux_switch_commands({})

        self.assertEqual(
            commands["agent three"]["argv"],
            [
                "tmux",
                "select-window",
                "-t",
                "speech-agent-workbench:Workbench",
                ";",
                "select-pane",
                "-t",
                "speech-agent-workbench:Workbench.2",
            ],
        )
        self.assertEqual(
            commands["agent three"]["tmux_send_target"],
            "speech-agent-workbench:Workbench.2",
        )

    def test_auto_tmux_switch_commands_accept_pane_id_targets(self):
        with mock.patch.dict(
            "os.environ",
            {
                "VOICE_AUTO_TMUX_SESSION": "speech-agent-workbench",
                "VOICE_AUTO_TMUX_SWITCHES": "agent two=pane:%12",
            },
            clear=True,
        ):
            commands = build_auto_tmux_switch_commands({})

        self.assertEqual(
            commands["agent two"]["argv"],
            ["tmux", "select-pane", "-t", "%12"],
        )
        self.assertEqual(commands["agent two"]["tmux_send_target"], "%12")
        self.assertEqual(commands["agent to"]["tmux_send_target"], "%12")
        self.assertEqual(commands["agent too"]["tmux_send_target"], "%12")
        self.assertEqual(commands["agent 2"]["tmux_send_target"], "%12")

    def test_build_command_text_aliases_includes_codex_homophones(self):
        self.assertIn("code x", build_command_text_aliases("codex"))
        self.assertIn("condex", build_command_text_aliases("codex"))

    def test_build_command_text_aliases_includes_agent_homophones(self):
        self.assertIn("block", build_command_text_aliases("brock"))
        self.assertIn("flex", build_command_text_aliases("flux"))
        self.assertIn("pipe", build_command_text_aliases("pike"))
        commands = {"flex": {"label": "flux", "argv": ["tmux"]}}
        self.assertEqual(
            match_auto_shell_command_prefix(
                "Hey Flex what are the daily active users",
                commands,
            ),
            (commands["flex"], "what are the daily active users"),
        )

    def test_transcript_correction_prompt_mentions_pipe_for_pike(self):
        messages = build_transcript_correction_messages(
            "Pipe check the latest branch",
            ["pike"],
            {
                "transcript_correction_prompt": (
                    load_example_transcript_correction_prompt()
                )
            },
        )

        self.assertIn("pipe", messages[0]["content"].lower())
        self.assertIn("write Pike", messages[0]["content"])

    def test_transcript_correction_prompt_mentions_evals_mishears(self):
        messages = build_transcript_correction_messages(
            "run the evalues",
            [],
            {
                "transcript_correction_prompt": (
                    load_example_transcript_correction_prompt()
                )
            },
        )

        prompt = messages[0]["content"].lower()
        self.assertIn("yaws", prompt)
        self.assertIn("evalues", prompt)
        self.assertIn("write evals", prompt)

    def test_correct_common_coding_terms_fixes_codex_and_tmux(self):
        self.assertEqual(
            correct_common_coding_terms("ask condex to inspect tea mux"),
            "ask Codex to inspect tmux",
        )

    def test_correct_common_coding_terms_fixes_langfuse_homophones(self):
        self.assertEqual(
            correct_common_coding_terms("open the length view trace"),
            "open the Langfuse trace",
        )

    def test_correct_common_coding_terms_fixes_evals_mishears(self):
        self.assertEqual(
            correct_common_coding_terms("run the yaws"),
            "run the EVALS",
        )
        self.assertEqual(
            correct_common_coding_terms("run evalues and e vals"),
            "run EVALS and EVALS",
        )

    def test_correct_common_coding_terms_fixes_dev_push_phrase(self):
        self.assertEqual(
            correct_common_coding_terms("hen all the chains push to death"),
            "did all the changes get pushed to dev",
        )
        self.assertEqual(
            correct_common_coding_terms("did all the change got pushed to dev"),
            "did all the changes get pushed to dev",
        )

    def test_correct_transcript_text_uses_common_terms_when_model_disabled(self):
        self.assertEqual(
            correct_transcript_text(
                "code x check git hub",
                {"transcript_correction_backend": "off"},
            ),
            "Codex check GitHub",
        )

    def test_gemma_backend_alias_uses_llama_cpp(self):
        self.assertEqual(
            get_transcript_correction_backend(
                {"transcript_correction_backend": "gemma"}
            ),
            "llama-cpp",
        )

    def test_correct_transcript_text_uses_llama_cpp_backend(self):
        completed = mock.Mock(
            returncode=0,
            stdout=(
                "Loading model...\n"
                "> Raw transcript: ask Langfuse to inspect Codex trace\n"
                "|\b \bask Langfuse to inspect the Codex trace\n"
                "Exiting...\n"
            ),
            stderr="",
        )
        config = {
            "transcript_correction_backend": "llama.cpp",
            "transcript_correction_llama_cpp_path": "/tmp/llama-cli",
            "transcript_correction_llama_cpp_model": "/tmp/model.gguf",
            "transcript_correction_llama_cpp_gpu_layers": 99,
            "transcript_correction_llama_cpp_timeout": 3.0,
            "transcript_correction_max_new_tokens": 32,
            "transcript_correction_console_log": False,
        }

        app.TRANSCRIPT_CORRECTION_FAILURES.clear()
        with mock.patch.object(
            app,
            "correct_transcript_with_llama_cpp_server",
            side_effect=RuntimeError("server unavailable"),
        ):
            with mock.patch.object(
                app.subprocess,
                "run",
                return_value=completed,
            ) as run:
                result = correct_transcript_text(
                    "ask length view to inspect code x trace",
                    config,
                )

        self.assertEqual(result, "ask Langfuse to inspect the Codex trace")
        argv = run.call_args.args[0]
        self.assertEqual(argv[0], "/tmp/llama-cli")
        self.assertIn("/tmp/model.gguf", argv)
        self.assertIn("--reasoning", argv)
        self.assertIn("-ngl", argv)

    def test_correct_transcript_text_prefers_llama_cpp_server(self):
        config = {
            "transcript_correction_backend": "llama.cpp",
            "transcript_correction_llama_cpp_model": "/tmp/model.gguf",
            "transcript_correction_max_new_tokens": 32,
            "transcript_correction_console_log": False,
        }

        app.TRANSCRIPT_CORRECTION_FAILURES.clear()
        with mock.patch.object(
            app,
            "correct_transcript_with_llama_cpp_server",
            return_value="Hey Flux what are the daily active users for today",
        ) as server:
            with mock.patch.object(app.subprocess, "run") as run:
                result = correct_transcript_text(
                    "Hey Flex what are the daily active users for today",
                    config,
                    command_labels=["brock", "flux", "pike", "wolf"],
                )

        self.assertEqual(
            result,
            "Hey Flux what are the daily active users for today",
        )
        server.assert_called_once()
        run.assert_not_called()

    def test_correct_transcript_details_records_raw_and_corrected_text(self):
        config = {
            "transcript_correction_backend": "llama.cpp",
            "transcript_correction_llama_cpp_model": "/tmp/model.gguf",
            "transcript_correction_max_new_tokens": 32,
            "transcript_correction_console_log": False,
        }

        app.TRANSCRIPT_CORRECTION_FAILURES.clear()
        with mock.patch.object(
            app,
            "correct_transcript_with_llama_cpp_server",
            return_value="ask Langfuse to inspect the Codex trace",
        ):
            details = correct_transcript_details(
                "ask length view to inspect code x trace",
                config,
            )

        self.assertEqual(
            details["raw_transcript"],
            "ask length view to inspect code x trace",
        )
        self.assertEqual(
            details["pre_llm_transcript"],
            "ask Langfuse to inspect Codex trace",
        )
        self.assertEqual(
            details["corrected_transcript"],
            "ask Langfuse to inspect the Codex trace",
        )
        self.assertEqual(
            details["model_output"],
            "ask Langfuse to inspect the Codex trace",
        )
        self.assertTrue(details["model_accepted"])

    def test_correct_transcript_details_rejects_model_output_when_raw_empty(self):
        config = {
            "transcript_correction_backend": "llama.cpp",
            "transcript_correction_llama_cpp_model": "/tmp/model.gguf",
            "transcript_correction_max_new_tokens": 32,
            "transcript_correction_console_log": False,
        }

        app.TRANSCRIPT_CORRECTION_FAILURES.clear()
        with mock.patch.object(
            app,
            "correct_transcript_with_llama_cpp_server",
            return_value="wolf terminate session",
        ):
            details = correct_transcript_details(
                "",
                config,
                command_labels=["brock", "flux", "pike", "wolf"],
            )

        self.assertEqual(details["raw_transcript"], "")
        self.assertEqual(details["pre_llm_transcript"], "")
        self.assertEqual(details["corrected_transcript"], "")
        self.assertEqual(details["model_output"], "wolf terminate session")
        self.assertFalse(details["model_accepted"])
        self.assertEqual(details["fallback_reason"], "implausible_model_output")

    def test_correct_transcript_details_logs_llama_cpp_translation(self):
        config = {
            "transcript_correction_backend": "llama.cpp",
            "transcript_correction_llama_cpp_model": "/tmp/model.gguf",
            "transcript_correction_max_new_tokens": 32,
            "transcript_correction_console_log": True,
        }

        app.TRANSCRIPT_CORRECTION_FAILURES.clear()
        with mock.patch.object(
            app,
            "correct_transcript_with_llama_cpp_server",
            return_value="Hey Flux check the deploy status",
        ):
            with mock.patch("builtins.print") as printed:
                details = correct_transcript_details(
                    "Hey Flex check the deploy status",
                    config,
                    command_labels=["flux"],
                )

        self.assertEqual(
            details["corrected_transcript"],
            "Hey Flux check the deploy status",
        )
        lines = [
            " ".join(str(arg) for arg in call.args)
            for call in printed.call_args_list
        ]
        self.assertTrue(
            any(
                line.startswith("[transcribe] transcript correction raw:")
                and "Hey Flex check the deploy status" in line
                for line in lines
            )
        )
        self.assertTrue(
            any(
                line.startswith("[transcribe] transcript correction llama.cpp accepted:")
                and "Hey Flux check the deploy status" in line
                for line in lines
            )
        )

    def test_auto_tmux_switch_commands_include_configured_terminate_commands(self):
        with mock.patch.dict(
            "os.environ",
            {
                "VOICE_AUTO_TMUX_SESSION": "speech-agent-workbench",
                "VOICE_AUTO_TMUX_SWITCHES": "voice=pane:%4",
                "VOICE_AUTO_ENABLE_TERMINATE_COMMANDS": "1",
                "VOICE_AUTO_TMUX_TERMINATE_WORDS": (
                    "voice terminate session,voice terminates session,"
                    "voice terminate sessions,voice terminates sessions"
                ),
            },
            clear=True,
        ):
            commands = build_auto_tmux_switch_commands({})

        command = commands["voice terminate session"]
        self.assertEqual(
            command["argv"],
            ["tmux", "kill-session", "-t", "speech-agent-workbench"],
        )
        self.assertNotIn("tmux_send_target", command)
        self.assertTrue(command["exit_after"])
        self.assertFalse(command["allow_prefix"])
        self.assertTrue(command["requires_explicit_audio"])
        for label in (
            "voice terminates session",
            "voice terminate sessions",
            "voice terminates sessions",
        ):
            self.assertEqual(commands[label]["argv"], command["argv"])
            self.assertTrue(commands[label]["exit_after"])

    def test_terminate_command_requires_explicit_raw_audio(self):
        command = {
            "label": "wolf terminate session",
            "argv": ["tmux", "kill-session"],
            "requires_explicit_audio": True,
        }

        self.assertFalse(
            auto_shell_command_has_explicit_audio(
                command,
                {
                    "raw_transcript": "",
                    "pre_llm_transcript": "",
                    "corrected_transcript": "wolf terminate session",
                },
            )
        )
        self.assertFalse(
            auto_shell_command_has_explicit_audio(
                command,
                {
                    "raw_transcript": "Hey Wall.",
                    "pre_llm_transcript": "Hey Wall.",
                    "corrected_transcript": "wolf terminate session",
                },
            )
        )
        self.assertTrue(
            auto_shell_command_has_explicit_audio(
                command,
                {
                    "raw_transcript": "Hey Wolf terminate session.",
                    "pre_llm_transcript": "Hey Wolf terminate session.",
                    "corrected_transcript": "wolf terminate session",
                },
            )
        )
        self.assertTrue(
            auto_shell_command_has_explicit_audio(
                command,
                {
                    "raw_transcript": "Hey Wolf terminate session please.",
                    "pre_llm_transcript": "Hey Wolf terminate session please.",
                    "corrected_transcript": "wolf terminate session",
                },
            )
        )
        self.assertFalse(
            auto_shell_command_has_explicit_audio(
                command,
                {
                    "raw_transcript": "Hey Wolf terminate session after tests.",
                    "pre_llm_transcript": "Hey Wolf terminate session after tests.",
                    "corrected_transcript": "wolf terminate session",
                },
            )
        )

    def test_auto_tmux_switch_commands_disable_terminate_commands_by_default(self):
        with mock.patch.dict(
            "os.environ",
            {
                "VOICE_AUTO_TMUX_SESSION": "speech-agent-workbench",
                "VOICE_AUTO_TMUX_SWITCHES": "voice=pane:%4",
                "VOICE_AUTO_TMUX_TERMINATE_WORDS": "voice terminate session",
            },
            clear=True,
        ):
            commands = build_auto_tmux_switch_commands({})

        self.assertIn("voice", commands)
        self.assertNotIn("voice terminate session", commands)

    def test_auto_tmux_terminate_commands_expand_session_variants(self):
        with mock.patch.dict(
            "os.environ",
            {
                "VOICE_AUTO_TMUX_SESSION": "speech-agent-workbench",
                "VOICE_AUTO_TMUX_SWITCHES": "voice=pane:%4",
                "VOICE_AUTO_ENABLE_TERMINATE_COMMANDS": "1",
                "VOICE_AUTO_TMUX_TERMINATE_WORDS": (
                    "voice confirm terminate session"
                ),
            },
            clear=True,
        ):
            commands = build_auto_tmux_switch_commands({})

        command = commands["voice confirm terminate session"]
        for label in (
            "voice confirm terminates session",
            "voice confirm terminate sessions",
            "voice confirm terminates sessions",
        ):
            self.assertEqual(commands[label]["argv"], command["argv"])
            self.assertFalse(commands[label]["allow_prefix"])

    def test_match_auto_shell_command_accepts_exact_switch_word(self):
        commands = {"agent two": {"label": "agent two", "argv": ["tmux"]}}

        self.assertEqual(
            match_auto_shell_command("Agent Two.", commands),
            commands["agent two"],
        )

    def test_match_auto_shell_command_accepts_switch_prefix(self):
        commands = {"agent three": {"label": "agent three", "argv": ["tmux"]}}

        self.assertEqual(
            match_auto_shell_command("switch to the agent three terminal", commands),
            commands["agent three"],
        )

    def test_match_auto_shell_command_ignores_dictation_after_trigger(self):
        commands = {"agent": {"label": "agent", "argv": ["tmux"]}}

        self.assertIsNone(
            match_auto_shell_command("agent update the tests", commands)
        )

    def test_match_auto_shell_command_prefix_skips_exact_only_commands(self):
        commands = {
            "voice confirm terminate session": {
                "label": "voice confirm terminate session",
                "argv": ["tmux", "kill-session"],
                "allow_prefix": False,
            }
        }

        self.assertIsNone(
            match_auto_shell_command_prefix(
                "Voice confirm terminate session please",
                commands,
            )
        )
        self.assertEqual(
            match_auto_shell_command("Voice confirm terminate session", commands),
            commands["voice confirm terminate session"],
        )
        self.assertEqual(
            match_auto_shell_command(
                "Voice confirm terminate session please",
                commands,
            ),
            commands["voice confirm terminate session"],
        )
        self.assertIsNone(
            match_auto_shell_command(
                "Voice confirm terminate session after tests",
                commands,
            )
        )

    def test_exact_only_command_beats_shorter_prefix_command(self):
        commands = {
            "wolf": {
                "label": "wolf",
                "argv": ["tmux", "select-pane"],
            },
            "wolf terminate session": {
                "label": "wolf terminate session",
                "argv": ["tmux", "kill-session"],
                "allow_prefix": False,
            },
        }

        self.assertIsNone(
            match_auto_shell_command_prefix("Wolf terminate session", commands)
        )
        self.assertEqual(
            match_auto_shell_command("Wolf terminate session", commands),
            commands["wolf terminate session"],
        )

    def test_match_auto_shell_command_prefix_accepts_target_and_message(self):
        commands = {"agent two": {"label": "agent two", "argv": ["tmux"]}}

        self.assertEqual(
            match_auto_shell_command_prefix(
                "Agent Two what are the current changes",
                commands,
            ),
            (commands["agent two"], "what are the current changes"),
        )

    def test_match_auto_shell_command_prefix_handles_punctuation(self):
        commands = {"agent three": {"label": "agent three", "argv": ["tmux"]}}

        self.assertEqual(
            match_auto_shell_command_prefix(
                "Agent Three. What are the latest changes",
                commands,
            ),
            (commands["agent three"], "What are the latest changes"),
        )

    def test_match_auto_shell_command_prefix_ignores_attention_word(self):
        commands = {"workspace": {"label": "workspace", "argv": ["tmux"]}}

        self.assertEqual(
            match_auto_shell_command_prefix(
                "Hey Workspace do you have the latest changes",
                commands,
            ),
            (commands["workspace"], "do you have the latest changes"),
        )

    def test_focus_auto_terminal_window_does_not_launch_terminal_by_default_on_gnome_wayland(self):
        calls = []

        def fake_which(name):
            if name in ("gdbus", "gnome-terminal"):
                return f"/usr/bin/{name}"
            return None

        def fake_run_command(argv, input_text=None, timeout=None):
            calls.append(argv)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "focus.log")
            with mock.patch.dict(
                "os.environ",
                {
                    "DESKTOP_SESSION": "ubuntu",
                    "VOICE_AUTO_FOCUS_LOG": log_path,
                    "VOICE_AUTO_REFOCUS_DELAY": "0",
                    "VOICE_AUTO_TMUX_SESSION": "speech-agent-workbench",
                    "XDG_CURRENT_DESKTOP": "ubuntu:GNOME",
                    "XDG_SESSION_TYPE": "wayland",
                },
                clear=True,
            ):
                with mock.patch.object(app.shutil, "which", side_effect=fake_which):
                    with mock.patch.object(app, "run_command", side_effect=fake_run_command):
                        self.assertFalse(app.focus_auto_terminal_window())

            with open(log_path, "r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]

        self.assertNotIn("gnome-terminal", [call[0] for call in calls])
        self.assertEqual(records[-2]["method"], "gnome-focus-mode")
        self.assertEqual(records[-2]["target"], "off")
        self.assertFalse(records[-1]["success"])

    def test_focus_auto_terminal_window_launches_terminal_when_enabled_on_gnome_wayland(self):
        calls = []

        def fake_which(name):
            if name == "gnome-terminal":
                return f"/usr/bin/{name}"
            return None

        def fake_run_command(argv, input_text=None, timeout=None):
            calls.append(argv)
            if argv[:2] == ["tmux", "has-session"]:
                return mock.Mock(returncode=0, stdout="", stderr="")
            if argv[:2] == ["gnome-terminal", "--title"]:
                return mock.Mock(returncode=0, stdout="", stderr="")
            return mock.Mock(returncode=1, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = os.path.join(tmp_dir, "focus.log")
            with mock.patch.dict(
                "os.environ",
                {
                    "DESKTOP_SESSION": "ubuntu",
                    "VOICE_AUTO_FOCUS_LOG": log_path,
                    "VOICE_AUTO_GNOME_TERMINAL_FOCUS_MODE": "launch",
                    "VOICE_AUTO_REFOCUS_DELAY": "0",
                    "VOICE_AUTO_TMUX_SESSION": "speech-agent-workbench",
                    "XDG_CURRENT_DESKTOP": "ubuntu:GNOME",
                    "XDG_SESSION_TYPE": "wayland",
                },
                clear=True,
            ):
                with mock.patch.object(app.shutil, "which", side_effect=fake_which):
                    with mock.patch.object(app, "run_command", side_effect=fake_run_command):
                        self.assertTrue(app.focus_auto_terminal_window())

            with open(log_path, "r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]

        self.assertIn(
            [
                "gnome-terminal",
                "--title",
                "speech-agent-workbench",
                "--",
                "tmux",
                "attach-session",
                "-t",
                "speech-agent-workbench",
            ],
            calls,
        )
        self.assertEqual(records[-1]["method"], "gnome-terminal-launch")
        self.assertTrue(records[-1]["success"])

    def test_send_text_to_tmux_target_pastes_buffer_and_enters(self):
        calls = []

        def fake_which(name):
            return "/usr/bin/tmux" if name == "tmux" else None

        def fake_run_command(argv, input_text=None, timeout=None):
            calls.append(argv)
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.dict("os.environ", {"VOICE_SUBMIT_ENTER_DELAY": "0"}):
            with mock.patch.object(app.shutil, "which", side_effect=fake_which):
                with mock.patch.object(app, "run_command", side_effect=fake_run_command):
                    self.assertTrue(
                        app.send_text_to_tmux_target(
                            {"label": "agent two", "tmux_send_target": "%1"},
                            "-starts with dash",
                        )
                    )

        buffer_name = calls[0][3]
        self.assertTrue(buffer_name.startswith(f"voice-workbench-{os.getpid()}-"))
        self.assertEqual(
            calls,
            [
                [
                    "tmux",
                    "set-buffer",
                    "-b",
                    buffer_name,
                    "--",
                    "-starts with dash",
                ],
                ["tmux", "paste-buffer", "-d", "-b", buffer_name, "-t", "%1"],
                ["tmux", "send-keys", "-t", "%1", "C-m"],
            ],
        )

    def test_append_transcript_history_appends_successful_text(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "history.txt")

            self.assertTrue(append_transcript_history("first transcript", path))
            self.assertTrue(append_transcript_history("second transcript", path))

            with open(path, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()

        self.assertEqual(
            [line.split("\t", 1)[1] for line in lines],
            ["first transcript", "second transcript"],
        )

    def test_append_transcript_history_can_store_correction_details(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "history.txt")

            self.assertTrue(
                append_transcript_history(
                    "inspect the trace",
                    path,
                    correction={
                        "raw_transcript": "flux inspect length view trace",
                        "pre_llm_transcript": "flux inspect Langfuse trace",
                        "corrected_transcript": "Flux inspect Langfuse trace",
                        "correction_backend": "llama-cpp",
                        "model_output": "Flux inspect Langfuse trace",
                        "model_accepted": True,
                    },
                )
            )

            with open(path, "r", encoding="utf-8") as handle:
                line = handle.read().strip()

        _timestamp, payload = line.split("\t", 1)
        record = json.loads(payload)
        self.assertEqual(record["text"], "inspect the trace")
        self.assertEqual(
            record["raw_transcript"],
            "flux inspect length view trace",
        )
        self.assertEqual(
            record["corrected_transcript"],
            "Flux inspect Langfuse trace",
        )
        self.assertTrue(record["model_accepted"])

    def test_get_transcript_history_path_can_be_disabled(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(
                get_transcript_history_path({"transcript_history_path": "off"})
            )

    def test_sherpa_vad_missing_model_falls_back(self):
        with mock.patch.dict("os.environ", {"VOICE_AUTO_VAD_BACKEND": "sherpa"}):
            with tempfile.TemporaryDirectory() as tmp_dir:
                config = {
                    "auto_sherpa_vad_model": os.path.join(tmp_dir, "missing.onnx")
                }
                self.assertIsNone(build_sherpa_vad(config, 16000))

    def test_paste_transcript_uses_enter_key_for_submit_command(self):
        with mock.patch.object(app, "paste_text", return_value=True) as paste_text:
            with mock.patch.object(app, "get_paste_mode", return_value="auto"):
                with mock.patch.object(app, "get_paste_delay", return_value=0.0):
                    with mock.patch.object(
                        app, "press_enter", return_value=True
                    ) as press_enter:
                        result = app.paste_transcript_text("submit this send")

        self.assertEqual(result, (True, "submit this", True))
        paste_text.assert_called_once_with("submit this")
        press_enter.assert_called_once_with()

    def test_paste_transcript_records_history_after_success(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "history.txt")
            with mock.patch.object(app, "paste_text", return_value=True):
                result = app.paste_transcript_text("saved transcript", path)

            with open(path, "r", encoding="utf-8") as handle:
                line = handle.read().strip()

        self.assertEqual(result, (True, "saved transcript", False))
        self.assertEqual(line.split("\t", 1)[1], "saved transcript")

    def test_paste_transcript_can_press_enter_for_submit_only(self):
        with mock.patch.object(app, "paste_text") as paste_text:
            with mock.patch.object(app, "get_paste_mode", return_value="auto"):
                with mock.patch.object(app, "get_paste_delay", return_value=0.0):
                    with mock.patch.object(
                        app, "press_enter", return_value=True
                    ) as press_enter:
                        result = app.paste_transcript_text("send")

        self.assertEqual(result, (True, "", True))
        paste_text.assert_not_called()
        press_enter.assert_called_once_with()

    def test_paste_transcript_type_mode_uses_combined_submit(self):
        with mock.patch.dict("os.environ", {"VOICE_PASTE_MODE": "type"}):
            with mock.patch.object(app, "get_paste_delay", return_value=0.0):
                with mock.patch.object(
                    app, "type_text_and_submit", return_value=True
                ) as submit:
                    with mock.patch.object(app, "paste_text") as paste_text:
                        with mock.patch.object(app, "press_enter") as press_enter:
                            result = app.paste_transcript_text("submit this send")

        self.assertEqual(result, (True, "submit this", True))
        submit.assert_called_once_with("submit this")
        paste_text.assert_not_called()
        press_enter.assert_not_called()

    def test_paste_transcript_type_mode_can_submit_only(self):
        with mock.patch.dict("os.environ", {"VOICE_PASTE_MODE": "type"}):
            with mock.patch.object(app, "get_paste_delay", return_value=0.0):
                with mock.patch.object(
                    app, "type_text_and_submit", return_value=True
                ) as submit:
                    result = app.paste_transcript_text("send")

        self.assertEqual(result, (True, "", True))
        submit.assert_called_once_with("")

    def test_type_text_and_submit_delays_before_typed_newline(self):
        calls = []

        def record_type(text):
            calls.append(("type", text))
            return True

        def record_sleep(delay):
            calls.append(("sleep", delay))

        with mock.patch.dict("os.environ", {"VOICE_SUBMIT_ENTER_DELAY": "0.5"}):
            with mock.patch.object(app, "type_text", side_effect=record_type):
                with mock.patch.object(app.time, "sleep", side_effect=record_sleep):
                    result = app.type_text_and_submit("queued text")

        self.assertTrue(result)
        self.assertEqual(
            calls,
            [("type", "queued text"), ("sleep", 0.5), ("type", "\n")],
        )

    def test_allows_repeated_domain_terms_in_real_sentence(self):
        text = (
            "Now run the end to end compensation flow test to make sure "
            "that the changes work that we don t need the intake coach "
            "anymore and are using the intake moment and that the intake "
            "process is not giving advice but trying to understand the "
            "situation"
        )

        self.assertFalse(is_likely_bad_transcript(text))

    def test_rejects_repetitive_bigram_loop(self):
        self.assertTrue(is_likely_bad_transcript("thank you thank you thank you"))


if __name__ == "__main__":
    unittest.main()
