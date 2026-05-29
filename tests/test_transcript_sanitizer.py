import os
import tempfile
import unittest
from unittest import mock

from app import (
    append_transcript_history,
    build_auto_tmux_switch_commands,
    build_sherpa_vad,
    extract_text_after_trigger_word,
    format_colored_detection_words,
    get_transcript_history_path,
    is_likely_bad_transcript,
    match_auto_shell_command,
    match_auto_shell_command_prefix,
    normalize_voice_command_text,
    parse_word_list,
    sanitize_transcript_text,
    split_trailing_submit_command,
)
import app


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

    def test_extract_text_after_trigger_word_keeps_words_after_trigger(self):
        self.assertEqual(
            extract_text_after_trigger_word("ignore this agent create issue", "agent"),
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
        self.assertEqual(extract_text_after_trigger_word("submit agent.", "agent"), "")

    def test_extract_text_after_trigger_word_accepts_aliases(self):
        self.assertEqual(
            extract_text_after_trigger_word("assistant create issue", "agent", ["assistant"]),
            "create issue",
        )

    def test_parse_word_list_accepts_comma_separated_aliases(self):
        self.assertEqual(parse_word_list("assistant, helper, assistant"), ["assistant", "helper"])

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
