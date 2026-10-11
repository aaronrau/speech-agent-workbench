import json
import os
import tempfile
import unittest
from unittest import mock

import app


class TmuxSummaryTailerTests(unittest.TestCase):
    def run_tailer(
        self, times, output=None, capture=None, summarize=None, commands=None,
        updates=None, config_overrides=None, command_text=None,
    ):
        clock = {"index": 0}
        commands = commands or {}
        updates = updates or {}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "console.log")

            def append_output(records):
                with open(path, "a", encoding="utf-8") as handle:
                    for label, text in records:
                        handle.write(json.dumps({"data": f"[tmux][{label}] {text}"}) + "\n")

            append_output(output or [])
            config = {
                "auto_tmux_console_log": path,
                "auto_tmux_console_replay": True,
                "auto_tmux_console_retention_seconds": 0,
                "auto_tmux_console_max_bytes": 0,
                "auto_tmux_summary_idle_seconds": 5,
                "auto_tmux_summary_stable_seconds": 60,
            }
            config.update(config_overrides or {})
            stop_event = mock.Mock()
            stop_event.is_set.side_effect = lambda: clock["index"] >= len(times)

            def advance(_delay):
                clock["index"] += 1
                if clock["index"] < len(times):
                    append_output(updates.get(times[clock["index"]], []))

            stop_event.wait.side_effect = advance
            sent = []
            attempts = []

            def summarize_output(config, label, command_text, lines):
                attempts.append((times[clock["index"]], label, list(lines)))
                if summarize:
                    return summarize(config, label, command_text, lines)
                return "Tests passed."

            def dispatch(config, label, command_text, summary, **kwargs):
                sent.append((times[clock["index"]], label, command_text, summary,
                             kwargs["detail_lines"]))

            def capture_output(config, command):
                if capture:
                    return capture(times[clock["index"]], command)
                return None, "tmux_capture_unavailable"

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
                app.threading, "Event", return_value=stop_event
            ), mock.patch.object(app.threading, "Thread") as thread, mock.patch.object(
                app.time, "monotonic", side_effect=lambda: times[clock["index"]]
            ), mock.patch.object(
                app, "build_auto_tmux_switch_commands", return_value=commands
            ), mock.patch.object(
                app, "capture_tmux_target_lines", side_effect=capture_output
            ), mock.patch.object(
                app, "get_tmux_sent_command", side_effect=lambda *_args: {
                    "text": command_text(times[clock["index"]])
                    if command_text else "run tests"
                }
            ), mock.patch.object(
                app, "summarize_tmux_agent_output", side_effect=summarize_output
            ), mock.patch.object(
                app, "dispatch_tmux_summary_webhook", side_effect=dispatch
            ), mock.patch("builtins.print"):
                thread.return_value.start.side_effect = (
                    lambda: thread.call_args.kwargs["target"]()
                )
                app.start_auto_tmux_console_log_tailer(config)
        return attempts, sent

    def test_failed_summary_retries_without_new_output(self):
        calls = 0

        def summarize(*_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("model temporarily unavailable")
            return "Tests passed."

        attempts, sent = self.run_tailer(
            [0, 5, 19, 20, 60, 120],
            output=[("Flux", "Tests passed.\n")],
            summarize=summarize,
        )
        self.assertEqual([attempt[0] for attempt in attempts], [5, 20])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][4], ["Tests passed."])

    def test_one_agent_failure_does_not_suppress_other_agent(self):
        failed = False

        def summarize(_config, label, _command, _lines):
            nonlocal failed
            if label == "Flux" and not failed:
                failed = True
                raise RuntimeError("temporary error")
            return f"{label} finished."

        _attempts, sent = self.run_tailer(
            [0, 5, 20],
            output=[("Flux", "Flux done\n"), ("Brock", "Brock done\n")],
            summarize=summarize,
        )
        self.assertEqual([(event[0], event[1]) for event in sent],
                         [(5, "Brock"), (20, "Flux")])

    def test_stable_pane_sends_once_after_one_minute_despite_redraws(self):
        times = list(range(0, 126, 5))
        attempts, sent = self.run_tailer(
            times,
            output=[("Flux", "\x1b[2K\r")],
            updates={now: [("Flux", "\x1b[2K\r")] for now in times[1:]},
            commands={"flux": {"label": "Flux", "tmux_send_target": "%1"}},
            capture=lambda _now, _command: (["Tests passed."], None),
        )
        self.assertEqual([attempt[0] for attempt in attempts], [60])
        self.assertEqual(sent[0][4], ["Tests passed."])

    def test_stable_pane_recovers_missing_pipe_output(self):
        attempts, sent = self.run_tailer(
            [0, 59, 60, 120],
            commands={"flux": {"label": "Flux", "tmux_send_target": "%1"}},
            capture=lambda _now, _command: (["Tests passed."], None),
        )
        self.assertEqual([attempt[0] for attempt in attempts], [60])
        self.assertEqual(len(sent), 1)

    def test_stable_fallback_does_not_repeat_an_idle_summary(self):
        attempts, sent = self.run_tailer(
            [0, 5, 60, 65, 120],
            output=[("Flux", "Tests passed.\n")],
            commands={"flux": {"label": "Flux", "tmux_send_target": "%1"}},
            capture=lambda _now, _command: (["Tests passed."], None),
        )
        self.assertEqual([attempt[0] for attempt in attempts], [5])
        self.assertEqual(len(sent), 1)

    def test_changed_pane_restarts_one_minute_timer(self):
        attempts, sent = self.run_tailer(
            [0, 55, 60, 114, 115, 180],
            commands={"flux": {"label": "Flux", "tmux_send_target": "%1"}},
            capture=lambda now, _command: (
                ["Running tests." if now < 55 else "Tests passed."], None
            ),
        )
        self.assertEqual([attempt[0] for attempt in attempts], [115])
        self.assertEqual(sent[0][4], ["Tests passed."])

    def test_empty_panes_and_voice_pane_are_not_summarized(self):
        attempts, sent = self.run_tailer(
            [0, 60, 120],
            commands={
                "flux": {"label": "Flux", "tmux_send_target": "%1"},
                "wolf": {"label": "Wolf", "tmux_send_target": "%2"},
            },
            capture=lambda _now, command: (
                ["Listener log"] if command["label"] == "Wolf" else [], None
            ),
        )
        self.assertEqual(attempts, [])
        self.assertEqual(sent, [])

    def test_empty_summary_is_retried_instead_of_marked_sent(self):
        replies = iter(["", "Tests passed."])
        attempts, sent = self.run_tailer(
            [0, 5, 20],
            output=[("Flux", "Tests passed.\n")],
            summarize=lambda *_args: next(replies),
        )
        self.assertEqual([attempt[0] for attempt in attempts], [5, 20])
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][3], "Tests passed.")

    def test_new_content_replaces_failed_summary_input(self):
        replies = iter([RuntimeError("temporarily unavailable"), "Tests passed."])

        def summarize(*_args):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        attempts, sent = self.run_tailer(
            [0, 5, 20, 79, 80, 140],
            output=[("Flux", "Running tests.\n")],
            commands={"flux": {"label": "Flux", "tmux_send_target": "%1"}},
            capture=lambda now, _command: (
                ["Running tests." if now < 20 else "Tests passed."], None
            ),
            summarize=summarize,
        )
        self.assertEqual(attempts, [
            (5, "Flux", ["Running tests."]), (80, "Flux", ["Tests passed."])
        ])
        self.assertEqual(len(sent), 1)

    def test_new_command_with_same_content_restarts_stability_timer(self):
        attempts, sent = self.run_tailer(
            [0, 60, 65, 120, 125, 185],
            commands={"flux": {"label": "Flux", "tmux_send_target": "%1"}},
            capture=lambda _now, _command: (["Tests passed."], None),
            command_text=lambda now: "run tests" if now < 65 else "rerun tests",
        )
        self.assertEqual([attempt[0] for attempt in attempts], [60, 125])
        self.assertEqual([event[2] for event in sent], ["run tests", "rerun tests"])

    def test_failed_capture_interrupts_stability_timer(self):
        attempts, sent = self.run_tailer(
            [0, 55, 60, 119, 120],
            commands={"flux": {"label": "Flux", "tmux_send_target": "%1"}},
            capture=lambda now, _command: (
                (None, "tmux_capture_timeout") if now == 55
                else (["Tests passed."], None)
            ),
        )
        self.assertEqual([attempt[0] for attempt in attempts], [120])
        self.assertEqual(len(sent), 1)

    def test_stable_fallback_can_be_disabled(self):
        attempts, sent = self.run_tailer(
            [0, 60, 120],
            commands={"flux": {"label": "Flux", "tmux_send_target": "%1"}},
            capture=lambda *_args: self.fail("fallback should not capture"),
            config_overrides={"auto_tmux_summary_stable_seconds": 0},
        )
        self.assertEqual(attempts, [])
        self.assertEqual(sent, [])

    def test_disabled_summaries_do_not_run_fallback(self):
        attempts, sent = self.run_tailer(
            [0, 60, 120],
            commands={"flux": {"label": "Flux", "tmux_send_target": "%1"}},
            capture=lambda *_args: self.fail("summaries disabled"),
            config_overrides={"auto_tmux_summary_enabled": False},
        )
        self.assertEqual(attempts, [])
        self.assertEqual(sent, [])


if __name__ == "__main__":
    unittest.main()
