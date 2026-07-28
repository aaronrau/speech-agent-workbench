import asyncio
import socket
import unittest
from unittest import mock

from aiohttp import ClientSession, WSMsgType, WSServerHandshakeError

import app


class VoiceApiWebSocketTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server = None
        app.set_active_voice_api_server(None)
        app.TMUX_SUMMARY_SNAPSHOTS.clear()
        app.TMUX_WEBHOOK_LAST_DETAIL_LINES.clear()

    def tearDown(self):
        if self.server is not None:
            self.server.server_close()
        app.set_active_voice_api_server(None)
        app.TMUX_SUMMARY_SNAPSHOTS.clear()
        app.TMUX_WEBHOOK_LAST_DETAIL_LINES.clear()

    def free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    def start_server(self, token="local-secret"):
        port = self.free_port()
        config = {
            "api_enabled": True,
            "api_host": "127.0.0.1",
            "api_port": port,
            "api_token": token,
            "api_websocket_enabled": True,
            "api_websocket_heartbeat_seconds": 0,
            "api_websocket_replay_events": 10,
        }
        commands = {
            "flux": {
                "label": "Flux",
                "tmux_send_target": "%1",
                "argv": ["tmux", "select-pane", "-t", "%1"],
            },
            "pike": {
                "label": "Pike",
                "tmux_send_target": "%2",
                "argv": ["tmux", "select-pane", "-t", "%2"],
            },
        }
        self.server = app.start_voice_api_server(config, commands)
        self.assertIsNotNone(self.server)
        return config, commands, port

    async def test_http_post_and_websocket_share_the_legacy_api_port(self):
        _config, _commands, port = self.start_server()
        headers = {"Authorization": "Bearer local-secret"}
        url = f"http://127.0.0.1:{port}"

        with mock.patch.object(
            app,
            "run_auto_shell_command",
            return_value=True,
        ), mock.patch.object(
            app,
            "send_text_to_tmux_target",
            return_value=True,
        ):
            async with ClientSession() as session:
                response = await session.post(
                    f"{url}/messages",
                    headers=headers,
                    json={"agent": "Flux", "message": "legacy request"},
                )
                self.assertEqual(response.status, 200)
                legacy_result = await response.json()
                self.assertTrue(legacy_result["ok"])
                self.assertTrue(legacy_result["sent"])

                websocket = await session.ws_connect(
                    f"{url}/ws",
                    headers=headers,
                )
                ready = await websocket.receive_json(timeout=2.0)
                self.assertEqual(ready["type"], "connection.ready")
                self.assertEqual(ready["version"], 1)
                self.assertEqual(ready["agents"], ["Flux", "Pike"])

                await websocket.send_json(
                    {
                        "type": "message.send",
                        "request_id": "request-1",
                        "agent": "Flux",
                        "message": "run tests",
                    }
                )
                accepted = await websocket.receive_json(timeout=2.0)
                self.assertEqual(accepted["type"], "message.accepted")
                self.assertEqual(accepted["request_id"], "request-1")
                self.assertTrue(accepted["result"]["sent"])

                response = await session.post(
                    f"{url}/messages",
                    headers=headers,
                    json={"agent": "Flux", "message": "legacy follow-up"},
                )
                self.assertEqual(response.status, 200)
                superseded = await websocket.receive_json(timeout=2.0)
                self.assertEqual(superseded["type"], "message.completed")
                self.assertEqual(superseded["request_id"], "request-1")
                self.assertEqual(
                    superseded["payload"]["completion_status"],
                    "superseded",
                )
                await websocket.close()

    async def test_progress_completion_busy_and_replay_are_correlated(self):
        config, _commands, port = self.start_server()
        headers = {"Authorization": "Bearer local-secret"}
        url = f"http://127.0.0.1:{port}"

        with mock.patch.object(
            app,
            "run_auto_shell_command",
            return_value=True,
        ), mock.patch.object(
            app,
            "send_text_to_tmux_target",
            return_value=True,
        ):
            async with ClientSession() as session:
                websocket = await session.ws_connect(
                    f"{url}/ws",
                    headers=headers,
                )
                await websocket.receive_json(timeout=2.0)
                await websocket.send_json(
                    {
                        "type": "message.send",
                        "request_id": "request-1",
                        "agent": "Flux",
                        "message": "run tests",
                    }
                )
                accepted = await websocket.receive_json(timeout=2.0)
                self.assertEqual(accepted["type"], "message.accepted")

                await websocket.send_json(
                    {
                        "type": "message.send",
                        "request_id": "request-1",
                        "agent": "Flux",
                        "message": "run tests",
                    }
                )
                duplicate = await websocket.receive_json(timeout=2.0)
                self.assertEqual(duplicate["type"], "message.accepted")
                self.assertTrue(duplicate["duplicate"])
                self.assertFalse(duplicate["pending"])

                await websocket.send_json(
                    {
                        "type": "message.send",
                        "request_id": "request-2",
                        "agent": "Flux",
                        "message": "start another task",
                    }
                )
                busy = await websocket.receive_json(timeout=2.0)
                self.assertEqual(busy["type"], "message.error")
                self.assertEqual(busy["error"], "agent_busy")
                self.assertEqual(busy["active_request_id"], "request-1")

                app.dispatch_tmux_summary_webhook(
                    config,
                    "Flux",
                    "run tests",
                    "Flux started the tests.",
                    detail_lines=["pytest started"],
                )
                progress = await websocket.receive_json(timeout=2.0)
                self.assertEqual(progress["type"], "message.progress")
                self.assertEqual(progress["request_id"], "request-1")
                self.assertEqual(
                    progress["payload"]["detail_lines"],
                    ["pytest started"],
                )
                progress_event_id = progress["event_id"]
                await websocket.close()

                app.dispatch_agent_completion_summary_webhook(
                    config,
                    {
                        "agent": "Flux",
                        "status": "done",
                        "message": "tests passed",
                    },
                )

                resumed_websocket = await session.ws_connect(
                    f"{url}/ws",
                    headers=headers,
                )
                ready = await resumed_websocket.receive_json(timeout=2.0)
                self.assertEqual(ready["type"], "connection.ready")
                await resumed_websocket.send_json(
                    {
                        "type": "connection.resume",
                        "resume_after_event_id": progress_event_id,
                    }
                )
                completion = await resumed_websocket.receive_json(timeout=2.0)
                self.assertEqual(completion["type"], "message.completed")
                self.assertEqual(completion["request_id"], "request-1")
                self.assertTrue(completion["payload"]["is_final"])
                self.assertEqual(
                    completion["payload"]["completion_status"],
                    "done",
                )
                resumed = await resumed_websocket.receive_json(timeout=2.0)
                self.assertEqual(resumed["type"], "connection.resumed")
                self.assertEqual(resumed["replayed_event_count"], 1)
                await resumed_websocket.close()

    async def test_local_summary_uses_correlated_websocket_response(self):
        config, _commands, port = self.start_server()
        headers = {"Authorization": "Bearer local-secret"}
        url = f"http://127.0.0.1:{port}"
        result = {
            "ok": True,
            "agent": "Pike",
            "detail": "tests passed",
            "detail_lines": ["tests passed"],
            "detail_line_count": 1,
            "message": "progress_summary",
            "sent": False,
            "source": "tmux_capture",
            "summary": "Pike finished the tests.",
            "type": "local",
        }

        with mock.patch.object(
            app,
            "route_api_local_summary",
            return_value=result,
        ) as route:
            async with ClientSession() as session:
                websocket = await session.ws_connect(
                    f"{url}/ws",
                    headers=headers,
                )
                await websocket.receive_json(timeout=2.0)
                await websocket.send_json(
                    {
                        "type": "summary.request",
                        "request_id": "summary-1",
                        "agent": "Pike",
                    }
                )
                response = await websocket.receive_json(timeout=2.0)
                self.assertEqual(response["type"], "summary.result")
                self.assertEqual(response["request_id"], "summary-1")
                self.assertEqual(
                    response["result"]["summary"],
                    "Pike finished the tests.",
                )
                route.assert_called_once_with(
                    config,
                    "Pike",
                    "progress_summary",
                    self.server.commands,
                )
                await websocket.close()

    async def test_websocket_requires_the_existing_api_token(self):
        _config, _commands, port = self.start_server()
        async with ClientSession() as session:
            with self.assertRaises(WSServerHandshakeError) as raised:
                await session.ws_connect(f"http://127.0.0.1:{port}/ws")
        self.assertEqual(raised.exception.status, 401)

    async def test_invalid_websocket_payload_does_not_close_connection(self):
        _config, _commands, port = self.start_server(token="")
        async with ClientSession() as session:
            websocket = await session.ws_connect(
                f"http://127.0.0.1:{port}/ws"
            )
            await websocket.receive_json(timeout=2.0)
            await websocket.send_str("{")
            error = await websocket.receive_json(timeout=2.0)
            self.assertEqual(error["type"], "message.error")
            self.assertEqual(error["error"], "invalid_json")
            self.assertFalse(websocket.closed)
            await websocket.close()

    async def test_terminate_control_emits_event_before_route_finishes(self):
        _config, commands, port = self.start_server(token="")
        commands.update(
            {
                "wolf": {
                    "label": "Wolf",
                    "tmux_send_target": "%3",
                    "argv": ["tmux", "select-pane", "-t", "%3"],
                },
                "wolf terminate session": {
                    "label": "Wolf terminate session",
                    "argv": ["tmux", "kill-session", "-t", "workbench"],
                    "exit_after": True,
                    "allow_prefix": False,
                    "requires_explicit_audio": True,
                },
            }
        )

        command_result = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(
            app,
            "focus_auto_terminal_window",
            return_value=True,
        ), mock.patch.object(
            app,
            "run_command",
            return_value=command_result,
        ) as run_command:
            async with ClientSession() as session:
                websocket = await session.ws_connect(
                    f"http://127.0.0.1:{port}/ws"
                )
                await websocket.receive_json(timeout=2.0)
                await websocket.send_json(
                    {
                        "type": "message.send",
                        "request_id": "terminate-1",
                        "agent": "Wolf",
                        "message": "terminate session",
                    }
                )
                terminating = await websocket.receive_json(timeout=2.0)
                self.assertEqual(terminating["type"], "session.terminating")
                self.assertEqual(terminating["request_id"], "terminate-1")
                closed = await websocket.receive(timeout=2.0)
                self.assertIn(
                    closed.type,
                    (WSMsgType.CLOSE, WSMsgType.CLOSED),
                )
        run_command.assert_called_once_with(
            ["tmux", "kill-session", "-t", "workbench"],
            timeout=2.0,
        )
        for _attempt in range(20):
            if not self.server._thread.is_alive():
                break
            await asyncio.sleep(0.01)
        self.assertFalse(self.server._thread.is_alive())


class VoiceApiWebSocketConfigTests(unittest.TestCase):
    def tearDown(self):
        app.set_active_voice_api_server(None)

    def test_build_websocket_url_uses_loopback_for_wildcard_bind(self):
        self.assertEqual(
            app.build_voice_api_websocket_url("0.0.0.0", 8787, "/ws"),
            "ws://127.0.0.1:8787/ws",
        )

    def test_websocket_path_is_normalized(self):
        self.assertEqual(
            app.get_voice_api_websocket_path({"api_websocket_path": "events/"}),
            "/events",
        )

    def test_summary_dispatch_keeps_webhook_and_websocket_delivery(self):
        websocket_server = mock.Mock(websocket_enabled=True)
        app.set_active_voice_api_server(websocket_server)
        config = {
            "tmux_summary_webhook_url": "http://127.0.0.1:9999/summary"
        }

        with mock.patch.object(
            app,
            "post_tmux_summary_webhook",
            return_value=True,
        ) as post_webhook:
            thread = app.dispatch_tmux_summary_webhook(
                config,
                "Flux",
                "run tests",
                "Flux is running tests.",
                detail_lines=["pytest started"],
            )
            thread.join(timeout=2.0)

        websocket_server.publish_summary.assert_called_once()
        post_webhook.assert_called_once()

    def test_summary_dispatch_uses_websocket_without_configured_webhook(self):
        websocket_server = mock.Mock(websocket_enabled=True)
        app.set_active_voice_api_server(websocket_server)

        with mock.patch.object(
            app,
            "post_tmux_summary_webhook",
        ) as post_webhook:
            thread = app.dispatch_tmux_summary_webhook(
                {},
                "Flux",
                "run tests",
                "Flux is running tests.",
                detail_lines=["pytest started"],
            )

        self.assertIsNone(thread)
        websocket_server.publish_summary.assert_called_once()
        post_webhook.assert_not_called()


if __name__ == "__main__":
    unittest.main()
