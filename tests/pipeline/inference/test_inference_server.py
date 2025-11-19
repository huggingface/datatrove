"""
Tests for the InferenceServer base class implementation.

These tests verify the core server lifecycle, health monitoring, auto-restart,
and resource cleanup functionality without requiring actual inference backends.
"""

import asyncio
import unittest

from datatrove.pipeline.inference.run_inference import InferenceConfig
from datatrove.pipeline.inference.servers.dummy_server import DummyServer


class TestInferenceServerLifecycle(unittest.TestCase):
    """Test basic server lifecycle operations."""

    def test_simple_startup_and_cleanup(self):
        """
        Scenario 1: Simple scenario - check that resources are correctly deallocated.

        Verifies:
        - Server starts successfully
        - Server becomes ready
        - Server can handle requests
        - All resources are properly cleaned up on exit
        """

        async def run_test():
            config = InferenceConfig(
                server_type="dummy",
                model_name_or_path="dummy",
            )
            server = DummyServer(config, rank=0)

            async with server as (srv, is_master):
                await server._server_ready

                print("Server is ready! Verifying port...")
                # Verify port was assigned
                self.assertIsNotNone(server._port)
                self.assertGreaterEqual(server._port, 3000)
                self.assertLessEqual(server._port, 65535)

                print("Sending test request...")
                # Send a test request
                status, body = await server.send_request(
                    "/v1/chat/completions", {"messages": [{"role": "user", "content": "test"}]}
                )
                print(f"Request completed with status: {status}")
                self.assertEqual(status, 200)
                self.assertIn(b"dummy text content", body)
                print("Test assertions passed, exiting context manager...")

            # Verify server process is cleaned up
            if server._server_process:
                self.assertIsNotNone(server._server_process.returncode)

            # Verify monitoring task is cancelled
            if server._server_monitoring_task:
                self.assertTrue(server._server_monitoring_task.cancelled() or server._server_monitoring_task.done())

            # Verify auto-restart task is cancelled
            self.assertTrue(server._auto_restart_task.cancelled() or server._auto_restart_task.done())

        asyncio.run(run_test())

    def test_server_auto_restart(self):
        """
        Test manual server restart by triggering cleanup and restart.
        """

        async def run_test():
            config = InferenceConfig(
                server_type="dummy",
                model_name_or_path="dummy",
            )
            server = DummyServer(config, rank=0)

            async with server:
                await server._server_ready

                # kill the server process
                server.kill_server()

                # Verify requests work
                status, body = await server.send_request(
                    "/v1/chat/completions", {"messages": [{"role": "user", "content": "test"}]}
                )
                self.assertEqual(status, 200)

        asyncio.run(run_test())

    def test_server_auto_restart_turn_off(self):
        async def run_test():
            config = InferenceConfig(
                server_type="dummy",
                model_name_or_path="dummy",
                auto_restart_server=False,
            )
            server = DummyServer(config, rank=0)

            async with server:
                await server._server_ready

                # kill the server process
                server.kill_server()

                # Ensure we have enough time for server to notice
                await asyncio.sleep(1)

                # Verify requests work
                with self.assertRaises(asyncio.CancelledError):
                    status, body = await server.send_request(
                        "/v1/chat/completions", {"messages": [{"role": "user", "content": "test"}]}
                    )

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
