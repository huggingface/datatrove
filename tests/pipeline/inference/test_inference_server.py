"""
Tests for the InferenceServer base class implementation.

These tests verify the core server lifecycle, health monitoring, auto-restart,
and resource cleanup functionality without requiring actual inference backends.
"""
import asyncio
import multiprocessing as mp
import os
import unittest
from dataclasses import dataclass

from datatrove.pipeline.inference.servers.dummy_server import DummyServer


@dataclass
class MockInferenceConfig:
    """Mock configuration for testing."""
    server_log_folder: str = None
    coordination_port: int = 8000
    master_port: int = 29500
    tp: int = 1
    dp: int = 1
    pp: int = 1


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
            config = MockInferenceConfig()
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
                    "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "test"}]}
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
                self.assertTrue(
                    server._server_monitoring_task.cancelled() or 
                    server._server_monitoring_task.done()
                )
            
            # Verify auto-restart task is cancelled
            self.assertTrue(
                server._auto_restart_task.cancelled() or 
                server._auto_restart_task.done()
            )
        
        asyncio.run(run_test())


class TestServerAutoRestart(unittest.TestCase):
    """Test server auto-restart and failure recovery."""

    def test_server_auto_restart(self):
        """
        Test manual server restart by triggering cleanup and restart.
        """
        async def run_test():
            config = MockInferenceConfig()
            server = DummyServer(config, rank=0)
            
            async with server:
                await server._server_ready

                # kill the server process
                server._server_process.kill()
                
                
                # Verify requests work
                status, body = await server.send_request(
                    "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "test"}]}
                )
                self.assertEqual(status, 200)
        
        asyncio.run(run_test())

    def test_server_auto_restart_distributed(self):
        """
        Test server auto-restart in distributed environment.
        """
        config = MockInferenceConfig()
        config.tp = 2  # Use 2 processes for distributed setup
            
        def run_master_server():
            # Mock the distributed functions
            from datatrove.pipeline.inference.distributed.utils import get_node_rank, is_master_node
            get_node_rank = lambda: 0  # Mock the rank to be 0
            is_master_node = lambda: True  # Mock the master node to be True
            get_master_node_host = lambda: "localhost"  # Mock the master node host to be localhost

            async def master_task():
                import os
                master_server = DummyServer(config, rank=0)  # Master node
                async with master_server:
                    await master_server._server_ready
                    
                    # Send initial request from master node
                    status, body = await master_server.send_request(
                        "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "test before kill"}]}
                    )
                    assert status == 200
                    
                    # Kill the master server process
                    master_server._server_process.kill()
                    
                    # Try to send request after killing - should trigger auto-restart
                    status, body = await master_server.send_request(
                        "/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "test after kill"}]}
                    )
                    assert status == 200
            
            asyncio.run(master_task())
        
        def run_worker_server():
            # Mock the distributed functions
            from datatrove.pipeline.inference.distributed.utils import get_node_rank, is_master_node, get_master_node_host
            get_node_rank = lambda: 1  # Mock the rank to be 1
            is_master_node = lambda: False  # Mock the master node to be False
            get_master_node_host = lambda: "localhost"  # Mock the master node host to be localhost

            async def worker_task():
                worker_server = DummyServer(config, rank=1)  # Worker node
                async with worker_server:
                    # Keep worker alive during the test
                    await asyncio.sleep(10)
            
            asyncio.run(worker_task())
        
        # Start both servers in separate processes
        master_process = mp.Process(target=run_master_server)
        worker_process = mp.Process(target=run_worker_server)
        
        master_process.start()
        worker_process.start()
        
        # Wait for master to complete (it runs the actual test)
        master_process.join()
        
        # Clean up worker process
        worker_process.terminate()
        worker_process.join()

        # Check that master process completed successfully
        self.assertEqual(master_process.exitcode, 0)
            


# class TestErrorHandling(unittest.TestCase):
#     """Test various error conditions."""

#     def test_request_before_ready(self):
#         """
#         Test that requests wait for server to be ready.
#         """
#         async def run_test():
#             config = MockInferenceConfig()
#             server = DummyServer(config, rank=0)
            
#             async with server:
#                 # Start a request immediately (server not ready yet)
#                 request_task = asyncio.create_task(
#                     server.send_request(
#                         "/v1/chat/completions",
#                         {"messages": [{"role": "user", "content": "test"}]}
#                     )
#                 )
                
#                 # Wait for server to become ready
#                 for i in range(40):
#                     if await server.is_ready():
#                         break
#                     await asyncio.sleep(0.5)
                
#                 # Request should complete successfully
#                 status, body = await request_task
#                 self.assertEqual(status, 200)
        
#         asyncio.run(run_test())

#     def test_cleanup_with_no_started_server(self):
#         """
#         Test cleanup when server was never started.
#         """
#         async def run_test():
#             config = MockInferenceConfig()
#             server = DummyServer(config, rank=0)
            
#             # Call cleanup without starting - should not raise
#             await server.server_cleanup()
            
#             # Server should still be in initial state
#             self.assertIsNone(server._server_process)
        
#         asyncio.run(run_test())

#     def test_double_cleanup(self):
#         """
#         Test that double cleanup doesn't cause issues.
#         """
#         async def run_test():
#             config = MockInferenceConfig()
#             server = DummyServer(config, rank=0)
            
#             async with server:
#                 # Wait for server to be ready
#                 for i in range(20):
#                     if await server.is_ready():
#                         break
#                     await asyncio.sleep(0.5)
            
#             # Cleanup already happened in __aexit__
#             # Call it again - should not raise
#             await server.server_cleanup()
        
#         asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
