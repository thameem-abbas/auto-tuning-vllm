"""Benchmark provider implementations."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict

from .config import BenchmarkConfig

logger = logging.getLogger(__name__)


class BenchmarkProvider(ABC):
    """Abstract benchmark provider interface."""

    def __init__(self):
        self._logger = logger  # Default to module logger
        self._trial_context = None  # Store trial context for file paths
        self._process = None  # Track running benchmark process for termination
        self._process_pid = None  # Store PID for cleanup even if process handle is gone
        self._process_pgid = None  # Store process group ID for cleanup
        self._cancellation_flag = None  # Function to check for cancellation
    
    def set_logger(self, custom_logger):
        """Set a custom logger for this benchmark provider."""
        self._logger = custom_logger

    def set_trial_context(self, study_name: str, trial_id: str):
        """Set trial context for benchmark result storage."""
        self._trial_context = {
            'study_name': study_name,
            'trial_id': trial_id
        }
    
    def terminate_benchmark(self):
        """Terminate the running benchmark process and its process group if active."""
        # Try to use stored PID/PGID first, in case process handle is gone
        pid = self._process_pid
        if pid is None:  # first check, if PID is still none after this we will return
            pid = self._process.pid if self._process else None
        pgid = self._process_pgid
        
        if pid is None:
            self._logger.debug("Benchmark: No benchmark process to terminate")
            return
            
        self._logger.info(
            f"Benchmark: Terminating benchmark process {pid} "
            f"and its process group..."
        )
        
        # Try to get process group ID if we don't have it
        if pgid is None:
            try:
                pgid = os.getpgid(pid)
                self._logger.debug(f"Benchmark: Retrieved process group ID: {pgid}")
            except (OSError, ProcessLookupError):
                self._logger.debug(
                    f"Benchmark: Process {pid} already gone or no process group"
                )
        
        # Try graceful shutdown with SIGTERM first
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
                self._logger.info(
                    f"Benchmark → Process Group: Sent SIGTERM to group {pgid}"
                )
            else:
                os.kill(pid, signal.SIGTERM)
                self._logger.info(f"Benchmark → Process: Sent SIGTERM to process {pid}")
        except (OSError, ProcessLookupError):
            # Process already gone
            self._logger.info(f"Benchmark: Process {pid} already terminated")
            self._process = None
            self._process_pid = None
            self._process_pgid = None
            return
        
        # Wait for graceful shutdown
        # (use a shorter timeout if process handle unavailable)
        wait_timeout = 5 if self._process else 2
        try:
            if self._process:
                self._process.wait(timeout=wait_timeout)
            else:
                # Wait by polling if no process handle
                import time
                for _ in range(int(wait_timeout * 10)):
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        time.sleep(0.1)
                    except (OSError, ProcessLookupError):
                        # Process is gone
                        break
                else:
                    # Timeout - process still exists
                    raise subprocess.TimeoutExpired(None, wait_timeout)
                    
            self._logger.info(
                f"Benchmark: ✓ Process {pid} terminated gracefully via SIGTERM"
            )
        except subprocess.TimeoutExpired:
            self._logger.warning(
                f"Benchmark: Process {pid} did not terminate within {wait_timeout}s. "
                f"Escalating to SIGKILL..."
            )
            
            # Force kill with SIGKILL
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                    self._logger.info(
                        f"Benchmark → Process Group: Sent SIGKILL to group {pgid}"
                    )
                else:
                    os.kill(pid, signal.SIGKILL)
                    self._logger.info(
                        f"Benchmark → Process: Sent SIGKILL to process {pid}"
                    )
                self._logger.info(
                    f"Benchmark: ✓ Process {pid} force killed via SIGKILL"
                )
            except (OSError, ProcessLookupError) as e:
                self._logger.debug(
                    f"Benchmark: Process {pid} already gone during SIGKILL: {e}"
                )
        finally:
            self._process = None
            self._process_pid = None
            self._process_pgid = None

    @abstractmethod
    def start_benchmark(
        self, model_url: str, config: BenchmarkConfig
    ) -> subprocess.Popen:
        """
        Start benchmark subprocess (non-blocking).

        Args:
            model_url: URL of the vLLM server (e.g., "http://localhost:8000/v1")
            config: Benchmark configuration

        Returns:
            Popen process handle for polling by caller
        """
        pass

    @abstractmethod
    def parse_results(self) -> Dict[str, Any]:
        """
        Parse benchmark results from output file.

        Returns:
            Dictionary with benchmark results. Must include metrics that can be
            converted to objective values for Optuna.
        """
        pass

