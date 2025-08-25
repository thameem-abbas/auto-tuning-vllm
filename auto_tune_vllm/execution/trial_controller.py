"""Trial controller implementations for Ray and local execution."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Optional

from ..benchmarks.providers import BenchmarkProvider, GuideLLMBenchmark
from ..core.trial import ExecutionInfo, TrialConfig, TrialResult

logger = logging.getLogger(__name__)


class TrialController(ABC):
    """Abstract base for trial execution controllers."""
    
    @abstractmethod
    def run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Execute a single optimization trial."""
        pass
    
    @abstractmethod
    def cleanup_resources(self):
        """Clean up any resources (servers, processes, etc.)."""
        pass


class BaseTrialController(TrialController):
    """Base implementation with common trial execution logic."""
    
    def __init__(self):
        self.vllm_process: Optional[subprocess.Popen] = None
        self.benchmark_provider: Optional[BenchmarkProvider] = None
        self._environment_validated = False
    
    def _validate_environment(self) -> None:
        """Validate that all required packages are available on this worker."""
        if self._environment_validated:
            return
        
        required_packages = {
            'vllm': 'vLLM serving framework',
            'guidellm': 'GuideLLM benchmarking tool', 
            'optuna': 'Optuna optimization framework',
            'ray': 'Ray distributed computing',
            'psycopg2': 'PostgreSQL client',
        }
        
        missing_packages = []
        
        for package, description in required_packages.items():
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(f"{package} ({description})")
        
        if missing_packages:
            missing_list = '\n  - '.join(missing_packages)
            raise RuntimeError(
                f"Missing required packages on Ray worker node:\n  - {missing_list}\n\n"
                f"Ray worker nodes must have the same Python environment as the head node.\n"
                f"Install auto-tune-vllm on all Ray cluster nodes:\n"
                f"  pip install auto-tune-vllm"
            )
        
        # Check if commands are available in PATH
        required_commands = {
            'python': 'Python interpreter',
            'guidellm': 'GuideLLM CLI tool',
        }
        
        import shutil
        missing_commands = []
        
        for command, description in required_commands.items():
            if not shutil.which(command):
                missing_commands.append(f"{command} ({description})")
        
        if missing_commands:
            missing_list = '\n  - '.join(missing_commands)
            raise RuntimeError(
                f"Missing required commands in PATH on Ray worker node:\n  - {missing_list}\n\n"
                f"Ensure all dependencies are properly installed and available in PATH."
            )
        
        # Check GPU availability if CUDA is expected
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"Ray worker has {gpu_count} CUDA GPU(s) available")
            else:
                logger.warning("No CUDA GPUs detected on Ray worker. vLLM may fail to start.")
        except Exception as e:
            logger.warning(f"Could not check GPU availability: {e}")
        
        self._environment_validated = True
        logger.info("Environment validation passed on Ray worker")
        
    def run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Execute trial with proper error handling and cleanup."""
        execution_info = ExecutionInfo()
        
        try:
            # Validate environment first
            self._validate_environment()
            
            # Setup benchmark provider
            self.benchmark_provider = self._create_benchmark_provider(trial_config)
            
            # Start vLLM server
            server_info = self._start_vllm_server(trial_config)
            execution_info.worker_node_id = self._get_worker_id()
            
            # Wait for server to be ready
            self._wait_for_server_ready(server_info["url"])
            
            # Run benchmark
            benchmark_result = self.benchmark_provider.run_benchmark(
                model_url=server_info["url"],
                config=trial_config.benchmark_config
            )
            
            # Extract objectives for Optuna
            objective_values = self._extract_objectives(benchmark_result)
            
            execution_info.mark_completed()
            
            return TrialResult(
                trial_number=trial_config.trial_number,
                objective_values=objective_values,
                detailed_metrics=benchmark_result,
                execution_info=execution_info,
                success=True
            )
            
        except Exception as e:
            execution_info.mark_completed()
            logger.error(f"Trial {trial_config.trial_number} failed: {e}")
            
            return TrialResult(
                trial_number=trial_config.trial_number,
                objective_values=[],
                detailed_metrics={},
                execution_info=execution_info,
                success=False,
                error_message=str(e)
            )
        finally:
            self.cleanup_resources()
    
    def _create_benchmark_provider(self, trial_config: TrialConfig) -> BenchmarkProvider:
        """Create appropriate benchmark provider."""
        benchmark_type = trial_config.benchmark_config.benchmark_type
        
        if benchmark_type == "guidellm":
            return GuideLLMBenchmark()
        else:
            # Import custom provider by name
            # This enables extensibility for custom benchmarks
            return self._import_custom_benchmark(benchmark_type)
    
    def _import_custom_benchmark(self, benchmark_type: str) -> BenchmarkProvider:
        """Dynamically import custom benchmark provider."""
        try:
            # Try to import from benchmarks.custom module
            module_name = f"auto_tune_vllm.benchmarks.custom.{benchmark_type}"
            module = __import__(module_name, fromlist=[benchmark_type])
            provider_class = getattr(module, f"{benchmark_type.title()}Benchmark")
            return provider_class()
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unknown benchmark provider: {benchmark_type}") from e
    
    def _start_vllm_server(self, trial_config: TrialConfig) -> dict:
        """Start vLLM server with trial parameters."""
        port = self._get_available_port()
        
        # Build vLLM command
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", trial_config.benchmark_config.model,
            "--port", str(port),
            "--host", "0.0.0.0"
        ]
        
        # Add trial-specific parameters
        cmd.extend(trial_config.vllm_args)
        
        logger.info(f"Starting vLLM server: {' '.join(cmd)}")
        
        # Start process
        self.vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return {
            "port": port,
            "url": f"http://localhost:{port}/v1",
            "pid": self.vllm_process.pid
        }
    
    def _get_available_port(self) -> int:
        """Get an available port for vLLM server."""
        import socket
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        
        return port
    
    def _wait_for_server_ready(self, url: str, timeout: int = 300):
        """Wait for vLLM server to be ready."""
        import requests
        
        start_time = time.time()
        health_url = url.replace("/v1", "/health")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"vLLM server ready at {url}")
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        raise RuntimeError(f"vLLM server failed to start within {timeout} seconds")
    
    def _extract_objectives(self, benchmark_result: dict) -> list[float]:
        """Extract objective values for Optuna from benchmark results."""
        # Default: single objective (maximize throughput)
        throughput = benchmark_result.get("output_tokens_per_second", 0.0)
        return [throughput]
    
    def cleanup_resources(self):
        """Clean up vLLM server process."""
        if self.vllm_process:
            try:
                self.vllm_process.terminate()
                self.vllm_process.wait(timeout=10)
                logger.info(f"Terminated vLLM process {self.vllm_process.pid}")
            except subprocess.TimeoutExpired:
                self.vllm_process.kill()
                logger.warning(f"Force killed vLLM process {self.vllm_process.pid}")
            finally:
                self.vllm_process = None
    
    @abstractmethod
    def _get_worker_id(self) -> str:
        """Get worker identifier (Ray node ID or local machine info)."""
        pass


class LocalTrialController(BaseTrialController):
    """Local execution trial controller."""
    
    def _get_worker_id(self) -> str:
        """Get local machine identifier."""
        import socket
        return f"local_{socket.gethostname()}"


class RayTrialController(BaseTrialController):
    """Ray-based distributed trial controller."""
    
    def _get_worker_id(self) -> str:
        """Get Ray worker node ID."""
        try:
            import ray
            return ray.get_runtime_context().node_id.hex()
        except Exception:
            return "ray_worker_unknown"
    
    def _start_vllm_server(self, trial_config: TrialConfig) -> dict:
        """Start vLLM server with GPU assignment from Ray."""
        # Ray handles GPU allocation via resources={"GPU": 1}
        # We can get the assigned GPU from CUDA_VISIBLE_DEVICES if needed
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        logger.info(f"Ray assigned GPUs: {gpu_ids}")
        
        # Call parent implementation
        return super()._start_vllm_server(trial_config)


# Ray remote wrapper for RayTrialController
try:
    import ray
    
    # Only create Ray remote class if Ray is available
    _RayTrialControllerBase = RayTrialController
    
    @ray.remote
    class RayTrialController(_RayTrialControllerBase):
        """Ray remote actor for distributed trial execution."""
        
        def run_trial(self, trial_config: TrialConfig) -> TrialResult:
            """Run trial on Ray worker."""
            return super().run_trial(trial_config)
        
        def __del__(self):
            """Ensure cleanup on actor destruction."""
            self.cleanup_resources()

except ImportError:
    # This should not happen since Ray is a required dependency
    raise ImportError("Ray is required but not installed. This should not happen with proper installation.")