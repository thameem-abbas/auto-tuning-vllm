"""Trial data structures and execution info."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import BenchmarkConfig to avoid circular imports
from ..benchmarks.config import BenchmarkConfig


@dataclass
class TrialConfig:
    """Configuration for a single optimization trial."""
    
    study_id: int
    trial_number: int
    parameters: Dict[str, Any]  # vLLM parameters from Optuna
    benchmark_config: BenchmarkConfig
    resource_requirements: Dict[str, float] = field(default_factory=lambda: {"num_gpus": 1, "num_cpus": 4})
    logging_config: Optional[Dict[str, Any]] = None  # Logging configuration from study
    
    @property
    def vllm_args(self) -> List[str]:
        """Convert parameters to vLLM command-line arguments."""
        args = []
        for param_name, value in self.parameters.items():
            # Convert underscore to dash for CLI
            cli_param = param_name.replace("_", "-")
            
            if isinstance(value, bool):
                if value:
                    args.append(f"--{cli_param}")
            else:
                args.extend([f"--{cli_param}", str(value)])
        
        return args


@dataclass
class ExecutionInfo:
    """Information about trial execution."""
    
    worker_node_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    
    def mark_completed(self):
        """Mark execution as completed."""
        self.end_time = time.time()
        self.duration_seconds = self.end_time - self.start_time


@dataclass
class TrialResult:
    """Results from a completed trial."""
    
    trial_number: int
    objective_values: List[float]  # For Optuna (throughput, latency, etc.)
    detailed_metrics: Dict[str, Any]  # Rich percentile data from benchmarks
    execution_info: ExecutionInfo
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def primary_objective(self) -> float:
        """Get primary objective value for single-objective optimization."""
        return self.objective_values[0] if self.objective_values else 0.0


