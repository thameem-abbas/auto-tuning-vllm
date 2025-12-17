"""Benchmark configuration."""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.
    
    This class contains only constant (non-tunable) benchmark parameters.
    Tunable parameters are merged at trial creation time via merge_tunables().
    """
    
    benchmark_type: str = "guidellm"  # "guidellm" or custom provider name
    model: str = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
    max_seconds: int = 300
    dataset: Optional[str] = None  # HF dataset or file path
    prompt_tokens: int = 1000  # For synthetic data (can be overridden by tunables)
    output_tokens: int = 1000  # For synthetic data (can be overridden by tunables)
    concurrency: int = 50  # Benchmark concurrency level (legacy, use rates instead)
    
    # Advanced GuideLLM parameters
    processor: Optional[str] = None  # Processor model, defaults to model if not set
    rate: int = 50  # Single rate value for concurrent requests (can be overridden by tunables)
    samples: int = 1000  # Number of samples to take (can be overridden by tunables)
    
    # Token statistics for synthetic data - only used when explicitly specified
    prompt_tokens_stdev: Optional[int] = None
    prompt_tokens_min: Optional[int] = None  
    prompt_tokens_max: Optional[int] = None
    output_tokens_stdev: Optional[int] = None
    output_tokens_min: Optional[int] = None
    output_tokens_max: Optional[int] = None

    # Set in benchmark section of study config
    # Logging level for GuideLLM
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    
    def merge_tunables(self, tunable_values: Dict[str, Any]) -> "BenchmarkConfig":
        """Merge tunable parameter values into this config, returning a new instance.
        
        Args:
            tunable_values: Dictionary of tunable parameter names to their values
            
        Returns:
            New BenchmarkConfig instance with tunable values merged
        """
        # Create a dict of all current field values
        current_values = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
        
        # Update with tunable values
        current_values.update(tunable_values)
        
        # Create new instance with merged values
        return BenchmarkConfig(**current_values)
    
    @property
    def use_synthetic_data(self) -> bool:
        """Whether to use synthetic data instead of a dataset."""
        return self.dataset is None