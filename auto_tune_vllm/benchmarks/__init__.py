"""Benchmark providers and interfaces."""

from .config import BenchmarkConfig
from .guidellm import GuideLLMBenchmark
from .providers import BenchmarkProvider, CustomBenchmarkTemplate
from .vllm_bench_serve import VLLMBenchServeBenchmark

# Registry for dynamic benchmark provider loading
BENCHMARK_PROVIDERS = {
    "guidellm": GuideLLMBenchmark,
    "vllm_bench_online": VLLMBenchServeBenchmark,
    "custom_template": CustomBenchmarkTemplate,
}


def get_benchmark_provider(provider_name: str) -> BenchmarkProvider:
    """Get benchmark provider by name."""
    if provider_name not in BENCHMARK_PROVIDERS:
        raise ValueError(
            f"Unknown benchmark provider: {provider_name}. "
            f"Available providers: {list(BENCHMARK_PROVIDERS.keys())}"
        )

    provider_class = BENCHMARK_PROVIDERS[provider_name]
    return provider_class()


__all__ = [
    "BenchmarkProvider",
    "GuideLLMBenchmark", 
    "VLLMBenchServeBenchmark",
    "BenchmarkConfig",
    "CustomBenchmarkTemplate",
    "BENCHMARK_PROVIDERS",
    "get_benchmark_provider",
]