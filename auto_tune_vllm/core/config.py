"""Configuration management with validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import re
import yaml
from pydantic import BaseModel, Field

# Import BenchmarkConfig from the benchmarks module
from ..benchmarks.config import BenchmarkConfig


class ParameterConfig(BaseModel, ABC):
    """Base class for parameter configurations."""
    
    name: str
    enabled: bool = True
    description: Optional[str] = None
    
    @abstractmethod
    def generate_optuna_suggest(self, trial) -> Any:
        """Generate Optuna trial suggestion for this parameter."""
        pass


class RangeParameter(ParameterConfig):
    """Range-based parameter (continuous or discrete)."""
    
    min_value: Union[int, float] = Field(alias="min")
    max_value: Union[int, float] = Field(alias="max") 
    step: Optional[Union[int, float]] = None
    data_type: str = "float"  # "int" or "float"
    
    def generate_optuna_suggest(self, trial) -> Union[int, float]:
        """Generate Optuna range suggestion."""
        if self.data_type == "int":
            return trial.suggest_int(
                self.name, 
                int(self.min_value), 
                int(self.max_value),
                step=int(self.step) if self.step else None
            )
        else:
            return trial.suggest_float(
                self.name,
                float(self.min_value),
                float(self.max_value), 
                step=float(self.step) if self.step else None
            )


class ListParameter(ParameterConfig):
    """List-based parameter (categorical choices)."""
    
    options: List[Any]
    data_type: str = "str"
    
    def generate_optuna_suggest(self, trial) -> Any:
        """Generate Optuna categorical suggestion."""
        return trial.suggest_categorical(self.name, self.options)


class BooleanParameter(ParameterConfig):
    """Boolean parameter."""
    
    data_type: str = "bool"
    
    def generate_optuna_suggest(self, trial) -> bool:
        """Generate Optuna boolean suggestion."""
        return trial.suggest_categorical(self.name, [True, False])


class GridParameter(ParameterConfig):
    """Hybrid parameter that supports both explicit options and ranges for grid search.
    
    This parameter type allows maximum flexibility for grid search:
    - If 'options' is provided, uses those exact values
    - If range parameters (min/max/step) are provided, generates discrete range values
    - Prioritizes user-specified options over schema-defined ranges
    """
    
    # List-based attributes
    options: Optional[List[Any]] = None
    
    # Range-based attributes  
    min_value: Optional[Union[int, float]] = Field(None, alias="min")
    max_value: Optional[Union[int, float]] = Field(None, alias="max")
    step: Optional[Union[int, float]] = None
    
    # Data type for validation and conversion
    data_type: str = "auto"  # "int", "float", "str", "bool", or "auto"
    
    def __post_init__(self):
        """Validate GridParameter configuration."""
        super().__post_init__()
        
        # Must have either options or range parameters
        has_options = self.options is not None and len(self.options) > 0
        has_range = (self.min_value is not None and self.max_value is not None)
        
        if not has_options and not has_range:
            raise ValueError(f"GridParameter '{self.name}' must specify either 'options' or range parameters (min/max)")
        
        # If both are provided, prioritize options but validate range for consistency
        if has_options and has_range:
            # Validate that options fall within the specified range
            for option in self.options:
                if not (self.min_value <= option <= self.max_value):
                    raise ValueError(f"Option {option} for parameter '{self.name}' is outside range [{self.min_value}, {self.max_value}]")
    
    def get_grid_values(self) -> List[Any]:
        """Get the actual values to use for grid search.
        
        Returns:
            List of values to test in grid search
        """
        if self.options is not None and len(self.options) > 0:
            # Use user-specified options
            return self._convert_values(self.options)
        else:
            # Generate values from range
            return self._generate_range_values()
    
    def _convert_values(self, values: List[Any]) -> List[Any]:
        """Convert values to the appropriate data type."""
        if self.data_type == "auto":
            # Auto-detect from first value
            if values and isinstance(values[0], (int, float)):
                return [self._convert_numeric(v) for v in values]
            else:
                return values
        elif self.data_type == "int":
            return [int(v) for v in values]
        elif self.data_type == "float":
            return [float(v) for v in values]
        elif self.data_type == "str":
            return [str(v) for v in values]
        elif self.data_type == "bool":
            return [bool(v) for v in values]
        else:
            return values
    
    def _convert_numeric(self, value: Any) -> Union[int, float]:
        """Convert a value to int or float based on context."""
        if isinstance(value, float) or (isinstance(value, str) and '.' in value):
            return float(value)
        else:
            return int(value)
    
    def _generate_range_values(self) -> List[Union[int, float]]:
        """Generate discrete values from range parameters."""
        if self.min_value is None or self.max_value is None:
            raise ValueError(f"Cannot generate range values for '{self.name}': min_value or max_value is None")
        
        values = []
        step = self.step or 1
        
        if step <= 0:
            raise ValueError(f"Parameter '{self.name}' has invalid step size: {step}")
        
        # Use integer-based generation for floats to avoid accumulation drift
        if isinstance(self.min_value, float) or isinstance(step, float):
            # Calculate number of steps and generate via multiplication
            n_steps = int((self.max_value - self.min_value) / step) + 1
            # Cap to reasonable grid size
            n_steps = min(n_steps, 10000)
            for i in range(n_steps):
                val = self.min_value + (i * step)
                if val > self.max_value:
                    break
                # Round to avoid floating precision artifacts
                values.append(round(val, 10))
        else:
            # Integer range - simple iteration
            current = self.min_value
            while current <= self.max_value:
                values.append(current)
                current += step
                # Safety break for large ranges
                if len(values) > 10000:
                    break
        
        return values
    
    def generate_optuna_suggest(self, trial) -> Any:
        """Generate Optuna suggestion for non-grid samplers."""
        # For non-grid samplers, use the first available method
        if self.options is not None and len(self.options) > 0:
            return trial.suggest_categorical(self.name, self.options)
        else:
            # Use range suggestion
            if self.data_type == "int" or (self.data_type == "auto" and isinstance(self.min_value, int)):
                return trial.suggest_int(
                    self.name, 
                    int(self.min_value), 
                    int(self.max_value),
                    step=int(self.step) if self.step else None
                )
            else:
                return trial.suggest_float(
                    self.name,
                    float(self.min_value),
                    float(self.max_value), 
                    step=float(self.step) if self.step else None
                )


@dataclass
class ObjectiveConfig:
    """Configuration for a single optimization objective."""
    
    metric: str  # "output_tokens_per_second", "request_latency", etc.
    direction: str  # "maximize" or "minimize"
    percentile: str = "median"  # "median", "p50", "p95", "p90", "p99", "mean"
    
    def __post_init__(self):
        """Validate objective configuration."""
        valid_metrics = {
            "output_tokens_per_second", "request_latency", "time_to_first_token_ms",
            "inter_token_latency_ms", "requests_per_second"
        }
        valid_directions = {"maximize", "minimize"}
        valid_percentiles = {"median", "p50", "p95", "p90", "p99", "mean"}
        
        if self.metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{self.metric}'. Valid options: {valid_metrics}")
        if self.direction not in valid_directions:
            raise ValueError(f"Invalid direction '{self.direction}'. Valid options: {valid_directions}")
        if self.percentile not in valid_percentiles:
            raise ValueError(f"Invalid percentile '{self.percentile}'. Valid options: {valid_percentiles}")


@dataclass 
class OptimizationConfig:
    """Optimization configuration with support for new structured format and backward compatibility."""
    
    # Backward compatibility fields
    objective: Union[str, List[str]] = None  # Old format: "maximize", "minimize", or list
    sampler: str = "tpe"  # "tpe", "random", "botorch", "nsga2", "grid" 
    n_trials: int = 100
    n_startup_trials: Optional[int] = None  # Number of startup trials for samplers that support it
    
    # New structured format fields
    approach: Optional[str] = None  # "single_objective" or "multi_objective"
    objectives: Optional[List[ObjectiveConfig]] = None  # For multi-objective
    preset: Optional[str] = None  # "high_throughput", "low_latency", "balanced"
    
    def __post_init__(self):
        """Process and validate optimization configuration."""
        # Handle preset configurations
        if self.preset:
            self._apply_preset()
            return
        
        # Handle new structured format
        if self.approach:
            self._validate_structured_format()
            return
        
        # Handle backward compatibility (old format)
        if self.objective:
            self._convert_old_format()
            return
        
        # Default fallback
        self._apply_default_config()
    
    def _apply_preset(self):
        """Apply preset optimization configurations."""
        if self.preset == "high_throughput":
            self.approach = "single_objective"
            self.objectives = [ObjectiveConfig(
                metric="output_tokens_per_second",
                direction="maximize",
                percentile="mean"
            )]
        elif self.preset == "low_latency":
            self.approach = "single_objective"
            self.objectives = [ObjectiveConfig(
                metric="request_latency",
                direction="minimize",
                percentile="p95"
            )]
        elif self.preset == "balanced":
            self.approach = "multi_objective"
            self.objectives = [
                ObjectiveConfig(
                    metric="output_tokens_per_second",
                    direction="maximize",
                    percentile="mean"
                ),
                ObjectiveConfig(
                    metric="request_latency",
                    direction="minimize",
                    percentile="median"
                )
            ]
        else:
            raise ValueError(f"Unknown preset '{self.preset}'. Valid options: high_throughput, low_latency, balanced")
    
    def _validate_structured_format(self):
        """Validate new structured format."""
        if self.approach not in ["single_objective", "multi_objective"]:
            raise ValueError(f"Invalid approach '{self.approach}'. Valid options: single_objective, multi_objective")
        
        if not self.objectives:
            raise ValueError("Objectives must be specified for structured format")
        
        if self.approach == "single_objective" and len(self.objectives) != 1:
            raise ValueError("Single objective optimization requires exactly one objective")
        
        if self.approach == "multi_objective" and len(self.objectives) < 2:
            raise ValueError("Multi-objective optimization requires at least two objectives")
    
    def _convert_old_format(self):
        """Convert old format to new structured format for backward compatibility."""
        if isinstance(self.objective, str):
            # Single objective
            self.approach = "single_objective"
            if self.objective == "maximize":
                # Default to maximizing throughput
                self.objectives = [ObjectiveConfig(
                    metric="output_tokens_per_second",
                    direction="maximize",
                    percentile="median"
                )]
            elif self.objective == "minimize":
                # Default to minimizing latency
                self.objectives = [ObjectiveConfig(
                    metric="request_latency",
                    direction="minimize",
                    percentile="median"
                )]
            else:
                raise ValueError(f"Invalid objective '{self.objective}'. Use 'maximize' or 'minimize'")
        elif isinstance(self.objective, list):
            # Multi-objective (legacy format)
            self.approach = "multi_objective"
            # Default to throughput vs latency
            self.objectives = [
                ObjectiveConfig(
                    metric="output_tokens_per_second",
                    direction="maximize",
                    percentile="median"
                ),
                ObjectiveConfig(
                    metric="request_latency",
                    direction="minimize",
                    percentile="median"
                )
            ]
    
    def _apply_default_config(self):
        """Apply default configuration when none is specified."""
        self.approach = "single_objective"
        self.objectives = [ObjectiveConfig(
            metric="output_tokens_per_second",
            direction="maximize",
            percentile="mean"
        )]
    
    @property
    def is_multi_objective(self) -> bool:
        """Check if this is multi-objective optimization."""
        return self.approach == "multi_objective"
    
    @property
    def optuna_directions(self) -> List[str]:
        """Get Optuna directions for study creation."""
        return [obj.direction for obj in self.objectives]
    
    def get_metric_key(self, objective_index: int = 0) -> str:
        """Get the metric key for extracting values from benchmark results."""
        if objective_index >= len(self.objectives):
            raise IndexError(f"Objective index {objective_index} out of range")
        
        obj = self.objectives[objective_index]
        if obj.percentile == "median":
            return obj.metric
        else:
            return f"{obj.metric}_{obj.percentile}"


@dataclass
class BaselineConfig:
    """Configuration for baseline trials."""
    
    enabled: bool = False
    run_first: bool = True  # Run baseline before optimization trials
    concurrency_levels: List[int] = field(default_factory=lambda: [50])  # Concurrency levels to test
    
    def __post_init__(self):
        """Validate baseline configuration."""
        if self.enabled and not self.concurrency_levels:
            raise ValueError("Baseline configuration requires at least one concurrency level")
        
        if self.enabled:
            for concurrency in self.concurrency_levels:
                if not isinstance(concurrency, int) or concurrency <= 0:
                    raise ValueError(f"Invalid concurrency level: {concurrency}. Must be positive integer")


@dataclass
class StudyConfig:
    """Complete study configuration."""
    
    study_name: str
    database_url: Optional[str]
    optimization: OptimizationConfig
    benchmark: BenchmarkConfig
    parameters: Dict[str, ParameterConfig] = field(default_factory=dict)
    baseline: Optional[BaselineConfig] = None  # NEW: Baseline configuration
    logging_config: Optional[Dict[str, Any]] = None
    storage_file: Optional[str] = None  # Alternative to database_url for file-based storage
    study_prefix: Optional[str] = None  # For auto-generated study names with custom prefix
    use_explicit_name: bool = False  # Flag to indicate explicit name usage (affects load_if_exists behavior)
    
    def __post_init__(self):
        """Validate study configuration after initialization."""
        self._validate_grid_sampler_configuration()
    
    def _validate_grid_sampler_configuration(self):
        """Validate that n_trials matches grid combinations when using grid sampler."""
        if self.optimization.sampler == "grid":
            grid_size, param_details = self._calculate_grid_size_with_details()
            
            # Create detailed breakdown message
            details_msg = "\nGrid search parameter breakdown:\n"
            for param_name, count in param_details.items():
                details_msg += f"  - {param_name}: {count} values\n"
            details_msg += f"Total combinations: {' × '.join(str(count) for count in param_details.values())} = {grid_size:,}"
            
            if self.optimization.n_trials != grid_size:
                # Add helpful guidance for large grid sizes
                guidance = ""
                if grid_size > 1000:
                    guidance = ("\n\nWARNING: This grid search has a very large number of combinations! "
                              "Consider:\n"
                              "- Reducing parameter ranges or increasing step sizes\n"
                              "- Using fewer parameters in your grid search\n"
                              "- Switching to 'tpe', 'random', or 'botorch' sampler for large spaces\n"
                              "- Using 'nsga2' sampler for multi-objective optimization")
                
                raise ValueError(
                    f"When using grid sampler, n_trials ({self.optimization.n_trials}) "
                    f"must equal the total number of parameter combinations ({grid_size:,}). "
                    f"Either adjust n_trials to {grid_size} or use a different sampler."
                    f"{details_msg}{guidance}"
                )
            else:
                # Log the grid information even when validation passes
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Grid sampler validation passed: {grid_size:,} combinations{details_msg}")
    
    def _calculate_grid_size(self) -> int:
        """Calculate total grid search combinations from parameter configuration."""
        grid_size, _ = self._calculate_grid_size_with_details()
        return grid_size
    
    def _calculate_grid_size_with_details(self) -> Tuple[int, Dict[str, int]]:
        """Calculate total grid search combinations with detailed parameter breakdown.
        
        This replicates the logic from StudyController._calculate_grid_size()
        but works with ParameterConfig objects instead of raw search space.
        
        Returns:
            tuple: (total_grid_size, dict of parameter_name -> value_count)
        """
        if not self.parameters:
            return 1, {}
        
        size = 1
        param_details = {}
        
        for param_name, param_config in self.parameters.items():
            if not param_config.enabled:
                continue
                
            # Count possible values for this parameter
            param_values_count = 0
            
            if hasattr(param_config, 'get_grid_values'):
                # GridParameter: use its specialized method to get actual values
                grid_values = param_config.get_grid_values()
                param_values_count = len(grid_values)
            elif hasattr(param_config, 'options') and param_config.options is not None:
                # ListParameter: discrete options
                param_values_count = len(param_config.options)
            elif hasattr(param_config, 'min_value') and param_config.min_value is not None:
                # RangeParameter: calculate discrete steps
                min_val = param_config.min_value
                max_val = param_config.max_value
                step = getattr(param_config, 'step', 1) or 1
                
                if step <= 0:
                    raise ValueError(f"Parameter {param_name} has invalid step size: {step}")
                
                # Calculate number of steps (inclusive of both endpoints)
                # This matches the logic in StudyController._create_sampler
                if isinstance(min_val, float) or isinstance(step, float):
                    n_steps = int((max_val - min_val) / step) + 1
                    # Cap to reasonable grid size (matching StudyController logic)
                    param_values_count = min(n_steps, 10000)
                else:
                    # Integer range
                    param_values_count = int((max_val - min_val) / step) + 1
            else:
                # BooleanParameter: assume True/False
                param_values_count = 2
            
            if param_values_count <= 0:
                raise ValueError(f"Parameter {param_name} has no valid values for grid search")
            
            param_details[param_name] = param_values_count
            size *= param_values_count
        
        return size, param_details
    
    @classmethod
    def from_file(cls, config_path: str, schema_path: Optional[str] = None, 
                 defaults_path: Optional[str] = None, vllm_version: Optional[str] = None) -> StudyConfig:
        """Load and validate configuration from YAML file."""
        config_validator = ConfigValidator(schema_path, defaults_path, vllm_version)
        return config_validator.load_and_validate(config_path)


class ConfigValidator:
    """Validates study configurations against parameter schema."""
    
    def __init__(self, schema_path: Optional[str] = None, defaults_path: Optional[str] = None, 
                 vllm_version: Optional[str] = None):
        """Initialize with parameter schema and optional defaults."""
        if schema_path is None:
            # Try to use version-specific schema if vllm_version is provided
            if vllm_version is not None:
                schema_path = self._get_versioned_schema_path(vllm_version)
            else:
                # Use default schema shipped with package
                schema_path = Path(__file__).parent.parent / "schemas" / "parameter_schema_original.yaml"
        
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        
        # Load defaults - support versioned defaults
        self.defaults = {}
        self.vllm_version = vllm_version
        
        if defaults_path is not None:
            self.defaults_path = Path(defaults_path)
            self.defaults = self._load_defaults()
        elif vllm_version is not None:
            # Load version-specific defaults
            self._load_version_defaults(vllm_version)
        else:
            # Try to load latest defaults
            self._load_latest_defaults()
    
    def _get_versioned_schema_path(self, vllm_version: str) -> Path:
        """
        Get the schema path for a specific vLLM version.
        
        Args:
            vllm_version: vLLM version string (e.g., "0.10.1.1")
            
        Returns:
            Path to versioned schema file
            
        Raises:
            FileNotFoundError: If versioned schema doesn't exist
        """
        # Convert version to filename format (e.g., "0.10.1.1" -> "v0_10_1_1.yaml")
        safe_version = vllm_version.replace('.', '_')
        schemas_dir = Path(__file__).parent.parent / "schemas"
        versioned_schema_path = schemas_dir / f"v{safe_version}.yaml"
        
        if versioned_schema_path.exists():
            print(f"Using version-specific schema for vLLM {vllm_version}: {versioned_schema_path}")
            return versioned_schema_path
        else:
            # Fallback to default schema with warning
            fallback_schema = schemas_dir / "parameter_schema_original.yaml"
            print(f"Warning: No schema found for vLLM version {vllm_version}, falling back to default schema: {fallback_schema}")
            return fallback_schema
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load parameter schema from YAML."""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        
        with open(self.schema_path) as f:
            return yaml.safe_load(f)
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load defaults from YAML file."""
        if not self.defaults_path.exists():
            raise FileNotFoundError(f"Defaults file not found: {self.defaults_path}")
        
        with open(self.defaults_path) as f:
            defaults_data = yaml.safe_load(f)
        
        return self._flatten_defaults(defaults_data)
    
    def _load_version_defaults(self, version: str):
        """Load version-specific defaults."""
        try:
            from ..utils.version_manager import VLLMVersionManager
            manager = VLLMVersionManager()
            defaults_data = manager.load_defaults(version)
            self.defaults = self._flatten_defaults(defaults_data)
            self.defaults_path = manager.get_defaults_path(version)
        except Exception as e:
            print(f"Warning: Could not load vLLM defaults for version {version}: {e}")
            self.defaults = {}
            self.defaults_path = None
    
    def _load_latest_defaults(self):
        """Try to load the latest available defaults."""
        try:
            from ..utils.version_manager import VLLMVersionManager
            manager = VLLMVersionManager()
            defaults_data = manager.load_defaults()  # Load latest
            self.defaults = self._flatten_defaults(defaults_data)
            self.defaults_path = manager.get_defaults_path()
        except Exception:
            # No versioned defaults available, use empty defaults
            self.defaults = {}
            self.defaults_path = None
    
    def _flatten_defaults(self, defaults_data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten the nested defaults structure for easy lookup."""
        flat_defaults = {}
        for section, section_defaults in defaults_data.get("defaults", {}).items():
            for param_name, default_value in section_defaults.items():
                flat_defaults[param_name] = default_value
        return flat_defaults
    
    def load_and_validate(self, config_path: str) -> StudyConfig:
        """Load and validate study configuration with environment variable expansion."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Read yaml contents
        with open(config_path) as f:
            raw_config = f.read()
        
        expanded_config = self.expand_environment_variables(raw_config)
        raw_config = yaml.safe_load(expanded_config)

        return self._validate_config(raw_config)

    def expand_environment_variables(self, yaml_content: str) -> str:
        """
        Expand environment variables in YAML content.

        Supports patterns:
        - ${VAR_NAME} - expands to environment variable value or empty string if not set
        - ${VAR_NAME:-default_value} - expands to environment variable value or default_value if not set

        Args:
            yaml_content: Raw YAML content as string

        Returns:
            YAML content with environment variables expanded

        Examples:
            ${POSTGRES_PASSWORD} -> value of POSTGRES_PASSWORD env var
            ${LOG_LEVEL:-INFO} -> value of LOG_LEVEL env var or "INFO" if not set
        """
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""

            env_value = os.getenv(var_name)
            if env_value is None:
                if default_value:
                    return default_value
                else:
                    # Log warning for missing required env vars without defaults
                    print(f"Warning: Environment variable '{var_name}' not found, using empty string")
                    return ""
            return env_value

        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*?)(?::-(.*?))?\}'
        expanded_content = re.sub(pattern, replace_env_var, yaml_content)

        return expanded_content
    
    def _generate_unique_study_name(self, prefix: Optional[str] = None) -> str:
        """Generate a unique study name in the format {prefix}_{N} or study_N."""
        # Use timestamp seconds as a simple unique number
        import time
        timestamp_seconds = int(time.time())
        # Take last 6 digits to keep numbers reasonable
        study_number = timestamp_seconds % 1000000
        
        if prefix:
            return f"{prefix}_{study_number}"
        else:
            return f"study_{study_number}"
    
    def _handle_study_naming(self, study_info: Dict[str, Any]) -> tuple[str, Optional[str], bool]:
        """
        Handle study naming logic with prefix support.
        
        Returns:
            tuple: (study_name, study_prefix, use_explicit_name)
        
        Rules:
        1. If just 'name' is provided: Use exact name, fail if exists (use_explicit_name=True)
        2. If just 'prefix' is provided: Auto-generate with prefix (use_explicit_name=False)
        3. If both provided: Validation error
        4. If neither provided: Auto-generate with default prefix (use_explicit_name=False)
        """
        name = study_info.get("name")
        prefix = study_info.get("prefix")
        
        # Scenario 3: Both name and prefix provided - ERROR
        if name and prefix:
            raise ValueError(
                "Cannot specify both 'name' and 'prefix' in study configuration. "
                "Use 'name' for exact study names that must be unique, or 'prefix' for auto-generated names."
            )
        
        # Scenario 1: Just name provided - use exact name, fail if exists
        if name and not prefix:
            print(f"Using explicit study name: {name} (will fail if study already exists)")
            return name, None, True
        
        # Scenario 2: Just prefix provided - auto-generate with prefix
        if prefix and not name:
            auto_name = self._generate_unique_study_name(prefix)
            print(f"Generated study name: {auto_name} from prefix: {prefix}")
            return auto_name, prefix, False
        
        # Scenario 4: Neither provided - auto-generate with default prefix
        auto_name = self._generate_unique_study_name()
        print(f"Auto-generated study name: {auto_name}")
        return auto_name, None, False
    
    def _validate_config(self, raw_config: Dict[str, Any]) -> StudyConfig:
        """Validate configuration against schema."""
        # Validate and build parameter configs
        validated_params = {}
        
        for param_name, param_config in raw_config.get("parameters", {}).items():
            if not param_config.get("enabled", True):
                continue
            
            # Get schema definition
            schema_def = self.schema.get("parameters", {}).get(param_name)
            if not schema_def:
                raise ValueError(f"Unknown parameter in schema: {param_name}")
            
            # Build parameter config based on type
            validated_param = self._build_parameter_config(param_name, param_config, schema_def)
            validated_params[param_name] = validated_param
        
        # Build other configs
        study_info = raw_config.get("study", {})
        if study_info is None:
            study_info = {}
        
        # Handle study naming logic with prefix support
        study_name, study_prefix, use_explicit_name = self._handle_study_naming(study_info)
        
        # Handle optimization config with validation
        opt_config_data = raw_config["optimization"]
        
        # Convert objective config if using new structured format
        if "objective" in opt_config_data and isinstance(opt_config_data["objective"], dict):
            # Single objective structured format
            obj_data = opt_config_data["objective"]
            opt_config_data["objectives"] = [ObjectiveConfig(**obj_data)]
            del opt_config_data["objective"]
        elif "objectives" in opt_config_data:
            # Multi-objective structured format
            objectives_data = opt_config_data["objectives"]
            opt_config_data["objectives"] = [ObjectiveConfig(**obj) for obj in objectives_data]
        
        optimization = OptimizationConfig(**opt_config_data)
        benchmark = BenchmarkConfig(**raw_config["benchmark"])
        
        # Handle optional database_url and storage_file
        database_url = study_info.get("database_url")
        storage_file = study_info.get("storage_file")
        
        # Validate storage configuration
        if not database_url and not storage_file:
            # Default to file-based storage using study name
            storage_file = f"./optuna_studies/{study_name}/study.db"
        
        if database_url and storage_file:
            raise ValueError("Cannot specify both database_url and storage_file. Choose one storage method.")
        
        # Handle baseline configuration
        baseline_config = None
        if "baseline" in raw_config:
            baseline_data = raw_config["baseline"]
            if baseline_data.get("enabled", False):
                baseline_config = BaselineConfig(**baseline_data)
        
        return StudyConfig(
            study_name=study_name,
            database_url=database_url,
            optimization=optimization,
            benchmark=benchmark, 
            parameters=validated_params,
            baseline=baseline_config,
            logging_config=raw_config.get("logging"),
            storage_file=storage_file,
            study_prefix=study_prefix,
            use_explicit_name=use_explicit_name
        )
    
    def _build_parameter_config(
        self, 
        name: str, 
        user_config: Dict[str, Any], 
        schema_def: Dict[str, Any]
    ) -> ParameterConfig:
        """Build parameter config from user config, defaults, and schema."""
        param_type = schema_def["type"]
        description = schema_def.get("description")
        
        base_config = {
            "name": name,
            "enabled": user_config.get("enabled", True),
            "description": description
        }
        
        def get_value(key: str, schema_fallback=None, allow_defaults: bool = False):
            """
            Priority: user_config > (defaults if allow_defaults) > schema > schema_fallback.
            Note: defaults are parameter values, not bounds; do NOT use for min/max/step/options.
            """
            if key in user_config:
                return user_config[key]
            if allow_defaults and name in self.defaults:
                return self.defaults[name]
            if key in schema_def:
                return schema_def[key]
            return schema_fallback
        
        # 🆕 GRID SEARCH ENHANCEMENT: Detect when to use GridParameter
        # Use GridParameter if:
        # 1. User provides explicit 'options' for any parameter type, OR
        # 2. User provides both range info AND options (hybrid parameter), OR  
        # 3. Parameter has complex grid search requirements
        user_has_options = "options" in user_config and user_config["options"] is not None
        user_has_range = any(key in user_config for key in ["min", "max", "min_value", "max_value", "range"])
        
        if user_has_options or (user_has_range and param_type in ["range", "list"]):
            # Use GridParameter for maximum flexibility
            grid_config = {**base_config}
            
            # Add options if provided by user
            if user_has_options:
                grid_config["options"] = user_config["options"]
                
                # Validate user options are subset of schema options (for list types)
                if param_type == "list" and "options" in schema_def:
                    schema_options = schema_def["options"]
                    invalid_options = set(user_config["options"]) - set(schema_options)
                    if invalid_options:
                        raise ValueError(f"Invalid options for {name}: {invalid_options}. Must be subset of {schema_options}")
            
            # Add range parameters (from user config or schema)
            if param_type == "range" or user_has_range:
                # Handle different range formats
                if "range" in user_config:
                    # Handle nested range format: range: {start: X, end: Y, step: Z}
                    range_config = user_config["range"]
                    grid_config["min"] = range_config.get("start")
                    grid_config["max"] = range_config.get("end") 
                    grid_config["step"] = range_config.get("step")
                else:
                    # Handle direct format: min: X, max: Y, step: Z
                    grid_config["min"] = get_value("min", schema_def.get("min"), allow_defaults=False)
                    grid_config["max"] = get_value("max", schema_def.get("max"), allow_defaults=False)
                    grid_config["step"] = get_value("step", schema_def.get("step"), allow_defaults=False)
            
            # Set data type
            grid_config["data_type"] = schema_def.get("data_type", "auto")
            
            return GridParameter(**grid_config)
        
        # 🔄 LEGACY BEHAVIOR: Use original parameter types when no grid-specific config
        elif param_type == "range":
            return RangeParameter(
                **base_config,
                min=get_value("min", schema_def["min"], allow_defaults=False),
                max=get_value("max", schema_def["max"], allow_defaults=False),
                step=get_value("step", schema_def.get("step"), allow_defaults=False),
                data_type=schema_def["data_type"]
            )
        elif param_type == "list":
            # Allow user to restrict schema options
            schema_options = schema_def["options"]
            user_options = user_config.get("options", schema_options)
            
            # Validate user options are subset of schema options
            invalid_options = set(user_options) - set(schema_options)
            if invalid_options:
                raise ValueError(f"Invalid options for {name}: {invalid_options}")
            
            return ListParameter(
                **base_config,
                options=user_options,
                data_type=schema_def["data_type"]
            )
        elif param_type == "boolean":
            return BooleanParameter(**base_config)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")