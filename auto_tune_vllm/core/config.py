"""Configuration management with validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


@dataclass 
class OptimizationConfig:
    """Optimization configuration."""
    
    objective: Union[str, List[str]] = "maximize"  # "maximize", "minimize", or list for multi-objective
    sampler: str = "tpe"  # "tpe", "random", "botorch", "nsga2", "grid" 
    n_trials: int = 100


@dataclass
class StudyConfig:
    """Complete study configuration."""
    
    study_name: str
    database_url: Optional[str]
    optimization: OptimizationConfig
    benchmark: BenchmarkConfig
    parameters: Dict[str, ParameterConfig] = field(default_factory=dict)
    logging_config: Optional[Dict[str, Any]] = None
    storage_file: Optional[str] = None  # Alternative to database_url for file-based storage
    
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
            # Use default schema shipped with package
            schema_path = Path(__file__).parent.parent / "schemas" / "parameter_schema.yaml"
        
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
        """Load and validate study configuration."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
        
        return self._validate_config(raw_config)
    
    def _validate_config(self, raw_config: Dict[str, Any]) -> StudyConfig:
        """Validate configuration against schema."""
        # Validate and build parameter configs
        validated_params = {}
        
        for param_name, param_config in raw_config.get("parameters", {}).items():
            if not param_config.get("enabled", False):
                continue
            
            # Get schema definition
            schema_def = self.schema.get("parameters", {}).get(param_name)
            if not schema_def:
                raise ValueError(f"Unknown parameter in schema: {param_name}")
            
            # Build parameter config based on type
            validated_param = self._build_parameter_config(param_name, param_config, schema_def)
            validated_params[param_name] = validated_param
        
        # Build other configs
        study_info = raw_config["study"]
        optimization = OptimizationConfig(**raw_config["optimization"])
        benchmark = BenchmarkConfig(**raw_config["benchmark"])
        
        # Handle optional database_url and storage_file
        database_url = study_info.get("database_url")
        storage_file = study_info.get("storage_file")
        
        # Validate storage configuration
        if not database_url and not storage_file:
            # Default to file-based storage using study name
            storage_file = f"./optuna_studies/{study_info['name']}/study.db"
        
        if database_url and storage_file:
            raise ValueError("Cannot specify both database_url and storage_file. Choose one storage method.")
        
        return StudyConfig(
            study_name=study_info["name"],
            database_url=database_url,
            optimization=optimization,
            benchmark=benchmark, 
            parameters=validated_params,
            logging_config=raw_config.get("logging"),
            storage_file=storage_file
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
        
        def get_value(key: str, schema_fallback=None):
            """Get value with priority: user_config > defaults > schema > schema_fallback."""
            if key in user_config:
                return user_config[key]
            if name in self.defaults:
                return self.defaults[name]
            if key in schema_def:
                return schema_def[key]
            return schema_fallback
        
        if param_type == "range":
            return RangeParameter(
                **base_config,
                min=get_value("min", schema_def["min"]),
                max=get_value("max", schema_def["max"]),
                step=get_value("step", schema_def.get("step")),
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