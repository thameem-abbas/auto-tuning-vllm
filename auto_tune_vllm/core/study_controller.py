"""Study controller for orchestrating optimization studies."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

import optuna
from optuna.samplers import (
    GridSampler, 
    NSGAIISampler,
    RandomSampler,
    TPESampler
)
import optuna.integration

from ..execution.backends import ExecutionBackend, TrialFuture
from .config import StudyConfig
from .trial import TrialConfig

logger = logging.getLogger(__name__)


class StudyController:
    """Main orchestration controller for optimization studies."""
    
    def __init__(
        self, 
        backend: ExecutionBackend, 
        study: optuna.Study, 
        config: StudyConfig
    ):
        self.backend = backend
        self.study = study
        self.config = config
        self.active_trials: Dict[int, TrialFuture] = {}
        self.completed_trials = 0
        
    @classmethod
    def create_from_config(
        cls, 
        backend: ExecutionBackend, 
        config: StudyConfig
    ) -> StudyController:
        """Create study controller with Optuna study from configuration."""
        # Create sampler based on config
        sampler = cls._create_sampler(config)
        
        # Determine optimization direction(s)
        if isinstance(config.optimization.objective, list):
            directions = config.optimization.objective
        else:
            directions = config.optimization.objective
        
        # Create Optuna study
        study = optuna.create_study(
            storage=config.database_url,
            study_name=config.study_name,
            direction=directions,  # "maximize", "minimize", or ["maximize", "minimize"] 
            sampler=sampler,
            load_if_exists=True
        )
        
        logger.info(f"Created study: {config.study_name} with {config.optimization.sampler} sampler")
        
        return cls(backend, study, config)
    
    @staticmethod
    def _create_sampler(config: StudyConfig) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler from configuration."""
        sampler_name = config.optimization.sampler.lower()
        
        if sampler_name == "tpe":
            return TPESampler()
        elif sampler_name == "random":
            return RandomSampler()
        elif sampler_name == "botorch":
            return optuna.integration.BoTorchSampler()
        elif sampler_name == "nsga2":
            return NSGAIISampler()
        elif sampler_name == "grid":
            # Build search space for grid sampler
            search_space = {}
            for param_name, param_config in config.parameters.items():
                if param_config.enabled:
                    # Convert parameter config to grid search space
                    if hasattr(param_config, 'options'):
                        search_space[param_name] = param_config.options
                    elif hasattr(param_config, 'min_value'):
                        # Generate discrete values for range parameters
                        values = []
                        current = param_config.min_value
                        while current <= param_config.max_value:
                            values.append(current)
                            current += param_config.step or 1
                        search_space[param_name] = values
                    else:
                        search_space[param_name] = [True, False]  # Boolean
            
            grid_size = StudyController._calculate_grid_size(search_space)
            logger.info(f"Grid search space: {len(search_space)} parameters, "
                       f"{grid_size} total combinations")
            return GridSampler(search_space)
        else:
            logger.warning(f"Unknown sampler: {sampler_name}, using TPE")
            return TPESampler()
    
    @staticmethod
    def _calculate_grid_size(search_space: Dict) -> int:
        """Calculate total grid search combinations."""
        size = 1
        for values in search_space.values():
            size *= len(values)
        return size
    
    async def run_optimization(
        self, 
        n_trials: Optional[int] = None,
        max_concurrent: Optional[int] = None
    ) -> optuna.Study:
        """
        Run optimization study.
        
        Args:
            n_trials: Number of trials to run (overrides config)
            max_concurrent: Maximum concurrent trials (None = unlimited)
            
        Returns:
            Completed Optuna study
        """
        total_trials = n_trials or self.config.optimization.n_trials
        max_concurrent = max_concurrent or float('inf')
        
        logger.info(f"Starting optimization: {total_trials} trials, "
                   f"max concurrent: {max_concurrent if max_concurrent != float('inf') else 'unlimited'}")
        
        try:
            while self.completed_trials < total_trials:
                # Submit new trials up to concurrency limit
                await self._submit_available_trials(
                    remaining_trials=total_trials - self.completed_trials - len(self.active_trials),
                    max_concurrent=max_concurrent
                )
                
                # Collect completed trials
                if self.active_trials:
                    newly_completed = await self._collect_completed_trials()
                    self.completed_trials += newly_completed
                    
                    logger.info(f"Progress: {self.completed_trials}/{total_trials} trials completed")
                
                # Brief sleep to prevent tight loop
                await asyncio.sleep(1)
            
            # Wait for any remaining trials
            while self.active_trials:
                newly_completed = await self._collect_completed_trials()
                self.completed_trials += newly_completed
                logger.info(f"Final cleanup: {self.completed_trials} trials completed")
                
                if not self.active_trials:
                    break
                await asyncio.sleep(1)
            
            logger.info(f"Optimization completed: {self.completed_trials} trials")
            return self.study
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
        finally:
            await self.backend.shutdown()
    
    async def _submit_available_trials(self, remaining_trials: int, max_concurrent: float):
        """Submit new trials up to limits."""
        while (remaining_trials > 0 and 
               len(self.active_trials) < max_concurrent):
            
            # Ask Optuna for next trial
            try:
                trial = self.study.ask()
            except Exception as e:
                logger.error(f"Failed to get next trial from Optuna: {e}")
                break
            
            # Build trial configuration
            trial_config = self._build_trial_config(trial)
            
            # Submit to execution backend
            try:
                future = await self.backend.submit_trial(trial_config)
                self.active_trials[trial.number] = future
                
                logger.info(f"Submitted trial {trial.number} with parameters: {trial.params}")
                remaining_trials -= 1
                
            except Exception as e:
                logger.error(f"Failed to submit trial {trial.number}: {e}")
                # Mark trial as failed in Optuna
                self.study.tell(trial.number, None)  # Failed trial
                break
    
    async def _collect_completed_trials(self) -> int:
        """Collect completed trials and report results to Optuna."""
        if not self.active_trials:
            return 0
        
        # Wait for any trials to complete
        futures_list = list(self.active_trials.values())
        completed_results, remaining_futures = await self.backend.wait_for_any(
            futures_list, timeout=2.0
        )
        
        completed_count = 0
        
        for result in completed_results:
            trial_number = result.trial_number
            
            # Remove from active trials
            if trial_number in self.active_trials:
                del self.active_trials[trial_number]
            
            # Report to Optuna
            try:
                if result.success and result.objective_values:
                    self.study.tell(trial_number, result.objective_values)
                    logger.info(f"Trial {trial_number} completed successfully: {result.objective_values}")
                else:
                    # Failed trial
                    self.study.tell(trial_number, None)
                    logger.error(f"Trial {trial_number} failed: {result.error_message}")
                
                completed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to report trial {trial_number} to Optuna: {e}")
        
        return completed_count
    
    def _build_trial_config(self, trial: optuna.Trial) -> TrialConfig:
        """Build trial configuration from Optuna trial."""
        # Generate parameter values using configured parameter definitions
        parameters = {}
        
        for param_name, param_config in self.config.parameters.items():
            if param_config.enabled:
                value = param_config.generate_optuna_suggest(trial)
                parameters[param_name] = value
        
        return TrialConfig(
            study_id=hash(self.config.study_name),  # Simple study ID derivation
            trial_number=trial.number,
            parameters=parameters,
            benchmark_config=self.config.benchmark
        )
    
    def get_optimization_results(self) -> Dict:
        """Get optimization results summary."""
        if isinstance(self.config.optimization.objective, list):
            # Multi-objective results
            return {
                "type": "multi_objective",
                "n_trials": len(self.study.trials),
                "n_pareto_solutions": len(self.study.best_trials),
                "pareto_front": [
                    {"trial": t.number, "values": t.values, "params": t.params} 
                    for t in self.study.best_trials[:10]  # Top 10
                ]
            }
        else:
            # Single objective results
            best_trial = self.study.best_trial
            return {
                "type": "single_objective",
                "n_trials": len(self.study.trials),
                "best_value": best_trial.value,
                "best_params": best_trial.params,
                "best_trial_number": best_trial.number
            }
    
    async def resume_study(self) -> StudyController:
        """Resume an existing study from the database."""
        logger.info(f"Resuming study: {self.config.study_name}")
        
        # Count existing trials
        n_existing = len(self.study.trials)
        logger.info(f"Found {n_existing} existing trials in study")
        
        return self