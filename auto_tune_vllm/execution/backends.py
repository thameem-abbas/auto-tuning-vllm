"""Execution backend abstractions for Ray and local execution."""

from __future__ import annotations

import concurrent.futures
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..core.trial import TrialConfig, TrialResult

logger = logging.getLogger(__name__)


@dataclass
class JobHandle:
    """Handle for submitted trial job."""
    trial_number: int
    backend_job_id: str  # Ray ObjectRef ID, process PID, etc.
    status: str = "running"  # "running", "completed", "failed"
    submitted_at: float = 0.0
    
    def __post_init__(self):
        if self.submitted_at == 0.0:
            self.submitted_at = time.time()


class ExecutionBackend(ABC):
    """Abstract execution backend - supports Ray or local execution."""
    
    @abstractmethod
    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit a trial for execution."""
        pass
    
    @abstractmethod
    def poll_trials(self, job_handles: List[JobHandle]) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll for completed trials, return completed results and remaining handles."""
        pass
    
    @abstractmethod
    def shutdown(self):
        """Clean shutdown of backend resources."""
        pass




class RayExecutionBackend(ExecutionBackend):
    """Ray-based distributed execution backend."""
    
    def __init__(self, resource_requirements: Dict[str, float]):
        self.resource_requirements = resource_requirements
        self.active_jobs: Dict[str, object] = {}  # job_id -> ray_ref
        self._ensure_ray_initialized()
    
    def _ensure_ray_initialized(self):
        """Initialize Ray if not already initialized."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
                logger.info("Initialized Ray cluster connection")
        except ImportError:
            raise ImportError("Ray is required for RayExecutionBackend. Install with: pip install ray[default]")
    
    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit trial to Ray cluster."""
        from .trial_controller import RayTrialController
        
        # Create Ray actor with resource requirements
        controller = RayTrialController.options(
            resources=self.resource_requirements
        ).remote()
        
        # Submit trial execution
        future_ref = controller.run_trial.remote(trial_config)
        job_id = str(future_ref)  # Use Ray ObjectRef as job ID
        
        # Track active job
        self.active_jobs[job_id] = future_ref
        
        logger.info(f"Submitted trial {trial_config.trial_number} to Ray cluster")
        return JobHandle(trial_config.trial_number, job_id)
    
    def poll_trials(self, job_handles: List[JobHandle]) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll for completed Ray trials."""
        import ray
        
        if not job_handles:
            return [], []
        
        # Get Ray refs for active jobs
        active_refs = []
        handle_map = {}
        
        for handle in job_handles:
            if handle.backend_job_id in self.active_jobs:
                ray_ref = self.active_jobs[handle.backend_job_id]
                active_refs.append(ray_ref)
                handle_map[ray_ref] = handle
        
        if not active_refs:
            return [], job_handles
        
        # Check which trials are ready (non-blocking)
        ready_refs, pending_refs = ray.wait(active_refs, num_returns=len(active_refs), timeout=0)
        
        completed_results = []
        remaining_handles = []
        
        for handle in job_handles:
            ray_ref = self.active_jobs.get(handle.backend_job_id)
            
            if ray_ref in ready_refs:
                try:
                    result = ray.get(ray_ref)  # Get completed result
                    completed_results.append(result)
                    logger.info(f"Completed trial {handle.trial_number}")
                    # Remove from active jobs
                    del self.active_jobs[handle.backend_job_id]
                except Exception as e:
                    # Trial failed - create error result
                    from ..core.trial import TrialResult, ExecutionInfo
                    error_result = TrialResult(
                        trial_number=handle.trial_number,
                        objective_values=[],
                        detailed_metrics={},
                        execution_info=ExecutionInfo(),
                        success=False,
                        error_message=str(e)
                    )
                    completed_results.append(error_result)
                    logger.error(f"Trial {handle.trial_number} failed: {e}")
                    # Remove from active jobs
                    del self.active_jobs[handle.backend_job_id]
            else:
                remaining_handles.append(handle)
        
        return completed_results, remaining_handles
    
    def shutdown(self):
        """Shutdown Ray cluster connection."""
        import ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Shutdown Ray cluster connection")




class LocalExecutionBackend(ExecutionBackend):
    """Local execution backend using thread/process pool."""
    
    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
        self.active_futures: Dict[str, concurrent.futures.Future] = {}
    
    def submit_trial(self, trial_config: TrialConfig) -> JobHandle:
        """Submit trial for local execution."""
        from .trial_controller import LocalTrialController
        
        # Create controller and submit to executor
        controller = LocalTrialController()
        
        future = self.executor.submit(
            controller.run_trial,
            trial_config
        )
        
        job_id = str(id(future))  # Use future object ID as job ID
        self.active_futures[job_id] = future
        
        logger.info(f"Submitted trial {trial_config.trial_number} for local execution")
        return JobHandle(trial_config.trial_number, job_id)
    
    def poll_trials(self, job_handles: List[JobHandle]) -> Tuple[List[TrialResult], List[JobHandle]]:
        """Poll for completed local trials."""
        if not job_handles:
            return [], []
        
        completed_results = []
        remaining_handles = []
        
        for handle in job_handles:
            future = self.active_futures.get(handle.backend_job_id)
            
            if future and future.done():
                try:
                    result = future.result()
                    completed_results.append(result)
                    logger.info(f"Completed local trial {handle.trial_number}")
                    # Remove from active futures
                    del self.active_futures[handle.backend_job_id]
                except Exception as e:
                    # Trial failed - create error result
                    from ..core.trial import TrialResult, ExecutionInfo
                    error_result = TrialResult(
                        trial_number=handle.trial_number,
                        objective_values=[],
                        detailed_metrics={},
                        execution_info=ExecutionInfo(),
                        success=False,
                        error_message=str(e)
                    )
                    completed_results.append(error_result)
                    logger.error(f"Local trial {handle.trial_number} failed: {e}")
                    # Remove from active futures
                    del self.active_futures[handle.backend_job_id]
            else:
                remaining_handles.append(handle)
        
        return completed_results, remaining_handles
    
    def shutdown(self):
        """Shutdown thread pool executor."""
        self.executor.shutdown(wait=True)
        logger.info("Shutdown local execution backend")