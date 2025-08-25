"""Execution backend abstractions for Ray and local execution."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from ..core.trial import TrialConfig, TrialResult

logger = logging.getLogger(__name__)


class TrialFuture(ABC):
    """Abstract future for trial execution."""
    
    def __init__(self, trial_number: int):
        self.trial_number = trial_number
    
    @abstractmethod
    async def result(self) -> TrialResult:
        """Get trial result (blocking until complete)."""
        pass
    
    @abstractmethod
    def is_done(self) -> bool:
        """Check if trial is completed."""
        pass


class ExecutionBackend(ABC):
    """Abstract execution backend - supports Ray or local execution."""
    
    @abstractmethod
    async def submit_trial(self, trial_config: TrialConfig) -> TrialFuture:
        """Submit a trial for execution."""
        pass
    
    @abstractmethod
    async def wait_for_any(
        self, 
        futures: List[TrialFuture], 
        timeout: float = 1.0
    ) -> Tuple[List[TrialResult], List[TrialFuture]]:
        """Wait for any futures to complete, return completed results and remaining futures."""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Clean shutdown of backend resources."""
        pass


class RayTrialFuture(TrialFuture):
    """Ray-based trial future."""
    
    def __init__(self, trial_number: int, ray_ref):
        super().__init__(trial_number)
        self.ray_ref = ray_ref
        self._cached_result: Optional[TrialResult] = None
    
    async def result(self) -> TrialResult:
        """Get result from Ray actor."""
        if self._cached_result is None:
            import ray
            self._cached_result = ray.get(self.ray_ref)
        return self._cached_result
    
    def is_done(self) -> bool:
        """Check if Ray task is complete."""
        import ray
        ready, _ = ray.wait([self.ray_ref], timeout=0)
        return len(ready) > 0


class RayExecutionBackend(ExecutionBackend):
    """Ray-based distributed execution backend."""
    
    def __init__(self, resource_requirements: Dict[str, float]):
        self.resource_requirements = resource_requirements
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
    
    async def submit_trial(self, trial_config: TrialConfig) -> RayTrialFuture:
        """Submit trial to Ray cluster."""
        from .trial_controller import RayTrialController
        
        # Create Ray actor with resource requirements
        controller = RayTrialController.options(
            resources=self.resource_requirements
        ).remote()
        
        # Submit trial execution
        future_ref = controller.run_trial.remote(trial_config)
        
        logger.info(f"Submitted trial {trial_config.trial_number} to Ray cluster")
        return RayTrialFuture(trial_config.trial_number, future_ref)
    
    async def wait_for_any(
        self, 
        futures: List[RayTrialFuture], 
        timeout: float = 1.0
    ) -> Tuple[List[TrialResult], List[RayTrialFuture]]:
        """Wait for any Ray futures to complete."""
        import ray
        
        if not futures:
            return [], []
        
        # Check which trials are ready
        refs = [f.ray_ref for f in futures]
        ready_refs, pending_refs = ray.wait(refs, num_returns=len(refs), timeout=timeout)
        
        # Collect completed results
        completed_results = []
        pending_futures = []
        
        for future in futures:
            if future.ray_ref in ready_refs:
                try:
                    result = await future.result()
                    completed_results.append(result)
                    logger.info(f"Completed trial {future.trial_number}")
                except Exception as e:
                    # Trial failed - create error result
                    from ..core.trial import TrialResult, ExecutionInfo
                    error_result = TrialResult(
                        trial_number=future.trial_number,
                        objective_values=[],
                        detailed_metrics={},
                        execution_info=ExecutionInfo(),
                        success=False,
                        error_message=str(e)
                    )
                    completed_results.append(error_result)
                    logger.error(f"Trial {future.trial_number} failed: {e}")
            else:
                pending_futures.append(future)
        
        return completed_results, pending_futures
    
    async def shutdown(self):
        """Shutdown Ray cluster connection."""
        import ray
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Shutdown Ray cluster connection")


class LocalTrialFuture(TrialFuture):
    """Local execution trial future."""
    
    def __init__(self, trial_number: int, asyncio_future: asyncio.Future):
        super().__init__(trial_number)
        self.asyncio_future = asyncio_future
    
    async def result(self) -> TrialResult:
        """Get result from asyncio future."""
        return await self.asyncio_future
    
    def is_done(self) -> bool:
        """Check if asyncio future is complete."""
        return self.asyncio_future.done()


class LocalExecutionBackend(ExecutionBackend):
    """Local execution backend using thread/process pool."""
    
    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent)
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def submit_trial(self, trial_config: TrialConfig) -> LocalTrialFuture:
        """Submit trial for local execution."""
        from .trial_controller import LocalTrialController
        
        # Limit concurrent executions
        await self.semaphore.acquire()
        
        # Create controller and submit to executor
        controller = LocalTrialController()
        loop = asyncio.get_event_loop()
        
        future = loop.run_in_executor(
            self.executor,
            self._run_trial_with_semaphore,
            controller,
            trial_config
        )
        
        logger.info(f"Submitted trial {trial_config.trial_number} for local execution")
        return LocalTrialFuture(trial_config.trial_number, future)
    
    def _run_trial_with_semaphore(self, controller, trial_config: TrialConfig) -> TrialResult:
        """Run trial and release semaphore when done."""
        try:
            return controller.run_trial(trial_config)
        finally:
            self.semaphore.release()
    
    async def wait_for_any(
        self, 
        futures: List[LocalTrialFuture], 
        timeout: float = 1.0
    ) -> Tuple[List[TrialResult], List[LocalTrialFuture]]:
        """Wait for any local futures to complete."""
        if not futures:
            return [], []
        
        # Use asyncio.wait to check for completed futures
        done_futures, pending_futures = await asyncio.wait(
            [f.asyncio_future for f in futures],
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Collect completed results
        completed_results = []
        remaining_futures = []
        
        for future in futures:
            if future.asyncio_future in done_futures:
                try:
                    result = await future.result()
                    completed_results.append(result)
                    logger.info(f"Completed local trial {future.trial_number}")
                except Exception as e:
                    # Trial failed - create error result
                    from ..core.trial import TrialResult, ExecutionInfo
                    error_result = TrialResult(
                        trial_number=future.trial_number,
                        objective_values=[],
                        detailed_metrics={},
                        execution_info=ExecutionInfo(),
                        success=False,
                        error_message=str(e)
                    )
                    completed_results.append(error_result)
                    logger.error(f"Local trial {future.trial_number} failed: {e}")
            else:
                remaining_futures.append(future)
        
        return completed_results, remaining_futures
    
    async def shutdown(self):
        """Shutdown thread pool executor."""
        self.executor.shutdown(wait=True)
        logger.info("Shutdown local execution backend")