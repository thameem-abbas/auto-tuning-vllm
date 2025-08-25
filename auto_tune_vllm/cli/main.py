"""Command-line interface for auto-tune-vllm."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..core.config import StudyConfig
from ..core.study_controller import StudyController
from ..execution.backends import RayExecutionBackend, LocalExecutionBackend
from ..logging.manager import LogStreamer

# Setup rich console and app
console = Console()
app = typer.Typer(
    name="auto-tune-vllm",
    help="Distributed hyperparameter optimization for vLLM serving",
    add_completion=False
)


def setup_logging(verbose: bool = False):
    """Setup logging with Rich handler."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )


@app.command("optimize")
def optimize_command(
    config: str = typer.Option(..., "--config", "-c", help="Study configuration file"),
    backend: str = typer.Option("ray", "--backend", "-b", help="Execution backend: 'ray' or 'local'"),
    n_trials: Optional[int] = typer.Option(None, "--trials", "-n", help="Number of trials (overrides config)"),
    max_concurrent: Optional[int] = typer.Option(None, "--max-concurrent", help="Max concurrent trials"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Run optimization study."""
    setup_logging(verbose)
    
    console.print("[bold green]Starting auto-tune-vllm optimization[/bold green]")
    console.print(f"Configuration: {config}")
    console.print(f"Backend: {backend}")
    
    try:
        # Load configuration
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[bold red]Error: Configuration file not found: {config}[/bold red]")
            raise typer.Exit(1)
        
        study_config = StudyConfig.from_file(config)
        console.print(f"Loaded study: {study_config.study_name}")
        
        # Create execution backend
        if backend.lower() == "ray":
            execution_backend = RayExecutionBackend({"GPU": 1, "CPU": 4})
            console.print("[blue]Using Ray distributed execution[/blue]")
        elif backend.lower() == "local":
            execution_backend = LocalExecutionBackend(max_concurrent or 2)
            console.print("[yellow]Using local execution[/yellow]")
        else:
            console.print(f"[bold red]Error: Unknown backend: {backend}[/bold red]")
            raise typer.Exit(1)
        
        # Run optimization
        asyncio.run(run_optimization_async(
            execution_backend, 
            study_config, 
            n_trials, 
            max_concurrent
        ))
        
    except Exception as e:
        console.print(f"[bold red]Optimization failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


async def run_optimization_async(
    backend, 
    config: StudyConfig, 
    n_trials: Optional[int], 
    max_concurrent: Optional[int]
):
    """Async optimization runner with progress display."""
    # Create study controller
    controller = StudyController.create_from_config(backend, config)
    
    total_trials = n_trials or config.optimization.n_trials
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Add progress task
        task = progress.add_task(
            f"Running {total_trials} optimization trials...", 
            total=total_trials
        )
        
        # Run optimization
        await controller.run_optimization(n_trials, max_concurrent)
        progress.update(task, completed=total_trials)
    
    # Display results
    display_optimization_results(controller)


def display_optimization_results(controller: StudyController):
    """Display optimization results in a nice table."""
    results = controller.get_optimization_results()
    
    console.print("\n[bold green]Optimization Results[/bold green]")
    
    if results["type"] == "single_objective":
        table = Table(title="Best Trial Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Trials", str(results["n_trials"]))
        table.add_row("Best Value", f"{results['best_value']:.4f}")
        table.add_row("Best Trial", str(results["best_trial_number"]))
        
        console.print(table)
        
        # Parameters table
        params_table = Table(title="Best Parameters")
        params_table.add_column("Parameter", style="cyan")
        params_table.add_column("Value", style="yellow")
        
        for param, value in results["best_params"].items():
            params_table.add_row(param, str(value))
        
        console.print(params_table)
        
    else:  # Multi-objective
        table = Table(title="Pareto-Optimal Solutions")
        table.add_column("Trial", style="cyan")
        table.add_column("Objective 1", style="green")
        table.add_column("Objective 2", style="green")
        table.add_column("Top Parameters", style="yellow")
        
        for solution in results["pareto_front"][:5]:  # Show top 5
            # Show first few parameters
            top_params = list(solution["params"].items())[:3]
            params_str = ", ".join(f"{k}={v}" for k, v in top_params)
            
            table.add_row(
                str(solution["trial"]),
                f"{solution['values'][0]:.4f}",
                f"{solution['values'][1]:.4f}",
                params_str
            )
        
        console.print(table)
        console.print(f"[blue]Total Pareto solutions: {results['n_pareto_solutions']}[/blue]")


@app.command("logs")
def logs_command(
    study_id: int = typer.Option(..., "--study-id", help="Study ID"),
    database_url: str = typer.Option(..., "--database-url", help="PostgreSQL database URL"),
    trial_number: Optional[int] = typer.Option(None, "--trial", help="Specific trial number"),
    component: Optional[str] = typer.Option(None, "--component", help="Component (vllm, benchmark, controller)"),
    follow: bool = typer.Option(True, "--follow/--no-follow", help="Follow logs in real-time"),
):
    """Stream logs from PostgreSQL database."""
    console.print(f"[blue]Streaming logs for study {study_id}[/blue]")
    
    try:
        streamer = LogStreamer(study_id, database_url)
        
        if trial_number is not None:
            asyncio.run(streamer.stream_trial_logs(trial_number, component, follow))
        else:
            asyncio.run(streamer.stream_study_logs(follow))
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Log streaming stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Log streaming failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("resume")
def resume_command(
    config: str = typer.Option(..., "--config", "-c", help="Study configuration file"),
    backend: str = typer.Option("ray", "--backend", "-b", help="Execution backend"),
    n_additional_trials: Optional[int] = typer.Option(None, "--additional-trials", help="Additional trials to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Resume an existing optimization study."""
    setup_logging(verbose)
    
    console.print("[bold blue]Resuming auto-tune-vllm study[/bold blue]")
    
    try:
        study_config = StudyConfig.from_file(config)
        
        # Create backend
        if backend.lower() == "ray":
            execution_backend = RayExecutionBackend({"GPU": 1, "CPU": 4})
        else:
            execution_backend = LocalExecutionBackend(max_concurrent=2)
        
        # Resume study
        asyncio.run(resume_study_async(execution_backend, study_config, n_additional_trials))
        
    except Exception as e:
        console.print(f"[bold red]Resume failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


async def resume_study_async(
    backend, 
    config: StudyConfig, 
    n_additional_trials: Optional[int]
):
    """Resume study execution."""
    controller = StudyController.create_from_config(backend, config)
    await controller.resume_study()
    
    if n_additional_trials:
        console.print(f"Running {n_additional_trials} additional trials...")
        await controller.run_optimization(n_additional_trials)
    else:
        console.print("Study resumed. Use --additional-trials to run more trials.")


@app.command("validate")
def validate_command(
    config: str = typer.Option(..., "--config", "-c", help="Configuration file to validate"),
):
    """Validate study configuration file."""
    console.print(f"[blue]Validating configuration: {config}[/blue]")
    
    try:
        config_path = Path(config)
        if not config_path.exists():
            console.print(f"[bold red]Error: File not found: {config}[/bold red]")
            raise typer.Exit(1)
        
        study_config = StudyConfig.from_file(config)
        
        console.print("[bold green]✓ Configuration is valid[/bold green]")
        
        # Display summary
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Study Name", study_config.study_name)
        table.add_row("Database URL", study_config.database_url[:50] + "...")
        table.add_row("Optimization", f"{study_config.optimization.objective} ({study_config.optimization.sampler})")
        table.add_row("Trials", str(study_config.optimization.n_trials))
        table.add_row("Model", study_config.benchmark.model)
        table.add_row("Parameters", str(len([p for p in study_config.parameters.values() if p.enabled])))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Validation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("check-env")
def check_environment_command(
    ray_cluster: bool = typer.Option(False, "--ray-cluster", help="Check all nodes in Ray cluster"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Check environment and dependencies on local machine or Ray cluster."""
    setup_logging(verbose)
    
    console.print("[blue]Checking environment and dependencies[/blue]")
    
    try:
        
        if ray_cluster:
            console.print("[yellow]Checking Ray cluster environment...[/yellow]")
            _check_ray_cluster_environment()
        else:
            console.print("[yellow]Checking local environment...[/yellow]")
            from ..execution.trial_controller import LocalTrialController
            controller = LocalTrialController()
            controller._validate_environment()
            console.print("[bold green]✓ Local environment check passed[/bold green]")
            
    except Exception as e:
        console.print(f"[bold red]Environment check failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def _check_ray_cluster_environment():
    """Check environment on all Ray cluster nodes."""
    try:
        import ray
        
        if not ray.is_initialized():
            console.print("[yellow]Initializing Ray connection...[/yellow]")
            ray.init(address="auto")
        
        # Check head node first
        console.print("Checking head node...")
        from ..execution.trial_controller import LocalTrialController
        controller = LocalTrialController()
        controller._validate_environment()
        console.print("✓ Head node environment OK")
        
        # Check worker nodes by running environment validation as Ray task
        @ray.remote
        def check_worker_environment():
            from auto_tune_vllm.execution.trial_controller import LocalTrialController
            controller = LocalTrialController()
            controller._validate_environment()
            
            # Return node info
            import ray
            node_id = ray.get_runtime_context().node_id.hex()
            return f"Worker node {node_id[:8]}"
        
        # Get cluster resources to determine number of nodes
        cluster_resources = ray.cluster_resources()
        
        # Run environment check on multiple workers (if available)
        console.print("Checking worker nodes...")
        num_workers = min(4, int(cluster_resources.get("CPU", 1)) // 2)  # Don't overwhelm cluster
        
        if num_workers > 1:
            futures = [check_worker_environment.remote() for _ in range(num_workers)]
            results = ray.get(futures)
            
            for result in results:
                console.print(f"✓ {result} environment OK")
        
        console.print(f"[bold green]✓ Ray cluster environment check passed ({num_workers} nodes checked)[/bold green]")
        
    except ImportError:
        raise RuntimeError("Ray is not installed. Cannot check Ray cluster environment.")
    except Exception as e:
        raise RuntimeError(f"Ray cluster environment check failed: {str(e)}")


def main():
    """Main CLI entry point."""
    # Handle no arguments case
    if len(sys.argv) == 1:
        console.print("[bold]auto-tune-vllm[/bold] - Distributed vLLM hyperparameter optimization")
        console.print("\nUse --help for available commands")
        console.print("\nQuick start:")
        console.print("  auto-tune-vllm optimize --config study.yaml")
        console.print("  auto-tune-vllm logs --study-id 42 --database-url postgresql://...")
        sys.exit(0)
    
    app()


if __name__ == "__main__":
    main()