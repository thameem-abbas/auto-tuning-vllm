"""Command-line interface for auto-tune-vllm."""

import logging
import sys
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..core.config import StudyConfig
from ..core.study_controller import StudyController
from ..visualization.dashboard import create_study_dashboard, format_study_data_for_visualization
from ..execution.backends import RayExecutionBackend
from ..logging.manager import LogStreamer, CentralizedLogger
from ..core.db_utils import verify_database_connection, clear_study_data

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


def _display_log_viewing_instructions(config: StudyConfig):
    """Display appropriate log viewing instructions based on logging configuration."""
    # Determine the correct logging database URL or file path
    log_database_url = None
    log_file_path = None
    
    if config.logging_config:
        log_database_url = config.logging_config.get("database_url")
        log_file_path = config.logging_config.get("file_path")
    
    # Default to main database if no specific logging config and database is available
    if not log_database_url and not log_file_path and config.database_url:
        log_database_url = config.database_url
    elif not log_database_url and not log_file_path and not config.database_url:
        # No PostgreSQL available - use file logging
        log_file_path = f"./logs/{config.study_name}"
    
    # Display appropriate instructions using the unified logs command
    if log_file_path:
        console.print(f"[blue]📋 View logs with: auto-tune-vllm logs --study-name {config.study_name} --log-path {log_file_path}[/blue]")
    elif log_database_url:
        console.print(f"[blue]📋 View logs with: auto-tune-vllm logs --study-name {config.study_name} --database-url {log_database_url}[/blue]")
    else:
        console.print("[blue]📋 Console logging only - no database or file logging configured[/blue]")


@app.command("optimize")
def optimize_command(
    config: str = typer.Option(..., "--config", "-c", help="Study configuration file"),
    backend: str = typer.Option("ray", "--backend", "-b", help="Execution backend: 'ray' (only supported option)"),
    n_trials: Optional[int] = typer.Option(None, "--trials", "-n", help="Number of trials (overrides config)"),
    max_concurrent: Optional[int] = typer.Option(None, "--max-concurrent", help="Max concurrent trials"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    create_db: bool = typer.Option(False, "--create-db", help="Create database if it doesn't exist"),
    start_ray_head: bool = typer.Option(False, "--start-ray-head", help="Start Ray head if no cluster is found"),
    python_executable: Optional[str] = typer.Option(None, "--python-executable", help="Explicit Python executable path for Ray workers"),
    venv_path: Optional[str] = typer.Option(None, "--venv-path", help="Virtual environment path for Ray workers"),
    conda_env: Optional[str] = typer.Option(None, "--conda-env", help="Conda environment name for Ray workers"),
):
    """Run optimization study."""
    setup_logging(verbose)
    
    # Validate Python environment options (exactly one should be specified)
    python_env_options = [python_executable, venv_path, conda_env]
    specified_options = [opt for opt in python_env_options if opt is not None]
    
    if len(specified_options) == 0:
        console.print("[bold red]Error: At least one Python environment option must be specified[/bold red]")
        console.print("Choose one of: --python-executable, --venv-path, or --conda-env")
        raise typer.Exit(1)
    
    if len(specified_options) > 1:
        console.print("[bold red]Error: Only one Python environment option can be specified at a time[/bold red]")
        console.print("Choose one of: --python-executable, --venv-path, or --conda-env")
        raise typer.Exit(1)
    
    console.print("[bold green]Starting auto-tune-vllm optimization[/bold green]")
    console.print(f"Configuration: {config}")
    console.print(f"Backend: {backend}")
    
    # Display Python environment configuration
    if python_executable:
        console.print(f"Python executable: {python_executable}")
    elif venv_path:
        console.print(f"Virtual environment: {venv_path}")
    elif conda_env:
        console.print(f"Conda environment: {conda_env}")
    elif backend.lower() == "ray":
        console.print("[yellow]No Python environment specified, using auto-detection[/yellow]")
    
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
            execution_backend = RayExecutionBackend(
                start_ray_head=start_ray_head,
                python_executable=python_executable,
                venv_path=venv_path,
                conda_env=conda_env
            )
            console.print("[blue]Using Ray distributed execution[/blue]")
            if start_ray_head:
                console.print("[blue]Will start Ray head if no cluster found[/blue]")
        else:
            console.print("[bold red]Error: Local execution backend is not supported in this version.[/bold red]")
            console.print("[bold red]Only Ray distributed execution is available.[/bold red]")
            console.print("[blue]Use --backend ray (default) or set up a Ray cluster.[/blue]")
            console.print("[blue]See docs/ray_cluster_setup.md for Ray setup instructions.[/blue]")
            raise typer.Exit(1)
        
        # Run optimization
        run_optimization_sync(
            execution_backend, 
            study_config, 
            n_trials, 
            max_concurrent,
            create_db
        )
        
    except Exception as e:
        console.print(f"[bold red]Optimization failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command("clear-study")
def clear_study_command(
    study_name: str = typer.Option(..., "--study-name", "-s", help="Name of the study to clear"),
    database_url: str = typer.Option(..., "--database-url", "-d", help="Database URL for Optuna study data"),
    clear_logs: bool = typer.Option(False, "--clear-logs", help="Also clear trial logs"),
    logs_database_url: Optional[str] = typer.Option(None, "--logs-database-url", help="Logs database URL (if different from study database)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompts"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Clear study data (trials, parameters, and optionally logs)."""
    setup_logging(verbose)
    
    console.print(f"[bold yellow]Preparing to clear study: {study_name}[/bold yellow]")
    
    try:
        # Verify database connections
        if not verify_database_connection(database_url):
            console.print(f"[bold red]Error: Cannot connect to database: {database_url}[/bold red]")
            raise typer.Exit(1)
        
        logs_db_url = logs_database_url or database_url
        if clear_logs and not verify_database_connection(logs_db_url):
            console.print(f"[bold red]Error: Cannot connect to logs database: {logs_db_url}[/bold red]")
            raise typer.Exit(1)
        
        # Show what will be deleted
        console.print("\n[bold]The following data will be deleted:[/bold]")
        console.print(f"  • Optuna study data for '{study_name}' from {database_url}")
        if clear_logs:
            console.print(f"  • Trial logs for '{study_name}' from {logs_db_url}")
        
        # Confirmation prompt
        if not force:
            confirm = typer.confirm(
                "\nAre you sure you want to permanently delete this data?",
                default=False
            )
            if not confirm:
                console.print("[yellow]Operation cancelled[/yellow]")
                raise typer.Exit(0)
        
        # Perform the clearing
        result = clear_study_data(study_name, database_url, clear_logs, logs_db_url)
        
        if result["success"]:
            console.print(f"[bold green]✅ Successfully cleared study '{study_name}'[/bold green]")
            console.print(f"  • Deleted {result['trials_deleted']} trials")
            if clear_logs:
                console.print(f"  • Deleted {result['logs_deleted']} log entries")
        else:
            console.print(f"[bold red]❌ Failed to clear study: {result['error']}[/bold red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[bold red]Error clearing study: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def run_optimization_sync(
    backend, 
    config: StudyConfig, 
    n_trials: Optional[int], 
    max_concurrent: Optional[int],
    create_db: bool = False
):
    """Synchronous optimization runner with progress display."""
    # Create study controller
    controller = StudyController.create_from_config(backend, config, create_db=create_db)
    
    # Display study name prominently in the CLI
    console.print(f"[bold cyan]🔍 Study Name: {config.study_name}[/bold cyan]")
    
    # Show appropriate log viewing instructions based on logging configuration
    _display_log_viewing_instructions(config)
    console.print()  # Add blank line for better readability
    
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
        controller.run_optimization(n_trials, max_concurrent)
        progress.update(task, completed=total_trials)
    
    # Display results
    display_optimization_results(controller)


def display_optimization_results(controller: StudyController):
    """Display optimization results in a nice table and generate visualizations."""
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
    
    # Generate visualizations
    generate_study_visualizations(controller, results)


def generate_study_visualizations(controller: StudyController, results: dict):
    """Generate and save study visualizations."""
    try:
        console.print("\n[bold blue]Generating Visualizations...[/bold blue]")
        
        # Format data for visualization
        study_data = format_study_data_for_visualization(controller)
        
        # Determine output directory
        output_dir = os.path.join(controller.log_dir or ".", "visualizations")
        
        # Create visualizations
        saved_files = create_study_dashboard(study_data, output_dir)
        
        # Report generated files
        if any(saved_files.values()):
            console.print("\n[green]📊 Visualizations Generated Successfully![/green]")
            
            # Show summary dashboard first
            if saved_files.get('summary'):
                console.print(f"[bold]📋 Main Dashboard:[/bold] {saved_files['summary'][0]}")
            
            # Show specific visualizations
            if saved_files.get('specific'):
                viz_type = "Single-objective" if results["type"] == "single_objective" else "Multi-objective"
                console.print(f"[bold]🎯 {viz_type} Analysis:[/bold]")
                for file_path in saved_files['specific']:
                    file_name = os.path.basename(file_path).replace('.html', '').replace('_', ' ').title()
                    console.print(f"  • {file_name}: {file_path}")
            
            # Show common visualizations
            if saved_files.get('common'):
                console.print(f"[bold]📈 General Analysis:[/bold]")
                for file_path in saved_files['common']:
                    file_name = os.path.basename(file_path).replace('.html', '').replace('_', ' ').title()
                    console.print(f"  • {file_name}: {file_path}")
            
            console.print(f"\n[dim]Open the dashboard in your browser to explore the results interactively.[/dim]")
        else:
            console.print("[yellow]⚠️  No visualizations generated - insufficient data or errors occurred[/yellow]")
            
    except Exception as e:
        console.print(f"[red]❌ Error generating visualizations: {e}[/red]")
        # Don't fail the entire process if visualization fails
        pass


@app.command("visualize")
def visualize_command(
    study_name: str = typer.Option(..., "--study-name", help="Study name to visualize"),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="PostgreSQL database URL"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Output directory for visualizations"),
):
    """Generate visualizations for an existing study."""
    try:
        console.print(f"[blue]Loading study: {study_name}[/blue]")
        
        # Load the study
        controller = StudyController.from_study_name(
            study_name=study_name,
            database_url=database_url
        )
        
        # Get results
        results = controller.get_optimization_results()
        
        # Set output directory
        if output_dir:
            viz_output_dir = output_dir
        else:
            viz_output_dir = os.path.join(controller.log_dir or ".", "visualizations")
        
        console.print(f"[blue]Generating visualizations in: {viz_output_dir}[/blue]")
        
        # Format data and create visualizations
        study_data = format_study_data_for_visualization(controller)
        study_data['output_dir'] = viz_output_dir
        
        saved_files = create_study_dashboard(study_data, viz_output_dir)
        
        # Report results
        if any(saved_files.values()):
            console.print("\n[green]✅ Visualizations Generated Successfully![/green]")
            
            total_files = sum(len(files) for files in saved_files.values())
            console.print(f"[bold]Generated {total_files} visualization files[/bold]")
            
            if saved_files.get('summary'):
                console.print(f"\n[bold blue]🎯 Main Dashboard:[/bold blue]")
                console.print(f"  {saved_files['summary'][0]}")
            
            console.print(f"\n[dim]Open the dashboard in your browser to explore the results.[/dim]")
        else:
            console.print("[red]❌ Failed to generate visualizations[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("logs")
def logs_command(
    study_name: str = typer.Option(..., "--study-name", help="Study Name"),
    database_url: Optional[str] = typer.Option(None, "--database-url", help="PostgreSQL database URL for remote logs"),
    log_path: Optional[str] = typer.Option(None, "--log-path", help="Base log directory path for file logs"),
    trial_number: Optional[int] = typer.Option(None, "--trial", help="Specific trial number"),
    component: Optional[str] = typer.Option(None, "--component", help="Component (vllm, benchmark, controller)"),
    follow: bool = typer.Option(True, "--follow/--no-follow", help="Follow logs in real-time"),
    tail_lines: int = typer.Option(100, "--tail", "-n", help="Number of recent lines to show when starting follow mode"),
):
    """View logs from either PostgreSQL database or local files.
    
    Provide either --database-url for remote database logs or --log-path for local file logs.
    """
    # Validate that exactly one logging source is provided
    if not database_url and not log_path:
        console.print("[bold red]Error: Must specify either --database-url or --log-path[/bold red]")
        console.print("Examples:")
        console.print("  # View database logs:")
        console.print("  auto-tune-vllm logs --study-name my_study --database-url postgresql://...")
        console.print("  # View file logs:")
        console.print("  auto-tune-vllm logs --study-name my_study --log-path /path/to/logs")
        raise typer.Exit(1)
    
    if database_url and log_path:
        console.print("[bold red]Error: Cannot specify both --database-url and --log-path[/bold red]")
        console.print("Choose one logging source.")
        raise typer.Exit(1)
    
    if database_url:
        # Use database logging (original logs command logic)
        console.print(f"[blue]Streaming logs for study {study_name} from database[/blue]")
        if follow:
            console.print(f"[dim]Showing last {tail_lines} lines, then following new logs...[/dim]")
        
        try:
            streamer = LogStreamer(study_name, database_url)
            
            # TODO: Make log streaming synchronous too
            import asyncio
            if trial_number is not None:
                asyncio.run(streamer.stream_trial_logs(trial_number, component, follow, tail_lines))
            else:
                asyncio.run(streamer.stream_study_logs(follow, tail_lines))
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Log streaming stopped by user[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Log streaming failed: {e}[/bold red]")
            raise typer.Exit(1)
    
    else:
        # Use file logging (original view-file-logs command logic)
        console.print(f"[blue]Viewing file logs for study {study_name}[/blue]")
        
        try:
            from pathlib import Path
            
            study_dir = Path(log_path) / study_name
            
            if not study_dir.exists():
                console.print(f"[yellow]No logs found for study {study_name} in {log_path}[/yellow]")
                console.print(f"Expected directory: {study_dir}")
                return
            
            # Find log files
            if trial_number is not None:
                trial_dir = study_dir / f"trial_{trial_number}"
                if not trial_dir.exists():
                    console.print(f"[yellow]No logs found for trial {trial_number}[/yellow]")
                    return
                    
                log_files = list(trial_dir.glob("*.log"))
                if component:
                    log_files = [f for f in log_files if f.stem == component]
            else:
                log_files = list(study_dir.glob("*/*.log"))
                if component:
                    log_files = [f for f in log_files if f.stem == component]
            
            if not log_files:
                console.print("[yellow]No log files found matching criteria[/yellow]")
                return
            
            # Sort files by modification time
            log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            for log_file in log_files:
                trial_name = log_file.parent.name
                component_name = log_file.stem
                
                console.print(f"\n[bold cyan]===== {trial_name}/{component_name}.log =====[/bold cyan]")
                
                try:
                    if follow:
                        # Simple file following
                        _follow_file(log_file)
                    else:
                        # Show recent lines (use tail_lines parameter instead of hardcoded 'lines')
                        _show_recent_lines(log_file, tail_lines)
                        
                except Exception as e:
                    console.print(f"[red]Error reading {log_file}: {e}[/red]")
                    
        except KeyboardInterrupt:
            console.print("\n[yellow]Log viewing stopped by user[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Failed to view file logs: {e}[/bold red]")
            raise typer.Exit(1)


@app.command("view-file-logs", deprecated=True)
def view_file_logs_command(
    study_name: str = typer.Option(..., "--study-name", help="Study Name"),
    log_path: str = typer.Option(..., "--log-path", help="Base log directory path"),
    trial_number: Optional[int] = typer.Option(None, "--trial", help="Specific trial number"),
    component: Optional[str] = typer.Option(None, "--component", help="Component (vllm, benchmark, controller)"),
    follow: bool = typer.Option(False, "--follow", help="Follow logs in real-time"),
    lines: Optional[int] = typer.Option(50, "--lines", "-n", help="Number of recent lines to show"),
):
    """[DEPRECATED] View logs from local files. Use 'logs --log-path' instead."""
    console.print("[yellow]⚠️  WARNING: 'view-file-logs' command is deprecated.[/yellow]")
    console.print("[yellow]   Use 'auto-tune-vllm logs --study-name {} --log-path {}' instead.[/yellow]".format(study_name, log_path))
    console.print()
    console.print(f"[blue]Viewing file logs for study {study_name}[/blue]")
    
    try:
        from pathlib import Path
        
        study_dir = Path(log_path) / study_name
        
        if not study_dir.exists():
            console.print(f"[yellow]No logs found for study {study_name} in {log_path}[/yellow]")
            console.print(f"Expected directory: {study_dir}")
            return
        
        # Find log files
        if trial_number is not None:
            trial_dir = study_dir / f"trial_{trial_number}"
            if not trial_dir.exists():
                console.print(f"[yellow]No logs found for trial {trial_number}[/yellow]")
                return
                
            log_files = list(trial_dir.glob("*.log"))
            if component:
                log_files = [f for f in log_files if f.stem == component]
        else:
            log_files = list(study_dir.glob("*/*.log"))
            if component:
                log_files = [f for f in log_files if f.stem == component]
        
        if not log_files:
            console.print("[yellow]No log files found matching criteria[/yellow]")
            return
        
        # Sort files by modification time
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        for log_file in log_files:
            trial_name = log_file.parent.name
            component_name = log_file.stem
            
            console.print(f"\n[bold cyan]===== {trial_name}/{component_name}.log =====[/bold cyan]")
            
            try:
                if follow:
                    # Simple file following
                    _follow_file(log_file)
                else:
                    # Show recent lines
                    _show_recent_lines(log_file, lines)
                    
            except Exception as e:
                console.print(f"[red]Error reading {log_file}: {e}[/red]")
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Log viewing stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Failed to view file logs: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("resume")
def resume_command(
    config: str = typer.Option(..., "--config", "-c", help="Study configuration file"),
    backend: str = typer.Option("ray", "--backend", "-b", help="Execution backend: 'ray' (only supported option)"),
    n_trials: Optional[int] = typer.Option(None, "--trials", "-n", help="Number of additional trials to run"),
    n_total_trials: Optional[int] = typer.Option(None, "--total-trials", help="Total number of trials to reach (overrides config)"),
    max_concurrent: Optional[int] = typer.Option(None, "--max-concurrent", help="Max concurrent trials"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    start_ray_head: bool = typer.Option(False, "--start-ray-head", help="Start Ray head if no cluster is found"),
    python_executable: Optional[str] = typer.Option(None, "--python-executable", help="Explicit Python executable path for Ray workers"),
    venv_path: Optional[str] = typer.Option(None, "--venv-path", help="Virtual environment path for Ray workers"),
    conda_env: Optional[str] = typer.Option(None, "--conda-env", help="Conda environment name for Ray workers"),
):
    """Resume an existing optimization study. Fails if the study doesn't exist."""
    setup_logging(verbose)
    
    # Validate Python environment options (exactly one should be specified)
    python_env_options = [python_executable, venv_path, conda_env]
    specified_options = [opt for opt in python_env_options if opt is not None]
    
    if len(specified_options) == 0:
        console.print("[bold red]Error: At least one Python environment option must be specified[/bold red]")
        console.print("Choose one of: --python-executable, --venv-path, or --conda-env")
        raise typer.Exit(1)
    
    if len(specified_options) > 1:
        console.print("[bold red]Error: Only one Python environment option can be specified at a time[/bold red]")
        console.print("Choose one of: --python-executable, --venv-path, or --conda-env")
        raise typer.Exit(1)
    
    console.print("[bold blue]Resuming auto-tune-vllm study[/bold blue]")
    
    try:
        study_config = StudyConfig.from_file(config)
        
        # Create backend
        if backend.lower() == "ray":
            execution_backend = RayExecutionBackend(
                start_ray_head=start_ray_head,
                python_executable=python_executable,
                venv_path=venv_path,
                conda_env=conda_env
            )
            console.print("[blue]Using Ray distributed execution[/blue]")
            if start_ray_head:
                console.print("[blue]Will start Ray head if no cluster found[/blue]")
      
        else:
            console.print("[bold red]Error: Local execution backend is not supported in this version.[/bold red]")
            console.print("[bold red]Only Ray distributed execution is available.[/bold red]")
            console.print("[blue]Use --backend ray (default) or set up a Ray cluster.[/blue]")
            console.print("[blue]See docs/ray_cluster_setup.md for Ray setup instructions.[/blue]")
            raise typer.Exit(1)
        
        # Resume study
        resume_study_sync(execution_backend, study_config, n_trials, n_total_trials, max_concurrent)
        
    except Exception as e:
        console.print(f"[bold red]Resume failed: {e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


def resume_study_sync(
    backend, 
    config: StudyConfig, 
    n_trials: Optional[int],
    n_total_trials: Optional[int],
    max_concurrent: Optional[int]
):
    """Resume study execution."""
    controller = StudyController.resume_from_config(backend, config)
    
    # Display study name prominently in the CLI
    console.print(f"[bold cyan]🔍 Study Name: {config.study_name}[/bold cyan]")
    
    # Show appropriate log viewing instructions based on logging configuration
    _display_log_viewing_instructions(config)
    console.print()  # Add blank line for better readability
    
    # Count existing trials to determine how many more to run
    n_existing = len(controller.study.trials)
    
    # Display current results if any trials exist
    if n_existing > 0:
        console.print(f"[bold yellow]Current Study Results ({n_existing} trials completed)[/bold yellow]")
        display_optimization_results(controller)
        console.print()  # Add blank line for readability
    
    if n_trials is not None:
        # --trials specifies additional trials to run
        console.print(f"Running {n_trials} additional trials...")
        controller.run_optimization(n_trials, max_concurrent)
        
        # Display updated results after running additional trials
        console.print(f"\n[bold green]Updated Study Results ({len(controller.study.trials)} total trials)[/bold green]")
        display_optimization_results(controller)
        
    elif n_total_trials is not None:
        # --total-trials specifies total trials to run
        if n_total_trials <= n_existing:
            console.print(f"Study already has {n_existing} trials. Specified --total-trials={n_total_trials} would not run any new trials.")
            console.print("Use --trials to run more trials, or increase --total-trials count.")
        else:
            trials_to_run = n_total_trials - n_existing
            console.print(f"Running {trials_to_run} more trials to reach total of {n_total_trials} trials...")
            controller.run_optimization(trials_to_run, max_concurrent)
            
            # Display updated results after running additional trials
            console.print(f"\n[bold green]Final Study Results ({len(controller.study.trials)} total trials)[/bold green]")
            display_optimization_results(controller)
            
    else:
        console.print("Study resumed. Use --trials to run additional trials or --total-trials to set total trial count.")


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
        # Safe DB/Storage display
        storage_display = study_config.database_url or study_config.storage_file or "(none)"
        if isinstance(storage_display, str) and len(storage_display) > 53:
            storage_display = storage_display[:50] + "..."
        table.add_row("Database/Storage", storage_display)
        # Optimization summary (supports structured objectives)
        try:
            if study_config.optimization.is_multi_objective:
                obj_summaries = []
                for obj in study_config.optimization.objectives:
                    obj_summaries.append(f"{obj.metric}:{obj.direction}:{obj.percentile}")
                opt_summary = f"multi_objective [{', '.join(obj_summaries)}]"
            else:
                obj = study_config.optimization.objectives[0]
                opt_summary = f"single_objective {obj.metric}:{obj.direction}:{obj.percentile}"
        except Exception:
            opt_summary = str(getattr(study_config.optimization, 'objective', 'unknown'))
        table.add_row("Optimization", f"{opt_summary} ({study_config.optimization.sampler})")
        table.add_row("Trials", str(study_config.optimization.n_trials))
        table.add_row("Model", study_config.benchmark.model)
        table.add_row("Parameters", str(len([p for p in study_config.parameters.values() if p.enabled])))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]Validation failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("setup-logging")
def setup_logging_command(
    database_url: Optional[str] = typer.Option(None, "--database-url", help="PostgreSQL database URL for logs"),
    file_path: Optional[str] = typer.Option(None, "--file-path", help="File path for local logging"),
    study_id: Optional[int] = typer.Option(None, "--study-id", help="Study ID (optional)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Setup logging infrastructure (PostgreSQL database tables or file directory).
    
    Note: File logging setup is optional - directories are created automatically during optimization.
    This command is mainly useful for testing permissions and setting up PostgreSQL tables.
    """
    setup_logging(verbose)
    
    if not database_url and not file_path:
        console.print("[bold red]Error: Must specify either --database-url or --file-path[/bold red]")
        console.print("[yellow]Note: For file logging, you can just add 'file_path' to your study config - no setup required![/yellow]")
        raise typer.Exit(1)
    
    console.print("[blue]Setting up logging infrastructure[/blue]")
    
    try:
        if database_url:
            # Setup PostgreSQL logging
            if not verify_database_connection(database_url):
                console.print(f"[bold red]Error: Cannot connect to database: {database_url}[/bold red]")
                raise typer.Exit(1)
            
            # Use dummy study ID if not provided
            dummy_study_id = study_id or 0
            
            # Initialize CentralizedLogger to create tables
            CentralizedLogger(
                study_id=dummy_study_id,
                pg_url=database_url,
                log_level="INFO"
            )
            
            console.print("[bold green]✅ PostgreSQL logging tables created successfully[/bold green]")
            console.print(f"Database: {database_url}")
            console.print("Tables: trial_logs (with indexes)")
            
            if study_id:
                console.print(f"\n[blue]You can now view logs for study with ID {study_id}:[/blue]")
                console.print(f"auto-tune-vllm logs --study-name <study_name> --database-url {database_url}")
        
        if file_path:
            # Setup file logging
            from pathlib import Path
            log_dir = Path(file_path)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = log_dir / "test_write.log"
            try:
                with open(test_file, 'w') as f:
                    f.write("test\n")
                test_file.unlink()  # Delete test file
            except Exception as e:
                console.print(f"[bold red]Error: Cannot write to {file_path}: {e}[/bold red]")
                raise typer.Exit(1)
            
            console.print("[bold green]✅ File logging directory ready[/bold green]")
            console.print(f"Log path: {file_path}")
            console.print(f"Structure: {file_path}/<study_name>/trial_<NUM>/<component>.log")
            console.print("[dim]Note: This setup is optional - just add 'file_path' to your study config instead![/dim]")
            
            if study_id:
                console.print("\n[blue]You can view logs for any study with:[/blue]")
                console.print(f"auto-tune-vllm logs --study-name <study_name> --log-path {file_path}")
        
    except Exception as e:
        console.print(f"[bold red]Failed to setup logging: {e}[/bold red]")
        if verbose:
            console.print_exception()
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
            node_id = ray.get_runtime_context().get_node_id()
            return f"Worker node {node_id[:8]}"
        
        # Get cluster resources to determine number of nodes
        cluster_resources = ray.cluster_resources()
        
        # Run environment check on multiple workers (if available)
        console.print("Checking worker nodes...")
        num_workers = min(4, int(cluster_resources.get("num_cpus", 1)) // 2)  # Don't overwhelm cluster
        
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


def _show_recent_lines(log_file: Path, lines: int):
    """Show recent lines from a log file."""
    try:
        import subprocess
        result = subprocess.run(
            ["tail", "-n", str(lines), str(log_file)],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout:
            console.print(result.stdout)
        else:
            console.print("[dim]File is empty[/dim]")
    except subprocess.CalledProcessError:
        # Fallback to Python implementation
        try:
            with open(log_file, 'r') as f:
                file_lines = f.readlines()
                recent_lines = file_lines[-lines:] if len(file_lines) > lines else file_lines
                for line in recent_lines:
                    console.print(line.rstrip())
        except Exception as e:
            console.print(f"[red]Error reading file: {e}[/red]")


def _follow_file(log_file: Path):
    """Follow a log file in real-time."""
    console.print(f"[blue]Following {log_file} (Press Ctrl+C to stop)[/blue]")
    
    try:
        import subprocess
        process = subprocess.Popen(
            ["tail", "-f", str(log_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in iter(process.stdout.readline, ''):
            console.print(line.rstrip())
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped following file[/yellow]")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        console.print(f"[red]Error following file: {e}[/red]")


def main():
    """Main CLI entry point."""
    # Handle no arguments case
    if len(sys.argv) == 1:
        console.print("[bold]auto-tune-vllm[/bold] - Distributed vLLM hyperparameter optimization")
        console.print("\nUse --help for available commands")
        console.print("\nQuick start:")
        console.print("  auto-tune-vllm optimize --config study.yaml")
        console.print("  auto-tune-vllm logs --study-name my_study --database-url postgresql://...")
        console.print("  auto-tune-vllm logs --study-name my_study --log-path /path/to/logs")
        console.print("  auto-tune-vllm setup-logging --database-url postgresql://...")
        console.print("  auto-tune-vllm clear-study --study-name my_study --database-url postgresql://...")
        sys.exit(0)
    
    app()


if __name__ == "__main__":
    main()