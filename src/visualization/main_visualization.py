import optuna
import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Import all optuna visualization modules
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline

def load_baseline_metrics(study_dir, study_id):
    """Load baseline metrics from the JSON file."""
    baseline_file = os.path.join(study_dir, f"benchmarks_{study_id}.baseline.json")
    
    if not os.path.exists(baseline_file):
        print(f"Warning: Baseline file not found at {baseline_file}")
        return None
    
    try:
        with open(baseline_file, 'r') as f:
            data = json.load(f)
        
        # Extract metrics from the benchmark data structure
        stats = data["benchmarks"][0]
        metrics = stats["metrics"]
        
        baseline_metrics = {}
        baseline_metrics['output_tokens_per_second'] = metrics['output_tokens_per_second']['successful']['median']
        baseline_metrics['request_latency'] = metrics['request_latency']['successful']['median']
        baseline_metrics['time_to_first_token_ms'] = metrics['time_to_first_token_ms']['successful']['median']
        baseline_metrics['tokens_per_second'] = metrics['tokens_per_second']['successful']['median']
        
        return baseline_metrics
    except Exception as e:
        print(f"Error loading baseline metrics: {e}")
        return None

def create_single_objective_visualization(study, baseline_metrics, study_dir, study_id):
    """Create visualization for single-objective optimization."""
    print("Creating single-objective optimization visualization...")
    
    # Create visualization directory
    viz_dir = os.path.join(study_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create optimization history plot
    fig = plot_optimization_history(study)
    
    # Add baseline as horizontal line if available
    if baseline_metrics:
        baseline_value = baseline_metrics.get('output_tokens_per_second', 0)
        fig.add_hline(
            y=baseline_value, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Baseline: {baseline_value:.2f} tokens/s",
            annotation_position="top right"
        )
    
    fig.update_layout(
        title=f"Optimization History - Study {study_id}",
        xaxis_title="Trial",
        yaxis_title="Output Tokens per Second"
    )
    
    # Save HTML
    html_path = os.path.join(viz_dir, f"optimization_history_{study_id}.html")
    
    fig.write_html(html_path)
    saved_files = [html_path]
    
    print(f"Single-objective visualization saved:")
    print(f"  HTML: {html_path}")
    
    return saved_files

def create_multi_objective_visualizations(study, baseline_metrics, study_dir, study_id):
    """Create visualizations for multi-objective optimization."""
    print("Creating multi-objective optimization visualizations...")
    
    # Create visualization directory
    viz_dir = os.path.join(study_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    saved_files = []
    
    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print("No completed trials found for visualization.")
        return []
    
    # Extract objective values and trial numbers
    trial_numbers = [trial.number for trial in completed_trials]
    throughputs = [trial.values[0] for trial in completed_trials]  # First objective: throughput
    latencies = [trial.values[1] for trial in completed_trials]    # Second objective: latency
    
    # Extract additional metrics from user_attrs for comprehensive hover info
    trial_metrics = []
    for trial in completed_trials:
        metrics = {}
        if hasattr(trial, 'user_attrs') and trial.user_attrs:
            # Extract common guidellm metrics
            metrics['output_tokens_per_second'] = trial.user_attrs.get('output_tokens_per_second', trial.values[0])
            metrics['request_latency'] = trial.user_attrs.get('request_latency', trial.values[1] if len(trial.values) > 1 else 0)
            metrics['time_to_first_token_ms'] = trial.user_attrs.get('time_to_first_token_ms', 'N/A')
            metrics['tokens_per_second'] = trial.user_attrs.get('tokens_per_second', 'N/A')
            metrics['inter_token_latency_ms'] = trial.user_attrs.get('inter_token_latency_ms', 'N/A')
            metrics['output_sequence_length'] = trial.user_attrs.get('output_sequence_length', 'N/A')
            metrics['input_sequence_length'] = trial.user_attrs.get('input_sequence_length', 'N/A')
        else:
            # Fallback to basic metrics
            metrics['output_tokens_per_second'] = trial.values[0]
            metrics['request_latency'] = trial.values[1] if len(trial.values) > 1 else 0
            metrics['time_to_first_token_ms'] = 'N/A'
            metrics['tokens_per_second'] = 'N/A'
            metrics['inter_token_latency_ms'] = 'N/A'
            metrics['output_sequence_length'] = 'N/A'
            metrics['input_sequence_length'] = 'N/A'
        trial_metrics.append(metrics)
    
    # 1. Throughput optimization history
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=trial_numbers,
        y=throughputs,
        mode='lines+markers',
        name='Throughput',
        line=dict(color='blue')
    ))
    
    if baseline_metrics:
        baseline_throughput = baseline_metrics.get('output_tokens_per_second', 0)
        fig1.add_hline(
            y=baseline_throughput,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline: {baseline_throughput:.2f} tokens/s",
            annotation_position="top right"
        )
    
    fig1.update_layout(
        title=f"Throughput Optimization History - Study {study_id}",
        xaxis_title="Trial",
        yaxis_title="Output Tokens per Second"
    )
    
    html_path1 = os.path.join(viz_dir, f"throughput_history_{study_id}.html")
    fig1.write_html(html_path1)
    saved_files.append(html_path1)
    
    # 2. Latency optimization history
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=trial_numbers,
        y=latencies,
        mode='lines+markers',
        name='Latency',
        line=dict(color='orange')
    ))
    
    if baseline_metrics:
        baseline_latency = baseline_metrics.get('request_latency', 0)
        fig2.add_hline(
            y=baseline_latency,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline: {baseline_latency:.2f} ms",
            annotation_position="top right"
        )
    
    fig2.update_layout(
        title=f"Latency Optimization History - Study {study_id}",
        xaxis_title="Trial",
        yaxis_title="Request Latency (ms)"
    )
    
    html_path2 = os.path.join(viz_dir, f"latency_history_{study_id}.html")
    fig2.write_html(html_path2)
    saved_files.append(html_path2)
    
    # 3. Combined 2D line graph with both metrics
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Create comprehensive hover text for each trial
    hover_texts = []
    for i, (trial_num, metrics) in enumerate(zip(trial_numbers, trial_metrics)):
        hover_text = f'<b>Trial {trial_num}</b><br>'
        hover_text += f'Output Tokens/s: {metrics["output_tokens_per_second"]:.2f}<br>'
        hover_text += f'Request Latency: {metrics["request_latency"]:.2f} ms<br>'
        if metrics["time_to_first_token_ms"] != 'N/A':
            hover_text += f'TTFT: {metrics["time_to_first_token_ms"]:.2f} ms<br>'
        if metrics["tokens_per_second"] != 'N/A':
            hover_text += f'Total Tokens/s: {metrics["tokens_per_second"]:.2f}<br>'
        if metrics["inter_token_latency_ms"] != 'N/A':
            hover_text += f'Inter-token Latency: {metrics["inter_token_latency_ms"]:.2f} ms<br>'
        if metrics["output_sequence_length"] != 'N/A':
            hover_text += f'Output Length: {metrics["output_sequence_length"]}<br>'
        if metrics["input_sequence_length"] != 'N/A':
            hover_text += f'Input Length: {metrics["input_sequence_length"]}'
        hover_texts.append(hover_text)
    
    # Plot throughput line
    fig3.add_trace(go.Scatter(
        x=trial_numbers,
        y=throughputs,
        mode='lines+markers',
        name='Throughput',
        line=dict(color='red', width=2),
        marker=dict(size=6),
        hovertemplate='%{text}<extra></extra>',
        text=hover_texts
    ), secondary_y=False)
    
    # Plot latency line on secondary y-axis
    fig3.add_trace(go.Scatter(
        x=trial_numbers,
        y=latencies,
        mode='lines+markers',
        name='Latency',
        line=dict(color='orange', width=2),
        marker=dict(size=6),
        yaxis="y2",
        hovertemplate='%{text}<extra></extra>',
        text=hover_texts
    ), secondary_y=True)
    
    # Add baseline horizontal lines
    if baseline_metrics:
        baseline_throughput = baseline_metrics.get('output_tokens_per_second', 0)
        baseline_latency = baseline_metrics.get('request_latency', 0)
        
        # Add horizontal line for baseline throughput
        fig3.add_hline(
            y=baseline_throughput,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Baseline Throughput: {baseline_throughput:.2f} tokens/s",
            annotation_position="top right",
            secondary_y=False
        )
        
        # Add horizontal line for baseline latency
        fig3.add_hline(
            y=baseline_latency,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Baseline Latency: {baseline_latency:.2f} ms",
            annotation_position="bottom right",
            secondary_y=True
        )
    
    # Add best value dots
    if throughputs and latencies:
        best_throughput = max(throughputs)
        best_latency = min(latencies)  # Lower latency is better
        best_throughput_idx = throughputs.index(best_throughput)
        best_latency_idx = latencies.index(best_latency)
        best_throughput_trial = trial_numbers[best_throughput_idx]
        best_latency_trial = trial_numbers[best_latency_idx]
        
        # Get comprehensive metrics for best trials
        best_throughput_metrics = trial_metrics[best_throughput_idx]
        best_latency_metrics = trial_metrics[best_latency_idx]
        
        # Create hover text for best throughput
        best_throughput_hover = f'<b>Best Throughput - Trial {best_throughput_trial}</b><br>'
        best_throughput_hover += f'Output Tokens/s: {best_throughput_metrics["output_tokens_per_second"]:.2f}<br>'
        best_throughput_hover += f'Request Latency: {best_throughput_metrics["request_latency"]:.2f} ms<br>'
        if best_throughput_metrics["time_to_first_token_ms"] != 'N/A':
            best_throughput_hover += f'TTFT: {best_throughput_metrics["time_to_first_token_ms"]:.2f} ms<br>'
        if best_throughput_metrics["tokens_per_second"] != 'N/A':
            best_throughput_hover += f'Total Tokens/s: {best_throughput_metrics["tokens_per_second"]:.2f}<br>'
        if best_throughput_metrics["inter_token_latency_ms"] != 'N/A':
            best_throughput_hover += f'Inter-token Latency: {best_throughput_metrics["inter_token_latency_ms"]:.2f} ms'
        
        # Create hover text for best latency
        best_latency_hover = f'<b>Best Latency - Trial {best_latency_trial}</b><br>'
        best_latency_hover += f'Output Tokens/s: {best_latency_metrics["output_tokens_per_second"]:.2f}<br>'
        best_latency_hover += f'Request Latency: {best_latency_metrics["request_latency"]:.2f} ms<br>'
        if best_latency_metrics["time_to_first_token_ms"] != 'N/A':
            best_latency_hover += f'TTFT: {best_latency_metrics["time_to_first_token_ms"]:.2f} ms<br>'
        if best_latency_metrics["tokens_per_second"] != 'N/A':
            best_latency_hover += f'Total Tokens/s: {best_latency_metrics["tokens_per_second"]:.2f}<br>'
        if best_latency_metrics["inter_token_latency_ms"] != 'N/A':
            best_latency_hover += f'Inter-token Latency: {best_latency_metrics["inter_token_latency_ms"]:.2f} ms'
        
        # Best throughput dot
        fig3.add_trace(go.Scatter(
            x=[best_throughput_trial],
            y=[best_throughput],
            mode='markers',
            name='Best Throughput',
            marker=dict(
                size=12,
                color='green',
                symbol='circle',
                line=dict(width=2, color='darkgreen')
            ),
            hovertemplate='%{text}<extra></extra>',
            text=[best_throughput_hover],
            showlegend=True
        ), secondary_y=False)
        
        # Best latency dot
        fig3.add_trace(go.Scatter(
            x=[best_latency_trial],
            y=[best_latency],
            mode='markers',
            name='Best Latency',
            marker=dict(
                size=12,
                color='green',
                symbol='circle',
                line=dict(width=2, color='darkgreen')
            ),
            yaxis="y2",
            hovertemplate='%{text}<extra></extra>',
            text=[best_latency_hover],
            showlegend=False  # Don't duplicate best in legend
        ), secondary_y=True)
    
    # Set x-axis title
    fig3.update_xaxes(title_text="Trial")
    
    # Set y-axes titles and ranges
    fig3.update_yaxes(title_text="Output Tokens per Second", range=[0, None], secondary_y=False)
    fig3.update_yaxes(title_text="Request Latency (ms)", secondary_y=True)
    
    fig3.update_layout(
        title=f"Multi-Objective Optimization: Combined Metrics - Study {study_id}",
        hovermode='x unified'
    )
    
    html_path3 = os.path.join(viz_dir, f"pareto_front_{study_id}.html")
    fig3.write_html(html_path3)
    saved_files.append(html_path3)
    
    print(f"Multi-objective visualizations saved:")
    for file_path in saved_files:
        print(f"  {file_path}")
    
    return saved_files

def main():
    """Main function to run the visualization program."""
    print("=== Optuna Study Visualization Tool ===")
    
    # Get study number from user
    while True:
        try:
            study_id = input("Enter the study number: ").strip()
            if not study_id:
                print("Please enter a valid study number.")
                continue
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
    # Construct paths
    current_dir = os.getcwd()
    study_dir = os.path.join(current_dir, "src", "studies", f"study_{study_id}")
    db_path = os.path.join(study_dir, "optuna.db")
    
    # Check if study exists
    if not os.path.exists(db_path):
        print(f"Error: Study database not found at {db_path}")
        print("Please make sure the study number is correct.")
        return
    
    print(f"Loading study from: {db_path}")
    
    try:
        # Load the study
        storage = optuna.storages.RDBStorage(f"sqlite:///{db_path}")
        
        # Try to find the study name - it might be different formats
        study_names_to_try = [f"study_{study_id}", f"study{study_id}", study_id]
        study = None
        
        for study_name in study_names_to_try:
            try:
                study = optuna.load_study(storage=storage, study_name=study_name)
                print(f"Found study with name: {study_name}")
                break
            except KeyError:
                continue
        
        if study is None:
            # List all available studies
            try:
                all_studies = optuna.study.get_all_study_summaries(storage)
                print("Available studies:")
                for summary in all_studies:
                    print(f"  - {summary.study_name}")
                if all_studies:
                    # Use the first study if only one exists
                    if len(all_studies) == 1:
                        study = optuna.load_study(storage=storage, study_name=all_studies[0].study_name)
                        print(f"Using the only available study: {all_studies[0].study_name}")
                    else:
                        print("Multiple studies found. Please check the study name.")
                        return
                else:
                    print("No studies found in the database.")
                    return
            except Exception as e:
                print(f"Could not load study or list studies: {e}")
                return
        
        print(f"Study loaded successfully!")
        print(f"Study name: {study.study_name}")
        
        # Handle both single and multi-objective studies
        if len(study.directions) == 1:
            print(f"Direction: {study.direction}")
        else:
            print(f"Directions: {study.directions}")
        
        print(f"Total trials: {len(study.trials)}")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print(f"Completed trials: {len(completed_trials)}")
        
        if not completed_trials:
            print("No completed trials found. Cannot create visualizations.")
            return
        
        # Load baseline metrics
        baseline_metrics = load_baseline_metrics(study_dir, study_id)
        if baseline_metrics:
            print("Baseline metrics loaded successfully!")
            print(f"Baseline throughput: {baseline_metrics['output_tokens_per_second']:.2f} tokens/s")
            print(f"Baseline latency: {baseline_metrics['request_latency']:.2f} ms")
        else:
            print("Warning: Could not load baseline metrics.")
        
        # Determine if it's multi-objective
        is_multi_objective = len(study.directions) > 1
        
        print(f"\nOptimization type: {'Multi-objective' if is_multi_objective else 'Single-objective'}")
        
        # Create appropriate visualizations
        if is_multi_objective:
            saved_files = create_multi_objective_visualizations(study, baseline_metrics, study_dir, study_id)
        else:
            saved_files = create_single_objective_visualization(study, baseline_metrics, study_dir, study_id)
        
        print(f"Visualization complete! {len(saved_files)} files saved.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

