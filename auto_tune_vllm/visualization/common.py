"""
Common visualization utilities shared between single and multi-objective studies.
"""

import os
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime


def get_visualization_style() -> Dict[str, Any]:
    """Get consistent styling for all visualizations."""
    return {
        'template': 'plotly_white',
        'color_palette': [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ],
        'pareto_color': '#2ca02c',
        'dominated_color': '#cccccc',
        'baseline_color': '#d62728',
        'improvement_color': '#1f77b4',
        'font_family': 'Arial, sans-serif',
        'title_size': 16,
        'axis_title_size': 14,
        'legend_size': 12
    }


def create_baseline_comparison(study_data: Dict, output_dir: str) -> str:
    """Create baseline comparison visualization."""
    style = get_visualization_style()
    
    # Extract baseline data if available
    baseline_data = study_data.get('baseline_metrics', {})
    trials = study_data.get('completed_trials', [])
    
    if not baseline_data or not trials:
        return ""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Performance vs Baseline',
            'Improvement Distribution', 
            'Best vs Baseline Metrics',
            'Improvement Over Time'
        ),
        specs=[
            [{"type": "bar"}, {"type": "histogram"}],
            [{"type": "bar"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Performance vs Baseline (bar chart)
    if study_data['type'] == 'single_objective':
        best_value = study_data.get('best_value', 0)
        baseline_value = list(baseline_data.values())[0].get('output_tokens_per_second', 0)
        improvement = ((best_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0
        
        fig.add_trace(
            go.Bar(
                x=['Baseline', 'Best Trial'],
                y=[baseline_value, best_value],
                marker_color=[style['baseline_color'], style['improvement_color']],
                text=[f'{baseline_value:.2f}', f'{best_value:.2f}'],
                textposition='auto',
                name='Performance'
            ),
            row=1, col=1
        )
        
        # 2. Improvement distribution
        trial_values = [trial.get('value', 0) for trial in trials]
        improvements = [((val - baseline_value) / baseline_value) * 100 
                       for val in trial_values if baseline_value > 0]
        
        fig.add_trace(
            go.Histogram(
                x=improvements,
                nbinsx=20,
                marker_color=style['improvement_color'],
                opacity=0.7,
                name='Improvements'
            ),
            row=1, col=2
        )
        
    else:  # Multi-objective
        # Show comparison for multiple baselines
        baseline_keys = list(baseline_data.keys())
        if baseline_keys:
            baseline_key = baseline_keys[0]  # Use first baseline for comparison
            baseline_throughput = baseline_data[baseline_key].get('output_tokens_per_second', 0)
            baseline_latency = baseline_data[baseline_key].get('request_latency', 0)
            
            # Get best solutions from Pareto front
            pareto_solutions = study_data.get('pareto_front', [])
            if pareto_solutions:
                best_throughput = max(sol['values'][0] for sol in pareto_solutions)
                best_latency = min(sol['values'][1] for sol in pareto_solutions)
                
                fig.add_trace(
                    go.Bar(
                        x=['Baseline Throughput', 'Best Throughput', 'Baseline Latency', 'Best Latency'],
                        y=[baseline_throughput, best_throughput, baseline_latency, best_latency],
                        marker_color=[style['baseline_color'], style['improvement_color']] * 2,
                        name='Metrics Comparison'
                    ),
                    row=1, col=1
                )
    
    # 3. Improvement over time
    trial_numbers = [trial.get('number', i) for i, trial in enumerate(trials)]
    if study_data['type'] == 'single_objective':
        # Running best improvement
        running_best = []
        current_best = float('-inf')
        for trial in trials:
            value = trial.get('value', 0)
            if value > current_best:
                current_best = value
            improvement = ((current_best - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0
            running_best.append(improvement)
        
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=running_best,
                mode='lines+markers',
                line=dict(color=style['improvement_color'], width=2),
                marker=dict(size=4),
                name='Running Best Improvement'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text="Baseline Comparison Analysis",
        showlegend=True,
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        height=800
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Metrics", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Improvement (%)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Trial Number", row=2, col=2)
    fig.update_yaxes(title_text="Improvement (%)", row=2, col=2)
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "baseline_comparison.html")
    fig.write_html(file_path)
    
    return file_path


def create_parameter_analysis(study_data: Dict, output_dir: str) -> str:
    """Create comprehensive parameter analysis visualization."""
    style = get_visualization_style()
    trials = study_data.get('completed_trials', [])
    
    if not trials:
        return ""
    
    # Extract parameter data
    all_params = {}
    for trial in trials:
        params = trial.get('params', {})
        for param_name, param_value in params.items():
            if param_name not in all_params:
                all_params[param_name] = []
            all_params[param_name].append(param_value)
    
    if not all_params:
        return ""
    
    param_names = list(all_params.keys())
    n_params = len(param_names)
    
    # Create subplots
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"{param}" for param in param_names],
        vertical_spacing=0.1
    )
    
    for i, param_name in enumerate(param_names):
        row = i // cols + 1
        col = i % cols + 1
        
        param_values = all_params[param_name]
        
        # Determine if parameter is numeric or categorical
        try:
            numeric_values = [float(v) for v in param_values]
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False
        
        if is_numeric and len(set(numeric_values)) > 10:
            # Histogram for continuous parameters
            fig.add_trace(
                go.Histogram(
                    x=numeric_values,
                    nbinsx=min(20, len(set(numeric_values))),
                    marker_color=style['color_palette'][i % len(style['color_palette'])],
                    opacity=0.7,
                    name=param_name,
                    showlegend=False
                ),
                row=row, col=col
            )
        else:
            # Bar chart for categorical or discrete parameters
            from collections import Counter
            value_counts = Counter(param_values)
            
            fig.add_trace(
                go.Bar(
                    x=list(value_counts.keys()),
                    y=list(value_counts.values()),
                    marker_color=style['color_palette'][i % len(style['color_palette'])],
                    name=param_name,
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text="Parameter Distribution Analysis",
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        height=300 * rows
    )
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "parameter_analysis.html")
    fig.write_html(file_path)
    
    return file_path


def create_trial_diagnostics(study_data: Dict, output_dir: str) -> str:
    """Create trial diagnostics and study health visualization."""
    style = get_visualization_style()
    all_trials = study_data.get('all_trials', [])
    completed_trials = study_data.get('completed_trials', [])
    
    if not all_trials:
        return ""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Trial Success Rate Over Time',
            'Trial State Distribution',
            'Trial Duration Analysis',
            'Parameter Space Coverage'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "pie"}],
            [{"type": "histogram"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Success rate over time
    trial_numbers = []
    success_rates = []
    window_size = max(10, len(all_trials) // 20)
    
    for i in range(window_size, len(all_trials) + 1):
        window_trials = all_trials[max(0, i-window_size):i]
        successful = sum(1 for t in window_trials if t.get('state') == 'COMPLETE')
        success_rate = successful / len(window_trials) * 100
        trial_numbers.append(i)
        success_rates.append(success_rate)
    
    if trial_numbers:
        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=success_rates,
                mode='lines+markers',
                line=dict(color=style['improvement_color'], width=2),
                marker=dict(size=4),
                name='Success Rate'
            ),
            row=1, col=1
        )
    
    # 2. Trial state distribution
    from collections import Counter
    state_counts = Counter(trial.get('state', 'UNKNOWN') for trial in all_trials)
    
    fig.add_trace(
        go.Pie(
            labels=list(state_counts.keys()),
            values=list(state_counts.values()),
            marker_colors=style['color_palette'],
            name='Trial States'
        ),
        row=1, col=2
    )
    
    # 3. Trial duration analysis (if available)
    durations = []
    for trial in completed_trials:
        duration = trial.get('duration', None)
        if duration is not None:
            durations.append(duration)
    
    if durations:
        fig.add_trace(
            go.Histogram(
                x=durations,
                nbinsx=20,
                marker_color=style['improvement_color'],
                opacity=0.7,
                name='Duration'
            ),
            row=2, col=1
        )
    
    # 4. Parameter space coverage (simplified)
    if completed_trials and len(completed_trials) > 1:
        # Use first two numeric parameters for coverage visualization
        numeric_params = []
        for trial in completed_trials[:5]:  # Sample a few trials
            params = trial.get('params', {})
            for param_name, param_value in params.items():
                try:
                    float(param_value)
                    if param_name not in [p[0] for p in numeric_params]:
                        numeric_params.append((param_name, []))
                except (ValueError, TypeError):
                    continue
                if len(numeric_params) >= 2:
                    break
            if len(numeric_params) >= 2:
                break
        
        if len(numeric_params) >= 2:
            param1_name, param2_name = numeric_params[0][0], numeric_params[1][0]
            param1_values = [trial['params'].get(param1_name) for trial in completed_trials 
                           if param1_name in trial.get('params', {})]
            param2_values = [trial['params'].get(param2_name) for trial in completed_trials 
                           if param2_name in trial.get('params', {})]
            
            # Convert to numeric
            try:
                param1_numeric = [float(v) for v in param1_values if v is not None]
                param2_numeric = [float(v) for v in param2_values if v is not None]
                
                if len(param1_numeric) == len(param2_numeric) and param1_numeric:
                    fig.add_trace(
                        go.Scatter(
                            x=param1_numeric,
                            y=param2_numeric,
                            mode='markers',
                            marker=dict(
                                color=style['improvement_color'],
                                size=6,
                                opacity=0.6
                            ),
                            name='Parameter Coverage'
                        ),
                        row=2, col=2
                    )
                    
                    fig.update_xaxes(title_text=param1_name, row=2, col=2)
                    fig.update_yaxes(title_text=param2_name, row=2, col=2)
            except (ValueError, TypeError):
                pass
    
    # Update layout
    fig.update_layout(
        title_text="Study Diagnostics and Health Analysis",
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Trial Number", row=1, col=1)
    fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
    fig.update_xaxes(title_text="Duration (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "trial_diagnostics.html")
    fig.write_html(file_path)
    
    return file_path


def format_hover_text(trial_data: Dict) -> str:
    """Format hover text for trial data points."""
    hover_text = f"<b>Trial {trial_data.get('number', 'Unknown')}</b><br>"
    
    # Add main metrics
    if 'value' in trial_data:
        hover_text += f"Objective Value: {trial_data['value']:.4f}<br>"
    elif 'values' in trial_data:
        values = trial_data['values']
        if len(values) >= 2:
            hover_text += f"Throughput: {values[0]:.2f} tokens/s<br>"
            hover_text += f"Latency: {values[1]:.2f} ms<br>"
    
    # Add key parameters (limit to top 3)
    params = trial_data.get('params', {})
    if params:
        hover_text += "<br><b>Key Parameters:</b><br>"
        for i, (param, value) in enumerate(list(params.items())[:3]):
            hover_text += f"{param}: {value}<br>"
        if len(params) > 3:
            hover_text += f"... and {len(params) - 3} more"
    
    return hover_text
