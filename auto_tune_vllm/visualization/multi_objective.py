"""
Multi-objective optimization visualizations.

This module provides specialized visualizations for multi-objective optimization studies,
including Pareto front analysis, trade-off visualization, and solution selection tools.
"""

import os
import math
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from .common import get_visualization_style, format_hover_text


def create_pareto_front_plot(study_data: Dict, output_dir: str) -> str:
    """Create interactive Pareto front scatter plot with dominance analysis."""
    style = get_visualization_style()
    trials = study_data.get('completed_trials', [])
    pareto_solutions = study_data.get('pareto_front', [])
    baseline_data = study_data.get('baseline_metrics', {})
    
    if not trials or not pareto_solutions:
        return ""
    
    # Extract objective values
    all_obj1 = [trial.get('values', [0, 0])[0] for trial in trials if trial.get('values')]
    all_obj2 = [trial.get('values', [0, 0])[1] for trial in trials if trial.get('values')]
    
    pareto_obj1 = [sol.get('values', [0, 0])[0] for sol in pareto_solutions]
    pareto_obj2 = [sol.get('values', [0, 0])[1] for sol in pareto_solutions]
    
    if not all_obj1 or not all_obj2:
        return ""
    
    # Create main plot
    fig = go.Figure()
    
    # Add all solutions (dominated) as background
    dominated_obj1 = []
    dominated_obj2 = []
    dominated_trials = []
    
    for trial in trials:
        values = trial.get('values', [])
        if len(values) >= 2:
            trial_num = trial.get('number', 0)
            # Check if this trial is in Pareto front
            is_pareto = any(sol.get('trial', -1) == trial_num for sol in pareto_solutions)
            if not is_pareto:
                dominated_obj1.append(values[0])
                dominated_obj2.append(values[1])
                dominated_trials.append(trial)
    
    if dominated_obj1:
        dominated_hover = [format_hover_text(trial) for trial in dominated_trials]
        fig.add_trace(go.Scatter(
            x=dominated_obj1,
            y=dominated_obj2,
            mode='markers',
            name='Dominated Solutions',
            marker=dict(
                color=style['dominated_color'],
                size=6,
                opacity=0.5
            ),
            hovertemplate='%{text}<extra></extra>',
            text=dominated_hover
        ))
    
    # Add Pareto-optimal solutions
    pareto_trials = [next((trial for trial in trials if trial.get('number') == sol.get('trial')), sol) 
                    for sol in pareto_solutions]
    pareto_hover = [format_hover_text(trial) for trial in pareto_trials]
    
    fig.add_trace(go.Scatter(
        x=pareto_obj1,
        y=pareto_obj2,
        mode='markers',
        name='Pareto-Optimal Solutions',
        marker=dict(
            color=style['pareto_color'],
            size=10,
            opacity=0.8,
            line=dict(width=2, color='darkgreen')
        ),
        hovertemplate='%{text}<extra></extra>',
        text=pareto_hover
    ))
    
    # Add Pareto front line (approximate)
    if len(pareto_obj1) > 1:
        # Sort Pareto points by first objective for line drawing
        pareto_points = list(zip(pareto_obj1, pareto_obj2))
        pareto_points.sort(key=lambda x: x[0], reverse=True)  # Descending for throughput
        
        sorted_obj1, sorted_obj2 = zip(*pareto_points)
        
        fig.add_trace(go.Scatter(
            x=sorted_obj1,
            y=sorted_obj2,
            mode='lines',
            name='Pareto Front',
            line=dict(
                color=style['pareto_color'],
                width=2,
                dash='dash'
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add baseline points if available
    if baseline_data:
        baseline_obj1 = []
        baseline_obj2 = []
        baseline_labels = []
        
        for concurrency, metrics in baseline_data.items():
            throughput = metrics.get('output_tokens_per_second', 0)
            latency = metrics.get('request_latency', 0)
            baseline_obj1.append(throughput)
            baseline_obj2.append(latency)
            baseline_labels.append(f'Baseline C={concurrency}')
        
        if baseline_obj1:
            fig.add_trace(go.Scatter(
                x=baseline_obj1,
                y=baseline_obj2,
                mode='markers',
                name='Baseline',
                marker=dict(
                    color=style['baseline_color'],
                    size=12,
                    symbol='diamond',
                    line=dict(width=2, color='darkred')
                ),
                text=baseline_labels,
                hovertemplate='%{text}<br>Throughput: %{x:.2f}<br>Latency: %{y:.2f}<extra></extra>'
            ))
    
    # Calculate ideal and nadir points for reference
    if all_obj1 and all_obj2:
        ideal_obj1 = max(all_obj1)  # Best throughput
        ideal_obj2 = min(all_obj2)  # Best latency
        nadir_obj1 = min(all_obj1)  # Worst throughput
        nadir_obj2 = max(all_obj2)  # Worst latency
        
        # Add ideal point
        fig.add_trace(go.Scatter(
            x=[ideal_obj1],
            y=[ideal_obj2],
            mode='markers',
            name='Ideal Point',
            marker=dict(
                color='gold',
                size=15,
                symbol='star',
                line=dict(width=2, color='orange')
            ),
            hovertemplate='Ideal Point<br>Best Throughput: %{x:.2f}<br>Best Latency: %{y:.2f}<extra></extra>'
        ))
    
    # Update layout
    objectives = study_data.get('objectives', [{}, {}])
    obj1_name = objectives[0].get('metric', 'Objective 1') if len(objectives) > 0 else 'Throughput'
    obj2_name = objectives[1].get('metric', 'Objective 2') if len(objectives) > 1 else 'Latency'
    
    fig.update_layout(
        title="Pareto Front Analysis",
        xaxis_title=f"{obj1_name} (tokens/s)" if 'throughput' in obj1_name.lower() else obj1_name,
        yaxis_title=f"{obj2_name} (ms)" if 'latency' in obj2_name.lower() else obj2_name,
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        hovermode='closest',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "pareto_front.html")
    fig.write_html(file_path)
    
    return file_path


def create_pareto_evolution(study_data: Dict, output_dir: str) -> str:
    """Create animation showing evolution of Pareto front over trials."""
    style = get_visualization_style()
    trials = study_data.get('completed_trials', [])
    
    if not trials or len(trials) < 10:
        return ""
    
    # Group trials by trial number and build Pareto front evolution
    trial_numbers = sorted([trial.get('number', i) for i, trial in enumerate(trials)])
    
    frames = []
    evolution_data = []
    
    # Calculate Pareto front at different time points
    checkpoints = [i for i in range(10, len(trials) + 1, max(1, len(trials) // 20))]
    if checkpoints[-1] != len(trials):
        checkpoints.append(len(trials))
    
    for checkpoint in checkpoints:
        # Get trials up to this checkpoint
        current_trials = trials[:checkpoint]
        
        # Extract objectives
        obj1_vals = []
        obj2_vals = []
        trial_data = []
        
        for trial in current_trials:
            values = trial.get('values', [])
            if len(values) >= 2:
                obj1_vals.append(values[0])
                obj2_vals.append(values[1])
                trial_data.append(trial)
        
        if obj1_vals:
            # Find Pareto front for current trials
            pareto_indices = find_pareto_front(obj1_vals, obj2_vals)
            pareto_obj1 = [obj1_vals[i] for i in pareto_indices]
            pareto_obj2 = [obj2_vals[i] for i in pareto_indices]
            
            evolution_data.append({
                'checkpoint': checkpoint,
                'all_obj1': obj1_vals,
                'all_obj2': obj2_vals,
                'pareto_obj1': pareto_obj1,
                'pareto_obj2': pareto_obj2,
                'trials': trial_data
            })
    
    if not evolution_data:
        return ""
    
    # Create initial plot
    first_data = evolution_data[0]
    
    fig = go.Figure()
    
    # Add all solutions
    fig.add_trace(go.Scatter(
        x=first_data['all_obj1'],
        y=first_data['all_obj2'],
        mode='markers',
        name='All Solutions',
        marker=dict(color=style['dominated_color'], size=6, opacity=0.5)
    ))
    
    # Add Pareto front
    fig.add_trace(go.Scatter(
        x=first_data['pareto_obj1'],
        y=first_data['pareto_obj2'],
        mode='markers+lines',
        name='Pareto Front',
        marker=dict(color=style['pareto_color'], size=8),
        line=dict(color=style['pareto_color'], width=2, dash='dash')
    ))
    
    # Create frames for animation
    for data in evolution_data:
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=data['all_obj1'],
                    y=data['all_obj2'],
                    mode='markers',
                    marker=dict(color=style['dominated_color'], size=6, opacity=0.5)
                ),
                go.Scatter(
                    x=data['pareto_obj1'],
                    y=data['pareto_obj2'],
                    mode='markers+lines',
                    marker=dict(color=style['pareto_color'], size=8),
                    line=dict(color=style['pareto_color'], width=2, dash='dash')
                )
            ],
            name=str(data['checkpoint'])
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title="Pareto Front Evolution",
        xaxis_title="Throughput (tokens/s)",
        yaxis_title="Latency (ms)",
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        height=600,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 500, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        sliders=[{
            'steps': [
                {
                    'args': [[str(data['checkpoint'])], {
                        'frame': {'duration': 300, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 300}
                    }],
                    'label': f"Trial {data['checkpoint']}",
                    'method': 'animate'
                }
                for data in evolution_data
            ],
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 20},
                'prefix': 'Trial: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'}
        }]
    )
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "pareto_evolution.html")
    fig.write_html(file_path)
    
    return file_path


def create_trade_off_analysis(study_data: Dict, output_dir: str) -> str:
    """Create trade-off analysis with parallel coordinates and solution ranking."""
    style = get_visualization_style()
    pareto_solutions = study_data.get('pareto_front', [])
    
    if not pareto_solutions or len(pareto_solutions) < 3:
        return ""
    
    # Prepare data for parallel coordinates
    dimensions = []
    
    # Add objective dimensions
    objectives = study_data.get('objectives', [{}, {}])
    if len(objectives) >= 2:
        obj1_name = objectives[0].get('metric', 'Objective 1')
        obj2_name = objectives[1].get('metric', 'Objective 2')
        
        obj1_values = [sol.get('values', [0, 0])[0] for sol in pareto_solutions]
        obj2_values = [sol.get('values', [0, 0])[1] for sol in pareto_solutions]
        
        dimensions.append(dict(
            range=[min(obj1_values), max(obj1_values)],
            label=obj1_name,
            values=obj1_values
        ))
        
        dimensions.append(dict(
            range=[min(obj2_values), max(obj2_values)],
            label=obj2_name,
            values=obj2_values
        ))
    
    # Add key parameter dimensions
    param_data = {}
    for sol in pareto_solutions:
        params = sol.get('params', {})
        for param_name, param_value in params.items():
            if param_name not in param_data:
                param_data[param_name] = []
            param_data[param_name].append(param_value)
    
    # Select most varying parameters (up to 5 additional dimensions)
    param_variations = {}
    for param_name, values in param_data.items():
        try:
            numeric_values = [float(v) for v in values]
            if len(set(numeric_values)) > 1:  # Has variation
                param_variations[param_name] = {
                    'values': numeric_values,
                    'range': [min(numeric_values), max(numeric_values)],
                    'variance': sum((v - sum(numeric_values)/len(numeric_values))**2 for v in numeric_values)
                }
        except (ValueError, TypeError):
            continue
    
    # Sort by variance and take top 5
    top_params = sorted(param_variations.items(), key=lambda x: x[1]['variance'], reverse=True)[:5]
    
    for param_name, param_info in top_params:
        dimensions.append(dict(
            range=param_info['range'],
            label=param_name,
            values=param_info['values']
        ))
    
    if len(dimensions) < 3:
        return ""
    
    # Create parallel coordinates plot
    fig = go.Figure(data=go.Parcoords(
        line=dict(
            color=obj1_values if len(obj1_values) == len(pareto_solutions) else list(range(len(pareto_solutions))),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=obj1_name if len(obj1_values) == len(pareto_solutions) else "Solution Index")
        ),
        dimensions=dimensions
    ))
    
    fig.update_layout(
        title="Pareto Solutions Trade-off Analysis",
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        height=600
    )
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "trade_off_analysis.html")
    fig.write_html(file_path)
    
    return file_path


def create_solution_selector(study_data: Dict, output_dir: str) -> str:
    """Create interactive solution selection dashboard."""
    style = get_visualization_style()
    pareto_solutions = study_data.get('pareto_front', [])
    
    if not pareto_solutions:
        return ""
    
    # Create subplots for different selection methods
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Solutions by Throughput Priority',
            'Solutions by Latency Priority',
            'Balanced Solutions (Knee Points)',
            'Solution Comparison Table'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "table"}]
        ]
    )
    
    # Extract data
    obj1_values = [sol.get('values', [0, 0])[0] for sol in pareto_solutions]
    obj2_values = [sol.get('values', [0, 0])[1] for sol in pareto_solutions]
    trial_numbers = [sol.get('trial', i) for i, sol in enumerate(pareto_solutions)]
    
    # 1. Solutions ranked by throughput (first objective)
    throughput_ranking = sorted(enumerate(obj1_values), key=lambda x: x[1], reverse=True)
    top_throughput_indices = [idx for idx, _ in throughput_ranking[:5]]
    
    fig.add_trace(go.Scatter(
        x=[obj1_values[i] for i in top_throughput_indices],
        y=[obj2_values[i] for i in top_throughput_indices],
        mode='markers',
        marker=dict(
            color=style['improvement_color'],
            size=[15, 12, 10, 8, 6],  # Larger for better rank
            line=dict(width=2, color='darkblue')
        ),
        name='High Throughput',
        text=[f"Trial {trial_numbers[i]}" for i in top_throughput_indices],
        hovertemplate='%{text}<br>Throughput: %{x:.2f}<br>Latency: %{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # 2. Solutions ranked by latency (second objective - lower is better)
    latency_ranking = sorted(enumerate(obj2_values), key=lambda x: x[1])
    top_latency_indices = [idx for idx, _ in latency_ranking[:5]]
    
    fig.add_trace(go.Scatter(
        x=[obj1_values[i] for i in top_latency_indices],
        y=[obj2_values[i] for i in top_latency_indices],
        mode='markers',
        marker=dict(
            color=style['pareto_color'],
            size=[15, 12, 10, 8, 6],  # Larger for better rank
            line=dict(width=2, color='darkgreen')
        ),
        name='Low Latency',
        text=[f"Trial {trial_numbers[i]}" for i in top_latency_indices],
        hovertemplate='%{text}<br>Throughput: %{x:.2f}<br>Latency: %{y:.2f}<extra></extra>'
    ), row=1, col=2)
    
    # 3. Knee points (balanced solutions)
    knee_indices = find_knee_points(obj1_values, obj2_values)
    
    if knee_indices:
        fig.add_trace(go.Scatter(
            x=[obj1_values[i] for i in knee_indices],
            y=[obj2_values[i] for i in knee_indices],
            mode='markers',
            marker=dict(
                color='gold',
                size=12,
                symbol='star',
                line=dict(width=2, color='orange')
            ),
            name='Balanced',
            text=[f"Trial {trial_numbers[i]}" for i in knee_indices],
            hovertemplate='%{text}<br>Throughput: %{x:.2f}<br>Latency: %{y:.2f}<extra></extra>'
        ), row=2, col=1)
    
    # 4. Solution comparison table
    # Prepare table data for top solutions
    table_solutions = []
    
    # Add top 3 from each category
    all_selected_indices = set(top_throughput_indices[:3] + top_latency_indices[:3])
    if knee_indices:
        all_selected_indices.update(knee_indices[:2])
    
    for idx in sorted(all_selected_indices):
        sol = pareto_solutions[idx]
        values = sol.get('values', [0, 0])
        params = sol.get('params', {})
        
        # Get key parameters (limit to 3)
        key_params = list(params.items())[:3]
        params_str = ", ".join(f"{k}={v}" for k, v in key_params)
        
        table_solutions.append({
            'trial': sol.get('trial', idx),
            'throughput': f"{values[0]:.2f}",
            'latency': f"{values[1]:.2f}",
            'params': params_str
        })
    
    if table_solutions:
        fig.add_trace(go.Table(
            header=dict(
                values=['Trial', 'Throughput', 'Latency', 'Key Parameters'],
                fill_color=style['improvement_color'],
                font_color='white',
                font_size=12
            ),
            cells=dict(
                values=[
                    [sol['trial'] for sol in table_solutions],
                    [sol['throughput'] for sol in table_solutions],
                    [sol['latency'] for sol in table_solutions],
                    [sol['params'] for sol in table_solutions]
                ],
                fill_color='lightgray',
                font_size=10,
                height=30
            )
        ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Interactive Solution Selection",
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        height=800,
        showlegend=True
    )
    
    # Update axes
    for row in [1, 2]:
        for col in [1, 2]:
            if not (row == 2 and col == 2):  # Skip table subplot
                fig.update_xaxes(title_text="Throughput (tokens/s)", row=row, col=col)
                fig.update_yaxes(title_text="Latency (ms)", row=row, col=col)
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "solution_selector.html")
    fig.write_html(file_path)
    
    return file_path


def create_multi_objective_dashboard(study_data: Dict, output_dir: str) -> List[str]:
    """Create comprehensive multi-objective dashboard with all visualizations."""
    saved_files = []
    
    # Generate individual visualizations
    files = [
        create_pareto_front_plot(study_data, output_dir),
        create_pareto_evolution(study_data, output_dir),
        create_trade_off_analysis(study_data, output_dir),
        create_solution_selector(study_data, output_dir)
    ]
    
    # Filter out empty results
    saved_files.extend([f for f in files if f])
    
    return saved_files


# Helper functions

def find_pareto_front(obj1_values: List[float], obj2_values: List[float]) -> List[int]:
    """Find indices of Pareto-optimal solutions (assumes maximizing obj1, minimizing obj2)."""
    n = len(obj1_values)
    pareto_indices = []
    
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i != j:
                # Check if solution j dominates solution i
                if (obj1_values[j] >= obj1_values[i] and obj2_values[j] <= obj2_values[i] and
                    (obj1_values[j] > obj1_values[i] or obj2_values[j] < obj2_values[i])):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_indices.append(i)
    
    return pareto_indices


def find_knee_points(obj1_values: List[float], obj2_values: List[float]) -> List[int]:
    """Find knee points (balanced solutions) on the Pareto front."""
    if len(obj1_values) < 3:
        return []
    
    # Normalize objectives to [0,1] range
    min_obj1, max_obj1 = min(obj1_values), max(obj1_values)
    min_obj2, max_obj2 = min(obj2_values), max(obj2_values)
    
    if max_obj1 == min_obj1 or max_obj2 == min_obj2:
        return []
    
    norm_obj1 = [(x - min_obj1) / (max_obj1 - min_obj1) for x in obj1_values]
    norm_obj2 = [(x - min_obj2) / (max_obj2 - min_obj2) for x in obj2_values]
    
    # For each point, calculate distance to ideal point (1, 0) since we want max obj1, min obj2
    distances = []
    for i in range(len(norm_obj1)):
        # Distance to ideal point (1, 0)
        dist = math.sqrt((1 - norm_obj1[i])**2 + (0 - (1 - norm_obj2[i]))**2)
        distances.append((i, dist))
    
    # Sort by distance and return indices of closest points (knee points)
    distances.sort(key=lambda x: x[1])
    knee_indices = [idx for idx, _ in distances[:min(3, len(distances))]]
    
    return knee_indices
