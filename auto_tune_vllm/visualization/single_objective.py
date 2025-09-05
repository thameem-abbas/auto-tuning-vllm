"""
Single-objective optimization visualizations.

This module provides specialized visualizations for single-objective optimization studies,
including optimization history, convergence analysis, and parameter importance.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from .common import get_visualization_style, format_hover_text


def create_optimization_history(study_data: Dict, output_dir: str) -> str:
    """Create optimization history visualization showing objective value over trials."""
    style = get_visualization_style()
    trials = study_data.get('completed_trials', [])
    baseline_data = study_data.get('baseline_metrics', {})
    
    if not trials:
        return ""
    
    # Extract data
    trial_numbers = [trial.get('number', i) for i, trial in enumerate(trials)]
    objective_values = [trial.get('value', 0) for trial in trials]
    
    # Create main plot
    fig = go.Figure()
    
    # Add optimization history
    hover_texts = [format_hover_text(trial) for trial in trials]
    
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=objective_values,
        mode='lines+markers',
        name='Trial Values',
        line=dict(color=style['improvement_color'], width=2),
        marker=dict(size=6, opacity=0.7),
        hovertemplate='%{text}<extra></extra>',
        text=hover_texts
    ))
    
    # Add running best line
    running_best = []
    current_best = float('-inf') if study_data.get('objective', {}).get('direction') == 'maximize' else float('inf')
    is_maximize = study_data.get('objective', {}).get('direction') == 'maximize'
    
    for value in objective_values:
        if is_maximize:
            current_best = max(current_best, value)
        else:
            current_best = min(current_best, value)
        running_best.append(current_best)
    
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=running_best,
        mode='lines',
        name='Running Best',
        line=dict(color=style['pareto_color'], width=3, dash='dash'),
        hovertemplate='Running Best: %{y:.4f}<extra></extra>'
    ))
    
    # Add baseline if available
    if baseline_data:
        baseline_value = list(baseline_data.values())[0].get('output_tokens_per_second', 0)
        fig.add_hline(
            y=baseline_value,
            line_dash="dot",
            line_color=style['baseline_color'],
            line_width=2,
            annotation_text=f"Baseline: {baseline_value:.2f}",
            annotation_position="top right"
        )
    
    # Update layout
    fig.update_layout(
        title="Optimization History",
        xaxis_title="Trial Number",
        yaxis_title=f"Objective Value ({study_data.get('objective', {}).get('metric', 'Unknown')})",
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        hovermode='x unified',
        height=500
    )
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "optimization_history.html")
    fig.write_html(file_path)
    
    return file_path


def create_convergence_analysis(study_data: Dict, output_dir: str) -> str:
    """Create convergence analysis showing optimization progress and efficiency."""
    style = get_visualization_style()
    trials = study_data.get('completed_trials', [])
    
    if not trials or len(trials) < 5:
        return ""
    
    # Extract data
    trial_numbers = [trial.get('number', i) for i, trial in enumerate(trials)]
    objective_values = [trial.get('value', 0) for trial in trials]
    is_maximize = study_data.get('objective', {}).get('direction') == 'maximize'
    
    # Calculate convergence metrics
    running_best = []
    improvements = []
    current_best = float('-inf') if is_maximize else float('inf')
    
    for i, value in enumerate(objective_values):
        if is_maximize:
            new_best = max(current_best, value)
            improvement = max(0, value - current_best) if current_best != float('-inf') else 0
        else:
            new_best = min(current_best, value)
            improvement = max(0, current_best - value) if current_best != float('inf') else 0
        
        current_best = new_best
        running_best.append(current_best)
        improvements.append(improvement)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Convergence Progress',
            'Improvement Rate',
            'Performance Distribution',
            'Convergence Speed'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "histogram"}, {"type": "scatter"}]
        ]
    )
    
    # 1. Convergence progress
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=running_best,
        mode='lines+markers',
        name='Running Best',
        line=dict(color=style['pareto_color'], width=2),
        marker=dict(size=4)
    ), row=1, col=1)
    
    # 2. Improvement rate (improvements per trial window)
    window_size = max(5, len(trials) // 10)
    windowed_improvements = []
    windowed_trials = []
    
    for i in range(window_size, len(improvements) + 1):
        window_improvements = sum(improvements[max(0, i-window_size):i])
        windowed_improvements.append(window_improvements)
        windowed_trials.append(i)
    
    fig.add_trace(go.Bar(
        x=windowed_trials,
        y=windowed_improvements,
        name='Improvements',
        marker_color=style['improvement_color'],
        opacity=0.7
    ), row=1, col=2)
    
    # 3. Performance distribution
    fig.add_trace(go.Histogram(
        x=objective_values,
        nbinsx=min(20, len(set(objective_values))),
        name='Value Distribution',
        marker_color=style['improvement_color'],
        opacity=0.7
    ), row=2, col=1)
    
    # 4. Convergence speed (time to reach % of final best)
    final_best = running_best[-1]
    convergence_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    convergence_trials = []
    
    for threshold in convergence_thresholds:
        target_value = final_best * threshold if is_maximize else final_best + (1 - threshold) * abs(final_best)
        
        converged_trial = len(trials)  # Default to last trial
        for i, best_val in enumerate(running_best):
            if (is_maximize and best_val >= target_value) or (not is_maximize and best_val <= target_value):
                converged_trial = i + 1
                break
        convergence_trials.append(converged_trial)
    
    fig.add_trace(go.Scatter(
        x=[f"{int(t*100)}%" for t in convergence_thresholds],
        y=convergence_trials,
        mode='lines+markers',
        name='Convergence Speed',
        line=dict(color=style['baseline_color'], width=2),
        marker=dict(size=6)
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Convergence Analysis",
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Trial Number", row=1, col=1)
    fig.update_yaxes(title_text="Best Value", row=1, col=1)
    fig.update_xaxes(title_text="Trial Window", row=1, col=2)
    fig.update_yaxes(title_text="Total Improvement", row=1, col=2)
    fig.update_xaxes(title_text="Objective Value", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="% of Final Best", row=2, col=2)
    fig.update_yaxes(title_text="Trials to Converge", row=2, col=2)
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "convergence_analysis.html")
    fig.write_html(file_path)
    
    return file_path


def create_parameter_importance(study_data: Dict, output_dir: str) -> str:
    """Create parameter importance analysis using correlation and variance."""
    style = get_visualization_style()
    trials = study_data.get('completed_trials', [])
    
    if not trials or len(trials) < 10:
        return ""
    
    # Extract parameter data and objective values
    param_data = {}
    objective_values = []
    
    for trial in trials:
        params = trial.get('params', {})
        value = trial.get('value', 0)
        objective_values.append(value)
        
        for param_name, param_value in params.items():
            if param_name not in param_data:
                param_data[param_name] = []
            param_data[param_name].append(param_value)
    
    if not param_data:
        return ""
    
    # Calculate correlations and importance scores
    param_importance = {}
    param_correlations = {}
    
    for param_name, param_values in param_data.items():
        try:
            # Convert to numeric if possible
            numeric_values = []
            numeric_objectives = []
            
            for i, val in enumerate(param_values):
                try:
                    numeric_val = float(val)
                    numeric_values.append(numeric_val)
                    numeric_objectives.append(objective_values[i])
                except (ValueError, TypeError):
                    continue
            
            if len(numeric_values) > 5 and len(set(numeric_values)) > 1:
                # Calculate correlation
                import statistics
                mean_param = statistics.mean(numeric_values)
                mean_obj = statistics.mean(numeric_objectives)
                
                numerator = sum((p - mean_param) * (o - mean_obj) 
                              for p, o in zip(numeric_values, numeric_objectives))
                denom_param = sum((p - mean_param) ** 2 for p in numeric_values) ** 0.5
                denom_obj = sum((o - mean_obj) ** 2 for o in numeric_objectives) ** 0.5
                
                if denom_param > 0 and denom_obj > 0:
                    correlation = numerator / (denom_param * denom_obj)
                    param_correlations[param_name] = correlation
                    param_importance[param_name] = abs(correlation)
                    
        except Exception:
            continue
    
    if not param_importance:
        return ""
    
    # Create visualizations
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Parameter Importance (Absolute Correlation)',
            'Parameter Correlations',
            'Top Parameter vs Objective',
            'Parameter Interactions'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}]
        ]
    )
    
    # Sort parameters by importance
    sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
    param_names = [p[0] for p in sorted_params]
    importance_scores = [p[1] for p in sorted_params]
    
    # 1. Parameter importance
    fig.add_trace(go.Bar(
        x=param_names,
        y=importance_scores,
        marker_color=style['improvement_color'],
        name='Importance'
    ), row=1, col=1)
    
    # 2. Parameter correlations (showing positive/negative)
    correlation_values = [param_correlations.get(name, 0) for name in param_names]
    colors = [style['pareto_color'] if c >= 0 else style['baseline_color'] for c in correlation_values]
    
    fig.add_trace(go.Bar(
        x=param_names,
        y=correlation_values,
        marker_color=colors,
        name='Correlation'
    ), row=1, col=2)
    
    # 3. Most important parameter vs objective
    if sorted_params:
        most_important_param = sorted_params[0][0]
        param_values_numeric = []
        corresponding_objectives = []
        
        for i, trial in enumerate(trials):
            param_val = trial.get('params', {}).get(most_important_param)
            if param_val is not None:
                try:
                    numeric_val = float(param_val)
                    param_values_numeric.append(numeric_val)
                    corresponding_objectives.append(objective_values[i])
                except (ValueError, TypeError):
                    continue
        
        if param_values_numeric:
            fig.add_trace(go.Scatter(
                x=param_values_numeric,
                y=corresponding_objectives,
                mode='markers',
                marker=dict(
                    color=style['improvement_color'],
                    size=6,
                    opacity=0.7
                ),
                name=f'{most_important_param} vs Objective'
            ), row=2, col=1)
    
    # 4. Parameter interactions (top 2 parameters if available)
    if len(sorted_params) >= 2:
        param1_name = sorted_params[0][0]
        param2_name = sorted_params[1][0]
        
        param1_vals = []
        param2_vals = []
        obj_vals = []
        
        for i, trial in enumerate(trials):
            params = trial.get('params', {})
            p1_val = params.get(param1_name)
            p2_val = params.get(param2_name)
            
            if p1_val is not None and p2_val is not None:
                try:
                    p1_numeric = float(p1_val)
                    p2_numeric = float(p2_val)
                    param1_vals.append(p1_numeric)
                    param2_vals.append(p2_numeric)
                    obj_vals.append(objective_values[i])
                except (ValueError, TypeError):
                    continue
        
        if param1_vals and param2_vals:
            fig.add_trace(go.Scatter(
                x=param1_vals,
                y=param2_vals,
                mode='markers',
                marker=dict(
                    color=obj_vals,
                    colorscale='Viridis',
                    size=8,
                    opacity=0.7,
                    colorbar=dict(title="Objective Value")
                ),
                name='Parameter Interaction'
            ), row=2, col=2)
            
            fig.update_xaxes(title_text=param1_name, row=2, col=2)
            fig.update_yaxes(title_text=param2_name, row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title_text="Parameter Importance Analysis",
        template=style['template'],
        font=dict(family=style['font_family'], size=style['legend_size']),
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Parameters", row=1, col=1)
    fig.update_yaxes(title_text="Importance", row=1, col=1)
    fig.update_xaxes(title_text="Parameters", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=2)
    
    if sorted_params:
        fig.update_xaxes(title_text=sorted_params[0][0], row=2, col=1)
        fig.update_yaxes(title_text="Objective Value", row=2, col=1)
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    
    # Save file
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "parameter_importance.html")
    fig.write_html(file_path)
    
    return file_path


def create_single_objective_dashboard(study_data: Dict, output_dir: str) -> List[str]:
    """Create comprehensive single-objective dashboard with all visualizations."""
    saved_files = []
    
    # Generate individual visualizations
    files = [
        create_optimization_history(study_data, output_dir),
        create_convergence_analysis(study_data, output_dir),
        create_parameter_importance(study_data, output_dir)
    ]
    
    # Filter out empty results
    saved_files.extend([f for f in files if f])
    
    return saved_files
