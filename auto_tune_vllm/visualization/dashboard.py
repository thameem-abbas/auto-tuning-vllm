"""
Unified dashboard creation for optimization studies.

This module creates comprehensive dashboards that combine all relevant visualizations
based on the study type and available data.
"""

import os
from typing import Dict, List, Optional, Any
from .single_objective import create_single_objective_dashboard
from .multi_objective import create_multi_objective_dashboard
from .common import create_baseline_comparison, create_parameter_analysis, create_trial_diagnostics


def create_study_dashboard(study_data: Dict, output_dir: str) -> Dict[str, List[str]]:
    """
    Create comprehensive study dashboard with all relevant visualizations.
    
    Args:
        study_data: Dictionary containing study results and metadata
        output_dir: Directory to save visualization files
        
    Returns:
        Dictionary with categorized list of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {
        'common': [],
        'specific': [],
        'summary': []
    }
    
    study_type = study_data.get('type', 'unknown')
    
    # Generate common visualizations (for both study types)
    common_files = []
    
    # Baseline comparison (if baseline data available)
    baseline_file = create_baseline_comparison(study_data, output_dir)
    if baseline_file:
        common_files.append(baseline_file)
    
    # Parameter analysis
    param_file = create_parameter_analysis(study_data, output_dir)
    if param_file:
        common_files.append(param_file)
    
    # Trial diagnostics
    diagnostic_file = create_trial_diagnostics(study_data, output_dir)
    if diagnostic_file:
        common_files.append(diagnostic_file)
    
    saved_files['common'] = common_files
    
    # Generate study-type specific visualizations
    if study_type == 'single_objective':
        specific_files = create_single_objective_dashboard(study_data, output_dir)
        saved_files['specific'] = specific_files
        
    elif study_type == 'multi_objective':
        specific_files = create_multi_objective_dashboard(study_data, output_dir)
        saved_files['specific'] = specific_files
    
    # Create summary dashboard (HTML page with links to all visualizations)
    summary_file = create_summary_dashboard(study_data, saved_files, output_dir)
    if summary_file:
        saved_files['summary'] = [summary_file]
    
    return saved_files


def create_summary_dashboard(study_data: Dict, saved_files: Dict[str, List[str]], output_dir: str) -> str:
    """Create HTML summary dashboard with links to all visualizations."""
    
    study_name = study_data.get('study_name', 'Unknown Study')
    study_type = study_data.get('type', 'unknown')
    n_trials = study_data.get('n_trials', 0)
    
    # Build HTML content
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{study_name} - Optimization Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-top: 10px;
        }}
        .stats {{
            display: flex;
            justify-content: space-around;
            margin: 30px 0;
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
        }}
        .stat-item {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .section {{
            margin: 40px 0;
        }}
        .section h2 {{
            color: #2c3e50;
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin-bottom: 20px;
        }}
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .viz-card {{
            background-color: #ffffff;
            border: 2px solid #bdc3c7;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        .viz-card:hover {{
            border-color: #3498db;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(52, 152, 219, 0.2);
        }}
        .viz-card h3 {{
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 10px;
        }}
        .viz-card p {{
            color: #7f8c8d;
            margin-bottom: 20px;
            line-height: 1.5;
        }}
        .viz-card a {{
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }}
        .viz-card a:hover {{
            background-color: #2980b9;
        }}
        .best-result {{
            background-color: #d5f4e6;
            border: 2px solid #27ae60;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .best-result h3 {{
            color: #27ae60;
            margin-top: 0;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{study_name}</h1>
            <div class="subtitle">
                {study_type.replace('_', ' ').title()} Optimization Dashboard
            </div>
        </div>
        
        <div class="stats">
            <div class="stat-item">
                <div class="stat-value">{n_trials}</div>
                <div class="stat-label">Total Trials</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(study_data.get('completed_trials', []))}</div>
                <div class="stat-label">Completed Trials</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{study_type.replace('_', '-').title()}</div>
                <div class="stat-label">Study Type</div>
            </div>
        </div>
"""
    
    # Add best results section
    if study_type == 'single_objective':
        best_value = study_data.get('best_value', 0)
        objective_name = study_data.get('objective', {}).get('metric', 'Objective')
        
        html_content += f"""
        <div class="best-result">
            <h3>🏆 Best Result</h3>
            <p><strong>{objective_name}:</strong> {best_value:.4f}</p>
            <p><strong>Trial:</strong> {study_data.get('best_trial_number', 'Unknown')}</p>
        </div>
"""
    elif study_type == 'multi_objective':
        n_pareto = study_data.get('n_pareto_solutions', 0)
        pareto_front = study_data.get('pareto_front', [])
        
        if pareto_front:
            best_throughput = max(sol['values'][0] for sol in pareto_front)
            best_latency = min(sol['values'][1] for sol in pareto_front)
            
            html_content += f"""
        <div class="best-result">
            <h3>🏆 Pareto Optimal Results</h3>
            <p><strong>Pareto Solutions Found:</strong> {n_pareto}</p>
            <p><strong>Best Throughput:</strong> {best_throughput:.2f} tokens/s</p>
            <p><strong>Best Latency:</strong> {best_latency:.2f} ms</p>
        </div>
"""
    
    # Add visualization sections
    viz_sections = [
        {
            'title': 'Study-Specific Visualizations',
            'files': saved_files.get('specific', []),
            'descriptions': get_specific_descriptions(study_type)
        },
        {
            'title': 'Common Analysis',
            'files': saved_files.get('common', []),
            'descriptions': {
                'baseline_comparison.html': 'Compare optimization results with baseline performance',
                'parameter_analysis.html': 'Analyze parameter distributions and relationships',
                'trial_diagnostics.html': 'Study health metrics and trial success analysis'
            }
        }
    ]
    
    for section in viz_sections:
        if section['files']:
            html_content += f"""
        <div class="section">
            <h2>{section['title']}</h2>
            <div class="viz-grid">
"""
            
            for file_path in section['files']:
                file_name = os.path.basename(file_path)
                title = file_name.replace('.html', '').replace('_', ' ').title()
                description = section['descriptions'].get(file_name, 'Detailed analysis visualization')
                
                html_content += f"""
                <div class="viz-card">
                    <h3>{title}</h3>
                    <p>{description}</p>
                    <a href="{file_name}" target="_blank">View Visualization</a>
                </div>
"""
            
            html_content += """
            </div>
        </div>
"""
    
    # Add footer
    html_content += f"""
        <div class="footer">
            <p>Generated by Auto-tune vLLM Optimization Framework</p>
            <p>Dashboard created on {study_data.get('timestamp', 'Unknown')}</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML file
    file_path = os.path.join(output_dir, "dashboard.html")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return file_path


def get_specific_descriptions(study_type: str) -> Dict[str, str]:
    """Get descriptions for study-type specific visualizations."""
    
    if study_type == 'single_objective':
        return {
            'optimization_history.html': 'Track objective value progression and running best performance',
            'convergence_analysis.html': 'Analyze optimization convergence speed and efficiency',
            'parameter_importance.html': 'Identify which parameters most impact the objective'
        }
    
    elif study_type == 'multi_objective':
        return {
            'pareto_front.html': 'Interactive Pareto front with dominated vs optimal solutions',
            'pareto_evolution.html': 'Animation showing how the Pareto front evolved over trials',
            'trade_off_analysis.html': 'Parallel coordinates analysis of trade-offs between objectives',
            'solution_selector.html': 'Interactive tool for selecting preferred solutions from Pareto front'
        }
    
    return {}


def format_study_data_for_visualization(study_controller) -> Dict[str, Any]:
    """
    Convert StudyController data to format expected by visualization functions.
    
    Args:
        study_controller: StudyController instance with optimization results
        
    Returns:
        Formatted data dictionary for visualization functions
    """
    # Get optimization results
    results = study_controller.get_optimization_results()
    
    # Extract study information
    study = study_controller.study
    all_trials = study.trials
    completed_trials = [t for t in all_trials if hasattr(t, 'state') and str(t.state) == 'COMPLETE']
    
    # Format trial data
    formatted_trials = []
    for trial in completed_trials:
        trial_data = {
            'number': trial.number,
            'state': str(trial.state),
            'params': trial.params,
        }
        
        # Add objective values
        if hasattr(trial, 'value') and trial.value is not None:
            trial_data['value'] = trial.value
        elif hasattr(trial, 'values') and trial.values:
            trial_data['values'] = trial.values
        
        # Add user attributes if available
        if hasattr(trial, 'user_attrs') and trial.user_attrs:
            trial_data['user_attrs'] = trial.user_attrs
        
        formatted_trials.append(trial_data)
    
    # Format all trials (including failed ones)
    all_formatted_trials = []
    for trial in all_trials:
        trial_data = {
            'number': trial.number,
            'state': str(trial.state),
            'params': trial.params,
        }
        
        if hasattr(trial, 'value') and trial.value is not None:
            trial_data['value'] = trial.value
        elif hasattr(trial, 'values') and trial.values:
            trial_data['values'] = trial.values
            
        all_formatted_trials.append(trial_data)
    
    # Build final data structure
    formatted_data = {
        'type': results['type'],
        'study_name': study.study_name,
        'n_trials': len(all_trials),
        'completed_trials': formatted_trials,
        'all_trials': all_formatted_trials,
        'approach': results.get('approach'),
        'timestamp': str(study_controller.study._storage.get_study_system_attrs(study._study_id).get('created_at', 'Unknown'))
    }
    
    # Add study-type specific data
    if results['type'] == 'single_objective':
        formatted_data.update({
            'best_value': results.get('best_value'),
            'best_params': results.get('best_params'),
            'best_trial_number': results.get('best_trial_number'),
            'objective': results.get('objective')
        })
    
    elif results['type'] == 'multi_objective':
        formatted_data.update({
            'objectives': results.get('objectives', []),
            'n_pareto_solutions': results.get('n_pareto_solutions'),
            'pareto_front': results.get('pareto_front', [])
        })
    
    # Add baseline metrics if available (this would need to be implemented based on your baseline storage)
    # formatted_data['baseline_metrics'] = load_baseline_metrics_from_controller(study_controller)
    
    return formatted_data
