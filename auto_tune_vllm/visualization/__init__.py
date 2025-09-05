"""
Auto-tune vLLM Visualization Module

This module provides comprehensive visualization capabilities for both single-objective 
and multi-objective optimization studies.

Features:
- Single-objective: Optimization history, convergence analysis, parameter importance
- Multi-objective: Pareto front analysis, trade-off visualization, solution selection
- Common: Baseline comparison, parameter analysis, study diagnostics
"""

from .dashboard import create_study_dashboard
from .single_objective import (
    create_optimization_history,
    create_parameter_importance,
    create_convergence_analysis,
    create_single_objective_dashboard
)
from .multi_objective import (
    create_pareto_front_plot,
    create_pareto_evolution,
    create_trade_off_analysis,
    create_multi_objective_dashboard
)
from .common import (
    create_baseline_comparison,
    create_parameter_analysis,
    create_trial_diagnostics
)

__all__ = [
    # Main dashboard
    'create_study_dashboard',
    
    # Single-objective
    'create_optimization_history',
    'create_parameter_importance', 
    'create_convergence_analysis',
    'create_single_objective_dashboard',
    
    # Multi-objective
    'create_pareto_front_plot',
    'create_pareto_evolution',
    'create_trade_off_analysis',
    'create_multi_objective_dashboard',
    
    # Common
    'create_baseline_comparison',
    'create_parameter_analysis',
    'create_trial_diagnostics',
]
