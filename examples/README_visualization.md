# Visualization Guide for Auto-tune vLLM

This guide explains how to use the comprehensive visualization system for both single-objective and multi-objective optimization studies.

## Overview

The visualization system automatically generates interactive HTML dashboards that help you:

1. **Single-objective studies**: Analyze optimization progress, parameter importance, and convergence
2. **Multi-objective studies**: Explore Pareto fronts, trade-offs, and solution selection
3. **All studies**: Compare with baselines, analyze parameters, and diagnose study health

## Automatic Visualization Generation

Visualizations are automatically generated after optimization completes:

```bash
# Run optimization - visualizations are created automatically
auto-tune-vllm optimize --config examples/study_config_local_exec.yaml
```

After optimization, you'll see output like:
```
📊 Visualizations Generated Successfully!
📋 Main Dashboard: /path/to/study/visualizations/dashboard.html
🎯 Multi-objective Analysis:
  • Pareto Front: /path/to/study/visualizations/pareto_front.html
  • Pareto Evolution: /path/to/study/visualizations/pareto_evolution.html
  • Trade Off Analysis: /path/to/study/visualizations/trade_off_analysis.html
  • Solution Selector: /path/to/study/visualizations/solution_selector.html
📈 General Analysis:
  • Baseline Comparison: /path/to/study/visualizations/baseline_comparison.html
  • Parameter Analysis: /path/to/study/visualizations/parameter_analysis.html
  • Trial Diagnostics: /path/to/study/visualizations/trial_diagnostics.html
```

## Manual Visualization Generation

Generate visualizations for existing studies:

```bash
# Generate visualizations for a specific study
auto-tune-vllm visualize --study-name my-study-name

# Specify custom output directory
auto-tune-vllm visualize --study-name my-study-name --output-dir ./my-viz

# Use with remote database
auto-tune-vllm visualize --study-name my-study --database-url postgresql://user:pass@host:port/db
```

## Single-Objective Visualizations

For single-objective studies (maximizing one metric like throughput):

### 1. Optimization History
- **File**: `optimization_history.html`
- **Purpose**: Track objective value progression over trials
- **Features**: 
  - Line plot of trial values
  - Running best performance
  - Baseline comparison line
  - Interactive hover with trial details and parameters

### 2. Convergence Analysis
- **File**: `convergence_analysis.html`
- **Purpose**: Analyze optimization efficiency and convergence speed
- **Features**:
  - Convergence progress tracking
  - Improvement rate over time
  - Performance distribution histogram
  - Speed to reach convergence thresholds

### 3. Parameter Importance
- **File**: `parameter_importance.html`
- **Purpose**: Identify which parameters most affect the objective
- **Features**:
  - Parameter importance ranking (correlation-based)
  - Parameter vs objective scatter plots
  - Parameter interaction analysis
  - Correlation matrix visualization

## Multi-Objective Visualizations

For multi-objective studies (balancing multiple metrics like throughput vs latency):

### 1. Pareto Front Analysis
- **File**: `pareto_front.html`
- **Purpose**: Interactive visualization of optimal trade-off solutions
- **Features**:
  - 2D scatter plot with throughput vs latency
  - Pareto-optimal points highlighted in green
  - Dominated solutions shown in gray
  - Baseline comparison points
  - Ideal point reference
  - Interactive hover with detailed metrics

### 2. Pareto Evolution
- **File**: `pareto_evolution.html`
- **Purpose**: Animation showing how Pareto front developed over time
- **Features**:
  - Animated progression of optimization
  - Play/pause controls
  - Timeline slider for specific points
  - Evolution of solution quality over trials

### 3. Trade-off Analysis
- **File**: `trade_off_analysis.html`
- **Purpose**: Multi-dimensional analysis of Pareto solutions
- **Features**:
  - Parallel coordinates plot
  - Color-coded by objective values
  - Parameter relationships visualization
  - Trade-off patterns identification

### 4. Solution Selector
- **File**: `solution_selector.html`
- **Purpose**: Interactive tool for selecting preferred solutions
- **Features**:
  - Solutions ranked by throughput priority
  - Solutions ranked by latency priority
  - Balanced solutions (knee points)
  - Comparison table with key parameters
  - Interactive selection interface

## Common Visualizations

Available for both study types:

### 1. Baseline Comparison
- **File**: `baseline_comparison.html`
- **Purpose**: Compare optimization results with baseline performance
- **Features**:
  - Performance vs baseline metrics
  - Improvement distribution analysis
  - Running improvement over time
  - Multiple baseline comparisons

### 2. Parameter Analysis
- **File**: `parameter_analysis.html`
- **Purpose**: Analyze parameter distributions and search space coverage
- **Features**:
  - Parameter value distributions
  - Categorical vs continuous parameter handling
  - Search space exploration analysis
  - Parameter variation patterns

### 3. Trial Diagnostics
- **File**: `trial_diagnostics.html`
- **Purpose**: Study health metrics and optimization diagnostics
- **Features**:
  - Trial success rate over time
  - Trial state distribution (completed/failed/pruned)
  - Trial duration analysis
  - Parameter space coverage visualization

## Main Dashboard

### Unified Interface
- **File**: `dashboard.html`
- **Purpose**: Single entry point with links to all visualizations
- **Features**:
  - Study overview and statistics
  - Best results summary
  - Organized links to all visualizations
  - Responsive design for different screen sizes
  - Modern, professional interface

## Understanding Pareto Optimal Solutions

### What are Pareto Optimal Solutions?
In multi-objective optimization, a solution is Pareto optimal if no other solution exists that improves one objective without worsening another. The set of all Pareto optimal solutions forms the "Pareto front."

### Key Concepts:

1. **Dominated vs Non-dominated**:
   - **Dominated**: A solution that can be improved in at least one objective without worsening others
   - **Non-dominated**: A Pareto optimal solution that cannot be improved without trade-offs

2. **Trade-offs**:
   - Each point on the Pareto front represents a different balance between objectives
   - Moving along the front means trading one objective for another

3. **Solution Selection**:
   - **High throughput**: Choose solutions on the right side of the Pareto front
   - **Low latency**: Choose solutions on the bottom of the Pareto front  
   - **Balanced**: Choose "knee points" that offer good balance between both

### Visualization Features for Pareto Analysis:

1. **Color Coding**:
   - 🟢 Green: Pareto optimal solutions
   - ⚪ Gray: Dominated solutions
   - 🔴 Red: Baseline performance
   - ⭐ Gold: Ideal point (best possible in each objective)

2. **Interactive Selection**:
   - Click points to see detailed parameters
   - Compare multiple solutions side-by-side
   - Export selected solution configurations

3. **Decision Support**:
   - Knee point identification for balanced solutions
   - Ranking by different criteria
   - Parameter sensitivity analysis

## Best Practices

### 1. Start with the Main Dashboard
- Always begin with `dashboard.html` for an overview
- Use it to navigate to specific analysis areas
- Check study statistics and best results summary

### 2. Single-Objective Analysis Workflow
1. Review **optimization history** to understand convergence
2. Check **parameter importance** to identify key factors
3. Use **convergence analysis** to assess optimization efficiency
4. Compare with **baseline** to quantify improvements

### 3. Multi-Objective Analysis Workflow
1. Examine **Pareto front** to understand trade-off space
2. Use **solution selector** to identify preferred solutions
3. Review **trade-off analysis** for parameter relationships
4. Watch **Pareto evolution** to understand optimization progress

### 4. Troubleshooting with Diagnostics
- Use **trial diagnostics** to identify optimization issues
- Check success rates and trial durations
- Analyze parameter space coverage
- Identify patterns in failed trials

## Integration Examples

### Example 1: Single-Objective Throughput Optimization

```yaml
# study_config_single_objective.yaml
optimization:
  approach: "single_objective"
  objective:
    metric: "output_tokens_per_second"
    direction: "maximize"
  n_trials: 100
```

**Generated visualizations**: Optimization history, convergence analysis, parameter importance

### Example 2: Multi-Objective Throughput vs Latency

```yaml
# study_config_multi_objective.yaml
optimization:
  approach: "multi_objective"
  objectives:
    - metric: "output_tokens_per_second"
      direction: "maximize"
    - metric: "request_latency"
      direction: "minimize"
  n_trials: 200
```

**Generated visualizations**: Pareto front, evolution, trade-off analysis, solution selector

## Technical Notes

### File Formats
- All visualizations are saved as interactive HTML files
- Self-contained with embedded JavaScript (no internet required)
- Compatible with all modern web browsers
- Can be shared and viewed offline

### Data Requirements
- Minimum 5 completed trials for basic visualizations
- Minimum 10 trials recommended for meaningful analysis
- Multi-objective requires at least 3 Pareto solutions for full features

### Performance
- Visualizations are optimized for studies with up to 1000 trials
- Large studies (>1000 trials) may have longer generation times
- Interactive features remain responsive for datasets of reasonable size

This visualization system provides comprehensive insights into your optimization studies, helping you understand results, make informed decisions, and identify the best configurations for your specific needs.
