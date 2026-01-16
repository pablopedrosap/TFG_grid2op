#!/usr/bin/env python3
"""
Load Margin Analysis - Main Script
==================================
Script principal para ejecutar el análisis de margen de carga
usando Grid2Op.

Uso:
    python run_analysis.py [--env ENV_NAME] [--agent TYPE] [--lambda-max LAMBDA] [--output DIR]

Ejemplos:
    python run_analysis.py --agent baseline --output results_baseline
    python run_analysis.py --agent greedy --output results_greedy
    python run_analysis.py --agent sensitivity --output results_sensitivity
    python run_analysis.py --agent all --output results_comparison

Author: Pablo Pedrosa Prats
TFG - ICAI 2025
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import grid2op

from src.load_margin import LoadMarginAnalyzer, create_results_dataframe
from src.agents.greedy_agent import GreedyLoadMarginOptimizer
from src.agents.sensitivity_agent import SensitivityLoadMarginOptimizer
from src.visualization import generate_full_report, plot_method_comparison, plot_pareto_frontier, plot_pareto_with_time
from src.pareto_analysis import ParetoAnalyzer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load Margin Analysis using Grid2Op"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="l2rpn_case14_sandbox",
        help="Grid2Op environment name (default: l2rpn_case14_sandbox)"
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=1.5,
        help="Maximum lambda to test (default: 1.5)"
    )
    parser.add_argument(
        "--lambda-step",
        type=float,
        default=0.01,
        help="Lambda increment step (default: 0.01)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    parser.add_argument(
        "--n1-lines",
        type=int,
        default=5,
        help="Number of lines to analyze in N-1 (default: 5)"
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=10,
        help="Maximum corrective actions (default: 10)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="all",
        choices=["baseline", "greedy", "sensitivity", "all"],
        help="Agent type to use: 'baseline' (no agent), 'greedy', 'sensitivity', or 'all' (default: all)"
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Run Pareto frontier analysis (lambda* vs number of actions)"
    )
    parser.add_argument(
        "--pareto-budget",
        type=int,
        default=10,
        help="Maximum action budget for Pareto analysis (default: 10)"
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 60)
    print("LOAD MARGIN ANALYSIS USING GRID2OP")
    print("TFG - Pablo Pedrosa Prats")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    print(f"\nInitializing environment: {args.env}")
    try:
        env = grid2op.make(args.env)
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nAvailable environments:")
        print("  - l2rpn_case14_sandbox (IEEE 14 bus)")
        print("  - l2rpn_neurips_2020_track1 (118 bus)")
        sys.exit(1)

    print(f"  Lines: {env.n_line}")
    print(f"  Substations: {env.n_sub}")
    print(f"  Generators: {env.n_gen}")
    print(f"  Loads: {env.n_load}")

    # Initialize analyzer
    analyzer = LoadMarginAnalyzer(
        env=env,
        v_min_pu=0.9,
        v_max_pu=1.1,
        rho_max=1.0
    )

    # 1. Base case analysis
    print("\n" + "=" * 60)
    print("1. BASE CASE ANALYSIS")
    print("=" * 60)

    result_base = analyzer.calculate_load_margin(
        lambda_start=1.0,
        lambda_end=args.lambda_max,
        lambda_step=args.lambda_step,
        verbose=True
    )

    # 2. N-1 Contingency analysis
    print("\n" + "=" * 60)
    print("2. N-1 CONTINGENCY ANALYSIS")
    print("=" * 60)

    obs = env.reset()
    most_loaded_lines = np.argsort(obs.rho)[-args.n1_lines:][::-1].tolist()
    print(f"Analyzing lines: {most_loaded_lines}")

    n1_results = analyzer.run_n1_analysis(
        line_ids=most_loaded_lines,
        lambda_end=args.lambda_max,
        lambda_step=args.lambda_step,
        verbose=True
    )

    # 3. Greedy agent optimization
    greedy_results = None
    greedy_time = None
    run_greedy = args.agent in ["greedy", "all"]

    if run_greedy:
        print("\n" + "=" * 60)
        print("3. GREEDY AGENT OPTIMIZATION")
        print("=" * 60)

        greedy_optimizer = GreedyLoadMarginOptimizer(
            env=env,
            rho_threshold=0.9,
            lookahead_steps=3
        )

        start_time = time.time()
        greedy_results = greedy_optimizer.optimize_load_margin(
            lambda_start=1.0,
            lambda_end=args.lambda_max,
            lambda_step=args.lambda_step,
            max_actions=args.max_actions,
            verbose=True
        )
        greedy_time = time.time() - start_time
        greedy_results['computation_time'] = greedy_time
        print(f"\nGreedy computation time: {greedy_time:.2f}s")

    # 4. Sensitivity agent optimization
    sensitivity_results = None
    sensitivity_time = None
    run_sensitivity = args.agent in ["sensitivity", "all"]

    if run_sensitivity:
        print("\n" + "=" * 60)
        print("4. SENSITIVITY AGENT OPTIMIZATION (PTDF/LODF)")
        print("=" * 60)

        sensitivity_optimizer = SensitivityLoadMarginOptimizer(
            env=env,
            top_k_candidates=5,
            rho_threshold=0.9
        )

        start_time = time.time()
        sensitivity_results = sensitivity_optimizer.optimize_load_margin(
            lambda_start=1.0,
            lambda_end=args.lambda_max,
            lambda_step=args.lambda_step,
            max_actions=args.max_actions,
            verbose=True
        )
        sensitivity_time = time.time() - start_time
        sensitivity_results['computation_time'] = sensitivity_time
        print(f"\nSensitivity computation time: {sensitivity_time:.2f}s")

    # 5. Pareto frontier analysis (optional)
    pareto_results = None
    if args.pareto:
        print("\n" + "=" * 60)
        print("5. PARETO FRONTIER ANALYSIS")
        print("=" * 60)

        pareto_analyzer = ParetoAnalyzer(
            env=env,
            rho_threshold=0.9,
            top_k_candidates=5
        )

        pareto_frontier = pareto_analyzer.compute_pareto_frontier(
            max_budget=args.pareto_budget,
            lambda_end=args.lambda_max,
            verbose=True
        )

        # Print Pareto table
        print("\n")
        print(pareto_analyzer.get_pareto_table(pareto_frontier))

        # Prepare results for plotting
        pareto_results = {
            'base_lambda': pareto_frontier.base_lambda,
            'max_lambda': pareto_frontier.max_lambda,
            'knee_point': pareto_frontier.knee_point,
            'optimal_budget': pareto_frontier.optimal_n_actions,
            'pareto_efficiency': pareto_frontier.pareto_efficiency,
            'points': [
                {
                    'n_actions': p.n_actions,
                    'lambda_max': p.lambda_max,
                    'improvement': p.improvement_vs_base,
                    'marginal': p.marginal_improvement,
                    'time': p.computation_time
                }
                for p in pareto_frontier.points
            ]
        }

        # Generate Pareto plots
        fig = plot_pareto_frontier(
            pareto_results,
            save_path=str(output_dir / "06_pareto_frontier.png")
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

        fig = plot_pareto_with_time(
            pareto_results,
            save_path=str(output_dir / "07_pareto_time.png")
        )
        plt.close(fig)

        # Save Pareto results to JSON
        import json
        with open(output_dir / "pareto_results.json", 'w') as f:
            json.dump(pareto_results, f, indent=2)

        print(f"\nPareto analysis saved to: {output_dir}")

    # 6. Generate report
    print("\n" + "=" * 60)
    print("6. GENERATING REPORT")
    print("=" * 60)

    df_summary = generate_full_report(
        base_results=n1_results,
        greedy_results=greedy_results,
        sensitivity_results=sensitivity_results,
        output_dir=str(output_dir)
    )

    # 6. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)

    print(f"\n{'Method':<25} {'λ*':<10} {'Improvement':<12} {'Time (s)':<10}")
    print("-" * 60)

    base_lambda = result_base.lambda_max
    print(f"{'Base Case':<25} {base_lambda:<10.3f} {'-':<12} {'N/A':<10}")

    if greedy_results:
        greedy_lambda = greedy_results['lambda_max']
        greedy_imp = (greedy_lambda - base_lambda) / base_lambda * 100
        greedy_t = greedy_results.get('computation_time', 0)
        print(f"{'Greedy (Brute Force)':<25} {greedy_lambda:<10.3f} {greedy_imp:+.1f}%{'':>5} {greedy_t:<10.2f}")

    if sensitivity_results:
        sens_lambda = sensitivity_results['lambda_max']
        sens_imp = (sens_lambda - base_lambda) / base_lambda * 100
        sens_t = sensitivity_results.get('computation_time', 0)
        print(f"{'Sensitivity (PTDF)':<25} {sens_lambda:<10.3f} {sens_imp:+.1f}%{'':>5} {sens_t:<10.2f}")

    # Show speedup if both agents ran
    if greedy_results and sensitivity_results and greedy_t > 0 and sens_t > 0:
        speedup = greedy_t / sens_t
        print(f"\n** Speedup Sensitivity vs Greedy: {speedup:.1f}x faster **")

    # Show Pareto summary if computed
    if pareto_results:
        print(f"\n{'='*60}")
        print("PARETO ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Knee point: {pareto_results['knee_point']} actions")
        print(f"Optimal budget: {pareto_results['optimal_budget']} actions")
        print(f"Pareto efficiency: {pareto_results['pareto_efficiency']:.1f}%")
        print(f"  (= % of max improvement achieved with 20% of action budget)")

    print(f"\nResults saved to: {output_dir}")
    print("\nAnalysis complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
