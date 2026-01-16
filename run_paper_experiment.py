#!/usr/bin/env python3
"""
Paper-Quality Experiment Runner
===============================
Script para ejecutar experimentos con calidad de publicacion.

Ejecuta analisis de Pareto en multiples redes con validacion estadistica.

Usage:
    python run_paper_experiment.py [--n-runs N] [--budget B] [--output DIR]

Author: Pablo Pedrosa Prats
TFG - ICAI 2025
"""

import argparse
import sys
import json
from pathlib import Path
import time

import numpy as np
import grid2op
import matplotlib.pyplot as plt

from src.statistical_analysis import (
    StatisticalParetoAnalyzer,
    run_paper_quality_experiment
)
from src.visualization import (
    plot_statistical_pareto_frontier,
    plot_monotonicity_analysis
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run paper-quality Pareto frontier experiments"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="l2rpn_case14_sandbox",
        help="Grid2Op environment (default: l2rpn_case14_sandbox)"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of runs for statistical analysis (default: 10)"
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=10,
        help="Maximum action budget (default: 10)"
    )
    parser.add_argument(
        "--lambda-max",
        type=float,
        default=2.0,
        help="Maximum lambda to test (default: 2.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_paper",
        help="Output directory (default: results_paper)"
    )
    parser.add_argument(
        "--all-networks",
        action="store_true",
        help="Run on all available networks (IEEE 14 and larger)"
    )

    return parser.parse_args()


def run_single_network_experiment(
    env_name: str,
    n_runs: int,
    max_budget: int,
    lambda_max: float,
    output_dir: str
) -> dict:
    """
    Run complete paper-quality experiment on a single network.

    Args:
        env_name: Grid2Op environment name
        n_runs: Number of statistical runs
        max_budget: Maximum action budget
        lambda_max: Maximum lambda to test
        output_dir: Output directory

    Returns:
        Dictionary with all results
    """
    print("=" * 70)
    print(f"PAPER-QUALITY EXPERIMENT: {env_name}")
    print("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create environment
    try:
        env = grid2op.make(env_name)
    except Exception as e:
        print(f"Error creating environment {env_name}: {e}")
        return None

    print(f"Network: {env.n_line} lines, {env.n_sub} substations")
    print(f"Statistical runs: {n_runs}")
    print(f"Action budget: 0 to {max_budget}")
    print()

    # Run statistical analysis
    analyzer = StatisticalParetoAnalyzer(env)

    start_time = time.time()

    frontier = analyzer.compute_statistical_frontier(
        max_budget=max_budget,
        n_runs=n_runs,
        lambda_end=lambda_max,
        verbose=True
    )

    # Additional analysis
    monotonicity = analyzer.analyze_monotonicity(frontier)
    theoretical = analyzer.compare_with_theoretical_bound(frontier)

    total_time = time.time() - start_time

    # Prepare results for saving
    results = {
        "metadata": {
            "env_name": env_name,
            "n_lines": env.n_line,
            "n_substations": env.n_sub,
            "n_runs": frontier.n_runs,
            "seeds_used": frontier.seeds_used,
            "max_budget": max_budget,
            "lambda_max": lambda_max,
            "total_computation_time": total_time
        },
        "summary": {
            "base_lambda_mean": frontier.base_lambda_mean,
            "base_lambda_std": frontier.base_lambda_std,
            "max_lambda_mean": frontier.max_lambda_mean,
            "optimal_n_actions": frontier.optimal_n_actions,
            "pareto_efficiency_mean": frontier.pareto_efficiency_mean,
            "pareto_efficiency_std": frontier.pareto_efficiency_std,
            "knee_point_mode": frontier.knee_point_mode,
            "monotonicity_rate": frontier.monotonicity_rate
        },
        "points": [
            {
                "n_actions": p.n_actions,
                "lambda_mean": p.lambda_mean,
                "lambda_std": p.lambda_std,
                "lambda_min": p.lambda_min,
                "lambda_max": p.lambda_max,
                "ci_lower": p.lambda_ci_lower,
                "ci_upper": p.lambda_ci_upper,
                "improvement_mean": p.improvement_mean,
                "improvement_std": p.improvement_std,
                "all_lambdas": p.all_lambdas
            }
            for p in frontier.points
        ],
        "analysis": {
            "monotonicity": monotonicity,
            "theoretical_comparison": theoretical
        }
    }

    # Save JSON results
    with open(output_path / "statistical_pareto_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save CSV for easy import to LaTeX/Excel
    csv_lines = [
        "n_actions,lambda_mean,lambda_std,ci_lower,ci_upper,improvement_mean,improvement_std"
    ]
    for p in frontier.points:
        csv_lines.append(
            f"{p.n_actions},{p.lambda_mean:.4f},{p.lambda_std:.4f},"
            f"{p.lambda_ci_lower:.4f},{p.lambda_ci_upper:.4f},"
            f"{p.improvement_mean:.2f},{p.improvement_std:.2f}"
        )
    with open(output_path / "statistical_pareto_results.csv", "w") as f:
        f.write("\n".join(csv_lines))

    # Generate paper-quality figures
    print()
    print("Generating paper-quality figures...")

    # Figure 1: Statistical Pareto Frontier
    fig1 = plot_statistical_pareto_frontier(
        results,
        title=f"Statistical Pareto Frontier - {env_name} (n={n_runs} runs)",
        save_path=str(output_path / "fig_statistical_pareto.png")
    )
    plt.close(fig1)

    # Figure 2: Monotonicity Analysis
    fig2 = plot_monotonicity_analysis(
        results,
        title=f"Monotonicity Analysis - {env_name}",
        save_path=str(output_path / "fig_monotonicity.png")
    )
    plt.close(fig2)

    print(f"Figures saved to {output_path}")

    # Print key findings
    print()
    print("=" * 70)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 70)
    print(f"1. Base load margin: {frontier.base_lambda_mean:.3f} +/- {frontier.base_lambda_std:.3f}")
    print(f"2. Maximum achievable: {frontier.max_lambda_mean:.3f}")
    print(f"3. Optimal action budget: {frontier.optimal_n_actions} actions")
    print(f"4. Pareto efficiency: {frontier.pareto_efficiency_mean:.1f}% +/- {frontier.pareto_efficiency_std:.1f}%")
    print(f"5. Monotonicity rate: {frontier.monotonicity_rate:.1f}%")

    pareto = theoretical["pareto_principle"]
    print(f"6. Pareto principle (80/20): {pareto['percent_of_max_achieved']:.1f}% benefit with 20% actions")
    print(f"   -> {'CONFIRMED' if pareto['follows_pareto'] else 'NOT CONFIRMED'}")

    n_violations = len(monotonicity.get("violations", []))
    if n_violations > 0:
        print(f"7. NON-MONOTONICITY DETECTED: {n_violations} violations")
        print("   -> KEY FINDING: More actions does NOT always mean better results!")
    else:
        print("7. Frontier is monotonic (as expected)")

    print()
    print(f"Total computation time: {total_time:.1f}s")
    print(f"Results saved to: {output_path}")

    return results


def run_multi_network_comparison(
    env_names: list,
    n_runs: int,
    max_budget: int,
    lambda_max: float,
    output_dir: str
):
    """
    Run experiments on multiple networks and generate comparison.

    Args:
        env_names: List of environment names
        n_runs: Number of runs per network
        max_budget: Maximum action budget
        lambda_max: Maximum lambda
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for env_name in env_names:
        env_output = output_path / env_name.replace("/", "_")
        try:
            results = run_single_network_experiment(
                env_name=env_name,
                n_runs=n_runs,
                max_budget=max_budget,
                lambda_max=lambda_max,
                output_dir=str(env_output)
            )
            if results:
                all_results[env_name] = results
        except Exception as e:
            print(f"Error running experiment on {env_name}: {e}")
            continue

        print()
        print("=" * 70)
        print()

    # Generate comparison summary
    if len(all_results) > 1:
        print("=" * 70)
        print("MULTI-NETWORK COMPARISON SUMMARY")
        print("=" * 70)

        print(f"\n{'Network':<30} {'Lines':>8} {'Base λ*':>12} {'Max λ*':>12} {'Pareto Eff':>12} {'Monotonic':>10}")
        print("-" * 90)

        for env_name, results in all_results.items():
            meta = results["metadata"]
            summ = results["summary"]
            print(
                f"{env_name:<30} "
                f"{meta['n_lines']:>8} "
                f"{summ['base_lambda_mean']:>12.3f} "
                f"{summ['max_lambda_mean']:>12.3f} "
                f"{summ['pareto_efficiency_mean']:>11.1f}% "
                f"{summ['monotonicity_rate']:>9.1f}%"
            )

        # Save comparison
        with open(output_path / "multi_network_comparison.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print()
        print(f"Comparison saved to {output_path}")


def main():
    """Main execution function."""
    args = parse_args()

    if args.all_networks:
        # Run on multiple networks
        networks = [
            "l2rpn_case14_sandbox",  # IEEE 14 (20 lines)
            "l2rpn_neurips_2020_track1_small",  # Larger network if available
        ]
        run_multi_network_comparison(
            env_names=networks,
            n_runs=args.n_runs,
            max_budget=args.budget,
            lambda_max=args.lambda_max,
            output_dir=args.output
        )
    else:
        # Single network experiment
        run_single_network_experiment(
            env_name=args.env,
            n_runs=args.n_runs,
            max_budget=args.budget,
            lambda_max=args.lambda_max,
            output_dir=args.output
        )

    print()
    print("Experiment complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
