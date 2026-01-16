"""
Statistical Analysis Module for Paper-Quality Results
======================================================
Implementa analisis estadistico robusto para validar los resultados
del analisis de Pareto con multiples runs y diferentes semillas.

Este modulo proporciona:
1. Multiples runs con diferentes semillas para robustez estadistica
2. Calculo de media, desviacion estandar, intervalos de confianza
3. Analisis de la monotonicidad de la frontera de Pareto
4. Comparacion con baselines teoricos
5. Tests estadisticos para validar hipotesis

Author: Pablo Pedrosa Prats
TFG - ICAI 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time
import json
from pathlib import Path
import grid2op
from grid2op.Environment import Environment

from .pareto_analysis import ParetoAnalyzer, ParetoFrontier, ParetoPoint


@dataclass
class StatisticalParetoPoint:
    """Punto de Pareto con estadisticas de multiples runs."""
    n_actions: int
    lambda_mean: float
    lambda_std: float
    lambda_min: float
    lambda_max: float
    lambda_ci_lower: float  # 95% CI lower bound
    lambda_ci_upper: float  # 95% CI upper bound
    improvement_mean: float
    improvement_std: float
    n_runs: int
    all_lambdas: List[float] = field(default_factory=list)


@dataclass
class StatisticalParetoFrontier:
    """Frontera de Pareto con analisis estadistico."""
    points: List[StatisticalParetoPoint]
    base_lambda_mean: float
    base_lambda_std: float
    max_lambda_mean: float
    optimal_n_actions: int
    pareto_efficiency_mean: float
    pareto_efficiency_std: float
    knee_point_mode: int  # Most frequent knee point across runs
    monotonicity_rate: float  # % of runs where frontier is monotonic
    n_runs: int
    seeds_used: List[int]


class StatisticalParetoAnalyzer:
    """
    Analizador de Pareto con validacion estadistica.

    Ejecuta multiples runs con diferentes semillas para obtener
    resultados estadisticamente robustos y publicables.
    """

    def __init__(
        self,
        env: Environment,
        rho_threshold: float = 0.9,
        top_k_candidates: int = 10
    ):
        """
        Inicializa el analizador estadistico.

        Args:
            env: Entorno Grid2Op
            rho_threshold: Umbral de sobrecarga
            top_k_candidates: Candidatas a evaluar
        """
        self.env = env
        self.rho_threshold = rho_threshold
        self.top_k_candidates = top_k_candidates

    def compute_statistical_frontier(
        self,
        max_budget: int = 10,
        n_runs: int = 10,
        lambda_start: float = 1.0,
        lambda_end: float = 2.0,
        lambda_step: float = 0.01,
        base_seed: int = 42,
        verbose: bool = True
    ) -> StatisticalParetoFrontier:
        """
        Calcula la frontera de Pareto con estadisticas de multiples runs.

        Args:
            max_budget: Numero maximo de acciones
            n_runs: Numero de ejecuciones con diferentes semillas
            lambda_start: Lambda inicial
            lambda_end: Lambda maximo
            lambda_step: Incremento
            base_seed: Semilla base (se incrementa para cada run)
            verbose: Imprimir progreso

        Returns:
            Frontera de Pareto con estadisticas
        """
        if verbose:
            print("=" * 70)
            print("STATISTICAL PARETO FRONTIER ANALYSIS")
            print("=" * 70)
            print(f"Running {n_runs} experiments with different seeds")
            print(f"Action budget range: 0 to {max_budget}")
            print(f"Network: {self.env.n_line} lines, {self.env.n_sub} substations")
            print()

        # Store results from all runs
        all_frontiers: List[ParetoFrontier] = []
        seeds_used = []

        start_time = time.time()

        for run in range(n_runs):
            seed = base_seed + run * 7  # Use different seeds
            seeds_used.append(seed)

            if verbose:
                print(f"Run {run + 1}/{n_runs} (seed={seed})...", end=" ", flush=True)

            run_start = time.time()

            # Create analyzer with this seed
            analyzer = ParetoAnalyzer(
                env=self.env,
                rho_threshold=self.rho_threshold,
                top_k_candidates=self.top_k_candidates,
                seed=seed
            )

            # Compute frontier
            frontier = analyzer.compute_pareto_frontier(
                max_budget=max_budget,
                lambda_start=lambda_start,
                lambda_end=lambda_end,
                lambda_step=lambda_step,
                verbose=False
            )

            all_frontiers.append(frontier)

            run_time = time.time() - run_start
            if verbose:
                print(f"done ({run_time:.1f}s) - base={frontier.base_lambda:.3f}, max={frontier.max_lambda:.3f}")

        total_time = time.time() - start_time

        if verbose:
            print()
            print(f"Total computation time: {total_time:.1f}s")
            print()

        # Aggregate statistics
        statistical_frontier = self._aggregate_frontiers(
            all_frontiers, seeds_used, max_budget
        )

        if verbose:
            self._print_statistical_summary(statistical_frontier)

        return statistical_frontier

    def _aggregate_frontiers(
        self,
        frontiers: List[ParetoFrontier],
        seeds: List[int],
        max_budget: int
    ) -> StatisticalParetoFrontier:
        """
        Agrega resultados de multiples fronteras en estadisticas.

        Args:
            frontiers: Lista de fronteras de diferentes runs
            seeds: Semillas usadas
            max_budget: Presupuesto maximo

        Returns:
            Frontera con estadisticas agregadas
        """
        n_runs = len(frontiers)

        # Aggregate per action count
        stat_points = []

        for n_actions in range(max_budget + 1):
            lambdas = [f.points[n_actions].lambda_max for f in frontiers]
            improvements = [f.points[n_actions].improvement_vs_base for f in frontiers]

            lambda_mean = np.mean(lambdas)
            lambda_std = np.std(lambdas)

            # 95% confidence interval
            ci_margin = 1.96 * lambda_std / np.sqrt(n_runs)

            stat_points.append(StatisticalParetoPoint(
                n_actions=n_actions,
                lambda_mean=lambda_mean,
                lambda_std=lambda_std,
                lambda_min=np.min(lambdas),
                lambda_max=np.max(lambdas),
                lambda_ci_lower=lambda_mean - ci_margin,
                lambda_ci_upper=lambda_mean + ci_margin,
                improvement_mean=np.mean(improvements),
                improvement_std=np.std(improvements),
                n_runs=n_runs,
                all_lambdas=lambdas
            ))

        # Aggregate frontier-level metrics
        base_lambdas = [f.base_lambda for f in frontiers]
        max_lambdas = [f.max_lambda for f in frontiers]
        pareto_efficiencies = [f.pareto_efficiency for f in frontiers]
        knee_points = [f.knee_point for f in frontiers]
        optimal_ns = [f.optimal_n_actions for f in frontiers]

        # Calculate monotonicity rate
        monotonicity_count = 0
        for f in frontiers:
            lambdas = [p.lambda_max for p in f.points]
            is_monotonic = all(lambdas[i] <= lambdas[i+1] for i in range(len(lambdas)-1))
            if is_monotonic:
                monotonicity_count += 1

        monotonicity_rate = monotonicity_count / n_runs * 100

        # Find mode of knee points
        knee_point_mode = max(set(knee_points), key=knee_points.count)
        optimal_mode = max(set(optimal_ns), key=optimal_ns.count)

        return StatisticalParetoFrontier(
            points=stat_points,
            base_lambda_mean=np.mean(base_lambdas),
            base_lambda_std=np.std(base_lambdas),
            max_lambda_mean=np.mean(max_lambdas),
            optimal_n_actions=optimal_mode,
            pareto_efficiency_mean=np.mean(pareto_efficiencies),
            pareto_efficiency_std=np.std(pareto_efficiencies),
            knee_point_mode=knee_point_mode,
            monotonicity_rate=monotonicity_rate,
            n_runs=n_runs,
            seeds_used=seeds
        )

    def _print_statistical_summary(self, frontier: StatisticalParetoFrontier):
        """Imprime resumen estadistico formateado."""
        print("=" * 70)
        print("STATISTICAL SUMMARY")
        print("=" * 70)
        print(f"Number of runs: {frontier.n_runs}")
        print(f"Base lambda*: {frontier.base_lambda_mean:.3f} +/- {frontier.base_lambda_std:.3f}")
        print(f"Max lambda* achievable: {frontier.max_lambda_mean:.3f}")
        print(f"Pareto efficiency: {frontier.pareto_efficiency_mean:.1f}% +/- {frontier.pareto_efficiency_std:.1f}%")
        print(f"Optimal budget (mode): {frontier.optimal_n_actions} actions")
        print(f"Knee point (mode): {frontier.knee_point_mode} actions")
        print(f"Monotonicity rate: {frontier.monotonicity_rate:.1f}%")
        print()

        # Detailed table
        print("-" * 90)
        print(f"{'N Actions':>10} | {'Lambda* Mean':>12} | {'Std':>8} | {'95% CI':>20} | {'Improvement':>12}")
        print("-" * 90)

        for p in frontier.points:
            ci_str = f"[{p.lambda_ci_lower:.3f}, {p.lambda_ci_upper:.3f}]"
            imp_str = f"{p.improvement_mean:+.1f}% +/- {p.improvement_std:.1f}%"
            print(f"{p.n_actions:>10} | {p.lambda_mean:>12.3f} | {p.lambda_std:>8.3f} | {ci_str:>20} | {imp_str:>12}")

        print("-" * 90)

    def analyze_monotonicity(
        self,
        frontier: StatisticalParetoFrontier
    ) -> Dict:
        """
        Analiza la monotonicidad de la frontera de Pareto.

        Esta es una contribucion clave: demostrar que la frontera
        NO es necesariamente monotona en problemas de control topologico.

        Args:
            frontier: Frontera estadistica

        Returns:
            Diccionario con analisis de monotonicidad
        """
        analysis = {
            "monotonicity_rate": frontier.monotonicity_rate,
            "is_generally_monotonic": frontier.monotonicity_rate > 80,
            "violations": [],
            "theoretical_explanation": ""
        }

        # Find violations in mean values
        means = [p.lambda_mean for p in frontier.points]
        for i in range(len(means) - 1):
            if means[i] > means[i + 1]:
                analysis["violations"].append({
                    "from_actions": i,
                    "to_actions": i + 1,
                    "lambda_drop": means[i] - means[i + 1],
                    "relative_drop": (means[i] - means[i + 1]) / means[i] * 100
                })

        # Theoretical explanation
        if len(analysis["violations"]) > 0:
            analysis["theoretical_explanation"] = """
The non-monotonicity of the Pareto frontier in topological control can be explained by:

1. **Action Interference**: Additional topological actions may conflict with previously
   beneficial actions, creating suboptimal configurations.

2. **Local Optima**: The greedy search gets trapped in different local optima depending
   on the action budget, and more actions don't guarantee escaping these optima.

3. **Flow Redistribution Non-linearity**: Power flow redistribution after topology changes
   is inherently non-linear. The LODF approximation is linear, but actual flows are not.

4. **Network Connectivity Constraints**: More disconnections may violate connectivity
   constraints or create islanding situations that reduce overall capacity.

5. **Stochastic Environment**: Grid2Op environments have inherent stochasticity in
   initial conditions, which affects the optimization landscape.

This finding suggests that operators should NOT assume that allowing more corrective
actions will always improve the load margin. Careful analysis of the Pareto frontier
is necessary for each specific network topology.
"""
        else:
            analysis["theoretical_explanation"] = """
The frontier is largely monotonic, suggesting that for this network topology,
additional corrective actions consistently improve the load margin. However,
the diminishing marginal returns observed indicate the presence of a natural
saturation point where further actions provide minimal benefit.
"""

        return analysis

    def compare_with_theoretical_bound(
        self,
        frontier: StatisticalParetoFrontier
    ) -> Dict:
        """
        Compara los resultados con cotas teoricas.

        Args:
            frontier: Frontera estadistica

        Returns:
            Diccionario con comparacion teorica
        """
        # Theoretical upper bound: linear improvement (best case)
        # Each action provides constant improvement
        base = frontier.base_lambda_mean
        max_achieved = frontier.max_lambda_mean

        max_improvement = (max_achieved - base) / base * 100
        n_for_max = len(frontier.points) - 1

        # Linear bound: equal improvement per action
        linear_bound_per_action = max_improvement / n_for_max if n_for_max > 0 else 0

        # Actual efficiency at each point
        efficiencies = []
        for p in frontier.points[1:]:  # Skip base case
            if p.n_actions > 0:
                actual_imp_per_action = p.improvement_mean / p.n_actions
                efficiency = actual_imp_per_action / linear_bound_per_action * 100 if linear_bound_per_action > 0 else 0
                efficiencies.append({
                    "n_actions": p.n_actions,
                    "efficiency_vs_linear": efficiency,
                    "improvement_per_action": actual_imp_per_action
                })

        # Pareto principle check (80/20 rule)
        twenty_percent_budget = max(1, int(n_for_max * 0.2))
        improvement_at_20pct = frontier.points[twenty_percent_budget].improvement_mean
        pareto_ratio = improvement_at_20pct / max_improvement * 100 if max_improvement > 0 else 0

        return {
            "max_improvement_achieved": max_improvement,
            "actions_for_max": n_for_max,
            "linear_bound_per_action": linear_bound_per_action,
            "efficiency_by_action": efficiencies,
            "pareto_principle": {
                "20_percent_budget": twenty_percent_budget,
                "improvement_at_20_percent": improvement_at_20pct,
                "percent_of_max_achieved": pareto_ratio,
                "follows_pareto": pareto_ratio >= 70  # 70%+ of benefit with 20% actions
            }
        }

    def save_results(
        self,
        frontier: StatisticalParetoFrontier,
        output_dir: str,
        additional_analysis: Dict = None
    ):
        """
        Guarda resultados en formato publicable.

        Args:
            frontier: Frontera estadistica
            output_dir: Directorio de salida
            additional_analysis: Analisis adicional a guardar
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Main results
        results = {
            "metadata": {
                "n_runs": frontier.n_runs,
                "seeds_used": frontier.seeds_used,
                "n_lines": self.env.n_line,
                "n_substations": self.env.n_sub,
                "rho_threshold": self.rho_threshold
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
            ]
        }

        if additional_analysis:
            results["analysis"] = additional_analysis

        # Save JSON
        with open(output_path / "statistical_pareto_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Save CSV for easy plotting
        csv_lines = ["n_actions,lambda_mean,lambda_std,ci_lower,ci_upper,improvement_mean,improvement_std"]
        for p in frontier.points:
            csv_lines.append(
                f"{p.n_actions},{p.lambda_mean:.4f},{p.lambda_std:.4f},"
                f"{p.lambda_ci_lower:.4f},{p.lambda_ci_upper:.4f},"
                f"{p.improvement_mean:.2f},{p.improvement_std:.2f}"
            )

        with open(output_path / "statistical_pareto_results.csv", "w") as f:
            f.write("\n".join(csv_lines))

        print(f"Results saved to {output_path}")


def run_paper_quality_experiment(
    env_name: str = "l2rpn_case14_sandbox",
    max_budget: int = 10,
    n_runs: int = 10,
    lambda_max: float = 2.0,
    output_dir: str = "results_paper"
) -> Dict:
    """
    Ejecuta experimento con calidad de paper.

    Args:
        env_name: Nombre del entorno Grid2Op
        max_budget: Presupuesto maximo de acciones
        n_runs: Numero de ejecuciones
        lambda_max: Lambda maximo
        output_dir: Directorio de salida

    Returns:
        Resultados completos
    """
    print("=" * 70)
    print("PAPER-QUALITY PARETO FRONTIER EXPERIMENT")
    print("=" * 70)
    print(f"Environment: {env_name}")
    print(f"Action budget: 0 to {max_budget}")
    print(f"Number of runs: {n_runs}")
    print(f"Lambda range: [1.0, {lambda_max}]")
    print()

    # Create environment
    env = grid2op.make(env_name)
    print(f"Network size: {env.n_line} lines, {env.n_sub} substations")
    print()

    # Run statistical analysis
    analyzer = StatisticalParetoAnalyzer(env)

    frontier = analyzer.compute_statistical_frontier(
        max_budget=max_budget,
        n_runs=n_runs,
        lambda_end=lambda_max,
        verbose=True
    )

    # Additional analysis
    monotonicity_analysis = analyzer.analyze_monotonicity(frontier)
    theoretical_comparison = analyzer.compare_with_theoretical_bound(frontier)

    additional = {
        "monotonicity": monotonicity_analysis,
        "theoretical_comparison": theoretical_comparison
    }

    # Save results
    analyzer.save_results(frontier, output_dir, additional)

    # Print key findings for paper
    print()
    print("=" * 70)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 70)
    print(f"1. Base case load margin: {frontier.base_lambda_mean:.3f} +/- {frontier.base_lambda_std:.3f}")
    print(f"2. Maximum achievable: {frontier.max_lambda_mean:.3f}")
    print(f"3. Optimal action budget: {frontier.optimal_n_actions} actions")
    print(f"4. Pareto efficiency: {frontier.pareto_efficiency_mean:.1f}% +/- {frontier.pareto_efficiency_std:.1f}%")
    print(f"5. Monotonicity rate: {frontier.monotonicity_rate:.1f}%")

    pareto = theoretical_comparison["pareto_principle"]
    print(f"6. Pareto principle: {pareto['percent_of_max_achieved']:.1f}% of benefit with 20% of actions")
    print(f"   -> {'CONFIRMED' if pareto['follows_pareto'] else 'NOT CONFIRMED'}")

    if len(monotonicity_analysis["violations"]) > 0:
        print(f"7. Non-monotonicity detected: {len(monotonicity_analysis['violations'])} violations")
        print("   -> This is a KEY FINDING: more actions != better results")

    return {
        "frontier": frontier,
        "monotonicity": monotonicity_analysis,
        "theoretical": theoretical_comparison
    }


if __name__ == "__main__":
    results = run_paper_quality_experiment(
        env_name="l2rpn_case14_sandbox",
        max_budget=10,
        n_runs=10,
        output_dir="results_paper_ieee14"
    )
