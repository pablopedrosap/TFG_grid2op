"""
Pareto Frontier Analysis (Lambda* vs Number of Actions)
=======================================================
Analiza el trade-off entre margen de carga (lambda*) y complejidad
operacional (numero de acciones topologicas).

Hipotesis: El 80% del beneficio en lambda* se consigue con el 20%
de las acciones posibles (Ley de Pareto aplicada a control topologico).

Este analisis es relevante para:
- Operadores: Saber cuantas acciones vale la pena tomar
- Planificadores: Disenar procedimientos de emergencia
- Investigacion: Cuantificar el "knee point" de la frontera de Pareto

La metodologia usa busqueda exhaustiva de las mejores acciones para
cada presupuesto, garantizando que lambda* sea monotono creciente
con el numero de acciones.

Author: Pablo Pedrosa Prats
TFG - ICAI 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time
import grid2op
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction


@dataclass
class ParetoPoint:
    """Un punto en la frontera de Pareto."""
    n_actions: int
    lambda_max: float
    actions_taken: List[Dict]
    computation_time: float
    improvement_vs_base: float  # Mejora porcentual vs caso base
    marginal_improvement: float  # Mejora vs punto anterior


@dataclass
class ParetoFrontier:
    """Frontera de Pareto completa."""
    points: List[ParetoPoint]
    base_lambda: float
    max_lambda: float
    optimal_n_actions: int  # Punto con mejor ratio beneficio/coste
    pareto_efficiency: float  # Porcentaje del beneficio con 20% de acciones
    knee_point: int  # Numero de acciones en el "codo" de la curva


class ParetoAnalyzer:
    """
    Analiza la frontera de Pareto entre lambda* y numero de acciones.

    Metodologia:
    1. Calcular lambda* sin acciones (caso base)
    2. Incrementar el numero maximo de acciones de 1 a N
    3. Para cada limite, encontrar lambda* optimo
    4. Construir la frontera de Pareto
    5. Identificar el "knee point" donde el beneficio marginal decrece
    """

    def __init__(
        self,
        env: Environment,
        rho_threshold: float = 0.9,
        top_k_candidates: int = 5,
        seed: int = 42
    ):
        """
        Inicializa el analizador de Pareto.

        Args:
            env: Entorno Grid2Op
            rho_threshold: Umbral de sobrecarga para activar acciones
            top_k_candidates: Candidatas a evaluar por iteracion
            seed: Semilla para reproducibilidad
        """
        self.env = env
        self.rho_threshold = rho_threshold
        self.top_k_candidates = top_k_candidates
        self.seed = seed

        # Set seed for reproducibility
        self.env.seed(seed)
        np.random.seed(seed)

    def _calculate_lambda_with_budget(
        self,
        max_actions: int,
        lambda_start: float = 1.0,
        lambda_end: float = 2.0,
        lambda_step: float = 0.01,
        verbose: bool = False
    ) -> Tuple[float, List[Dict], float]:
        """
        Calcula lambda* con un presupuesto limitado de acciones.

        Usa busqueda greedy con evaluacion exhaustiva de candidatas
        para encontrar la mejor combinacion de acciones.

        Args:
            max_actions: Numero maximo de acciones permitidas
            lambda_start: Lambda inicial
            lambda_end: Lambda maximo a probar
            lambda_step: Incremento de lambda
            verbose: Imprimir progreso

        Returns:
            (lambda_max, acciones_tomadas, tiempo_computo)
        """
        obs = self.env.reset()
        simulator = obs.get_simulator()

        init_load_p = obs.load_p.copy()
        init_load_q = obs.load_q.copy()
        init_gen_p = obs.gen_p.copy()

        lambda_values = np.arange(lambda_start, lambda_end + lambda_step, lambda_step)

        actions_taken = []
        actions_count = 0
        lambda_max = lambda_start

        # Track disconnected lines to avoid re-trying them
        disconnected_lines = set()

        start_time = time.time()

        for lam in lambda_values:
            new_load_p = init_load_p * lam
            new_load_q = init_load_q * lam
            new_gen_p = init_gen_p * lam

            # Build combined action
            action = self._build_combined_action(disconnected_lines)

            try:
                result = simulator.predict(
                    action,
                    new_gen_p=new_gen_p,
                    new_load_p=new_load_p,
                    new_load_q=new_load_q
                )

                if not result.converged:
                    break

                sim_obs = result.current_obs
                rho_max = np.max(sim_obs.rho)

                # Check for overloads
                if rho_max > 1.0:
                    if actions_count < max_actions:
                        # Try to find a corrective action
                        best_line, best_rho = self._find_best_disconnection(
                            sim_obs, simulator, new_gen_p, new_load_p, new_load_q,
                            disconnected_lines
                        )

                        if best_line is not None and best_rho <= 1.0:
                            actions_count += 1
                            disconnected_lines.add(best_line)
                            actions_taken.append({
                                'lambda': lam,
                                'rho_before': rho_max,
                                'rho_after': best_rho,
                                'action': f'Disconnect line {best_line}',
                                'action_number': actions_count
                            })
                            lambda_max = lam
                            continue
                        else:
                            # No more beneficial actions available
                            break
                    else:
                        # Budget exhausted
                        break
                else:
                    lambda_max = lam

            except Exception as e:
                if verbose:
                    print(f"Error at lambda={lam:.3f}: {e}")
                break

        computation_time = time.time() - start_time

        return lambda_max, actions_taken, computation_time

    def _build_combined_action(self, disconnected_lines: set) -> BaseAction:
        """
        Construye una accion combinada con todas las lineas desconectadas.

        Args:
            disconnected_lines: Conjunto de lineas a desconectar

        Returns:
            Accion de Grid2Op
        """
        if not disconnected_lines:
            return self.env.action_space()

        try:
            set_status = [(int(line_id), -1) for line_id in disconnected_lines]
            action = self.env.action_space({"set_line_status": set_status})
            return action
        except:
            return self.env.action_space()

    def _find_best_disconnection(
        self,
        obs: BaseObservation,
        simulator,
        new_gen_p: np.ndarray,
        new_load_p: np.ndarray,
        new_load_q: np.ndarray,
        already_disconnected: set
    ) -> Tuple[Optional[int], float]:
        """
        Encuentra la mejor linea a desconectar para aliviar sobrecargas.

        Prueba todas las lineas candidatas y selecciona la que
        minimiza rho_max despues de la desconexion.

        Args:
            obs: Observacion actual con sobrecargas
            simulator: Simulador
            new_gen_p, new_load_p, new_load_q: Cargas escaladas
            already_disconnected: Lineas ya desconectadas

        Returns:
            (mejor_linea, rho_despues) o (None, inf) si no hay mejora
        """
        best_line = None
        best_rho = float('inf')

        # Get overloaded lines
        overloaded = np.where(obs.rho > self.rho_threshold)[0]

        # Build candidate set: exclude already disconnected and heavily overloaded
        candidates = []
        for line_id in range(obs.n_line):
            if line_id in already_disconnected:
                continue
            if not obs.line_status[line_id]:
                continue
            if obs.rho[line_id] > 2.0:  # Don't disconnect very overloaded lines
                continue
            candidates.append(line_id)

        # Also add lines with moderate flow that might help
        for line_id in candidates[:self.top_k_candidates * 2]:
            if line_id in already_disconnected:
                continue

            try:
                # Build action with this line also disconnected
                disconnected = already_disconnected | {line_id}
                action = self._build_combined_action(disconnected)

                result = simulator.predict(
                    action,
                    new_gen_p=new_gen_p,
                    new_load_p=new_load_p,
                    new_load_q=new_load_q
                )

                if result.converged:
                    rho_max = np.max(result.current_obs.rho)
                    if rho_max < best_rho:
                        best_rho = rho_max
                        best_line = line_id
            except:
                pass

        return best_line, best_rho

    def compute_pareto_frontier(
        self,
        max_budget: int = 10,
        lambda_start: float = 1.0,
        lambda_end: float = 2.0,
        lambda_step: float = 0.01,
        verbose: bool = True
    ) -> ParetoFrontier:
        """
        Calcula la frontera de Pareto completa.

        Args:
            max_budget: Numero maximo de acciones a probar
            lambda_start: Lambda inicial
            lambda_end: Lambda maximo
            lambda_step: Incremento de lambda
            verbose: Imprimir progreso

        Returns:
            Frontera de Pareto con todos los puntos
        """
        if verbose:
            print("=" * 60)
            print("PARETO FRONTIER ANALYSIS")
            print("=" * 60)
            print(f"Testing action budgets from 0 to {max_budget}")
            print()

        points = []

        # Base case (0 actions)
        if verbose:
            print("Computing base case (0 actions)...")

        base_lambda, _, base_time = self._calculate_lambda_with_budget(
            max_actions=0,
            lambda_start=lambda_start,
            lambda_end=lambda_end,
            lambda_step=lambda_step
        )

        points.append(ParetoPoint(
            n_actions=0,
            lambda_max=base_lambda,
            actions_taken=[],
            computation_time=base_time,
            improvement_vs_base=0.0,
            marginal_improvement=0.0
        ))

        if verbose:
            print(f"  Base case: lambda* = {base_lambda:.3f}")
            print()

        prev_lambda = base_lambda

        # Test increasing budgets
        for budget in range(1, max_budget + 1):
            if verbose:
                print(f"Testing budget = {budget} actions...")

            lambda_max, actions, comp_time = self._calculate_lambda_with_budget(
                max_actions=budget,
                lambda_start=lambda_start,
                lambda_end=lambda_end,
                lambda_step=lambda_step
            )

            improvement = (lambda_max - base_lambda) / base_lambda * 100
            marginal = (lambda_max - prev_lambda) / base_lambda * 100

            points.append(ParetoPoint(
                n_actions=budget,
                lambda_max=lambda_max,
                actions_taken=actions,
                computation_time=comp_time,
                improvement_vs_base=improvement,
                marginal_improvement=marginal
            ))

            if verbose:
                print(f"  lambda* = {lambda_max:.3f} ({improvement:+.1f}% vs base)")
                print(f"  Actions used: {len(actions)}")
                print(f"  Marginal improvement: {marginal:+.2f}%")
                print()

            prev_lambda = lambda_max

        # Compute frontier metrics
        max_lambda = max(p.lambda_max for p in points)
        max_improvement = max(p.improvement_vs_base for p in points)

        # Find knee point (where marginal improvement drops significantly)
        knee_point = self._find_knee_point(points)

        # Calculate Pareto efficiency (improvement with 20% of actions)
        twenty_percent_budget = max(1, int(max_budget * 0.2))
        improvement_at_20pct = points[twenty_percent_budget].improvement_vs_base
        pareto_efficiency = improvement_at_20pct / max_improvement * 100 if max_improvement > 0 else 100

        # Find optimal action count (best ratio)
        optimal_n = self._find_optimal_budget(points)

        frontier = ParetoFrontier(
            points=points,
            base_lambda=base_lambda,
            max_lambda=max_lambda,
            optimal_n_actions=optimal_n,
            pareto_efficiency=pareto_efficiency,
            knee_point=knee_point
        )

        if verbose:
            print("=" * 60)
            print("PARETO ANALYSIS SUMMARY")
            print("=" * 60)
            print(f"Base lambda*: {base_lambda:.3f}")
            print(f"Max lambda* achievable: {max_lambda:.3f}")
            print(f"Maximum improvement: {max_improvement:.1f}%")
            print(f"Knee point: {knee_point} actions")
            print(f"Pareto efficiency: {pareto_efficiency:.1f}% of benefit with 20% of actions")
            print(f"Optimal budget: {optimal_n} actions")

        return frontier

    def _find_knee_point(self, points: List[ParetoPoint]) -> int:
        """
        Encuentra el punto de inflexion (knee) de la frontera de Pareto.

        Usa el metodo de maxima distancia a la linea entre primer y ultimo punto.

        Args:
            points: Lista de puntos de Pareto

        Returns:
            Numero de acciones en el knee point
        """
        if len(points) < 3:
            return 1

        # Normalize data
        n_actions = np.array([p.n_actions for p in points])
        lambdas = np.array([p.lambda_max for p in points])

        # Normalize to [0, 1]
        n_norm = (n_actions - n_actions.min()) / (n_actions.max() - n_actions.min() + 1e-10)
        l_norm = (lambdas - lambdas.min()) / (lambdas.max() - lambdas.min() + 1e-10)

        # Line from first to last point
        p1 = np.array([n_norm[0], l_norm[0]])
        p2 = np.array([n_norm[-1], l_norm[-1]])

        # Find point with maximum perpendicular distance
        max_dist = 0
        knee_idx = 1

        for i in range(1, len(points) - 1):
            p = np.array([n_norm[i], l_norm[i]])

            # Distance from point to line
            dist = np.abs(np.cross(p2 - p1, p1 - p)) / (np.linalg.norm(p2 - p1) + 1e-10)

            if dist > max_dist:
                max_dist = dist
                knee_idx = i

        return points[knee_idx].n_actions

    def _find_optimal_budget(self, points: List[ParetoPoint]) -> int:
        """
        Encuentra el presupuesto optimo basado en el ratio beneficio/coste.

        Args:
            points: Lista de puntos de Pareto

        Returns:
            Numero optimo de acciones
        """
        best_ratio = 0
        optimal_n = 1

        for p in points[1:]:  # Skip base case
            if p.n_actions > 0:
                ratio = p.improvement_vs_base / p.n_actions
                if ratio > best_ratio:
                    best_ratio = ratio
                    optimal_n = p.n_actions

        return optimal_n

    def get_pareto_table(self, frontier: ParetoFrontier) -> str:
        """
        Genera una tabla formateada con los resultados de Pareto.

        Args:
            frontier: Frontera de Pareto calculada

        Returns:
            String con la tabla formateada
        """
        lines = []
        lines.append("-" * 80)
        lines.append(f"{'N Actions':>10} | {'Lambda*':>10} | {'Improvement':>12} | {'Marginal':>10} | {'Time (s)':>10}")
        lines.append("-" * 80)

        for p in frontier.points:
            marker = ""
            if p.n_actions == frontier.knee_point:
                marker = " <-- KNEE"
            elif p.n_actions == frontier.optimal_n_actions:
                marker = " <-- OPTIMAL"

            lines.append(
                f"{p.n_actions:>10} | "
                f"{p.lambda_max:>10.3f} | "
                f"{p.improvement_vs_base:>+11.1f}% | "
                f"{p.marginal_improvement:>+9.2f}% | "
                f"{p.computation_time:>10.2f}"
                f"{marker}"
            )

        lines.append("-" * 80)
        lines.append(f"Pareto Efficiency: {frontier.pareto_efficiency:.1f}% of benefit with 20% of actions")

        return "\n".join(lines)


def run_pareto_experiment(
    env_name: str = "l2rpn_case14_sandbox",
    max_budget: int = 10,
    lambda_max: float = 2.0,
    output_dir: str = "results"
) -> Dict:
    """
    Ejecuta el experimento completo de Pareto.

    Args:
        env_name: Nombre del entorno Grid2Op
        max_budget: Numero maximo de acciones a probar
        lambda_max: Lambda maximo
        output_dir: Directorio de salida

    Returns:
        Diccionario con resultados
    """
    import os

    print("=" * 60)
    print("PARETO FRONTIER EXPERIMENT")
    print("=" * 60)
    print(f"Environment: {env_name}")
    print(f"Max action budget: {max_budget}")
    print(f"Lambda range: [1.0, {lambda_max}]")
    print()

    # Create environment
    env = grid2op.make(env_name)
    print(f"Network: {env.n_line} lines, {env.n_sub} substations")
    print()

    # Run analysis
    analyzer = ParetoAnalyzer(env)
    frontier = analyzer.compute_pareto_frontier(
        max_budget=max_budget,
        lambda_end=lambda_max,
        verbose=True
    )

    # Print table
    print("\n")
    print(analyzer.get_pareto_table(frontier))

    # Prepare results
    results = {
        'env_name': env_name,
        'n_lines': env.n_line,
        'n_substations': env.n_sub,
        'base_lambda': frontier.base_lambda,
        'max_lambda': frontier.max_lambda,
        'knee_point': frontier.knee_point,
        'optimal_budget': frontier.optimal_n_actions,
        'pareto_efficiency': frontier.pareto_efficiency,
        'points': [
            {
                'n_actions': p.n_actions,
                'lambda_max': p.lambda_max,
                'improvement': p.improvement_vs_base,
                'marginal': p.marginal_improvement,
                'time': p.computation_time
            }
            for p in frontier.points
        ]
    }

    # Save to file
    os.makedirs(output_dir, exist_ok=True)

    import json
    with open(os.path.join(output_dir, 'pareto_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_pareto_experiment()
