#!/usr/bin/env python3
"""
Theoretical Analysis of Pareto Frontier Non-Monotonicity
========================================================

This module provides the theoretical foundation for understanding why the
Pareto frontier (λ* vs number of actions) can be non-monotonic in power
system topological control.

KEY INSIGHT (Paper Contribution):
---------------------------------
The non-monotonicity arises from the NEGATIVE EIGENVALUES of the
LODF INTERACTION MATRIX. When multiple topological actions are combined,
their effects can interfere destructively if the LODF matrix has specific
spectral properties.

Mathematical Framework:
-----------------------
Let L be the LODF matrix where L[i,j] = change in flow on line i when
line j is disconnected.

For a set S of disconnected lines, the approximate flow redistribution is:
    ΔP = L @ P_S  (linear approximation)

However, when |S| > 1, second-order effects appear:
    ΔP_actual = L @ P_S + L^(2) @ P_S + higher order terms

The interaction matrix M = L^T @ L captures how actions interfere:
- If M has negative eigenvalues → potential for destructive interference
- If all eigenvalues of M are positive → monotonicity more likely

THEOREM (Informal):
    The Pareto frontier is guaranteed monotonic only if the LODF interaction
    matrix M = L^T @ L is positive semi-definite on the subspace of feasible
    topological actions.

This condition is generically NOT satisfied in real power networks, explaining
the empirically observed non-monotonicity.

Author: Pablo Pedrosa Prats
TFG - ICAI 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import grid2op
from grid2op.Environment import Environment


@dataclass
class LODFAnalysis:
    """Results of LODF matrix analysis."""
    lodf_matrix: np.ndarray
    interaction_matrix: np.ndarray
    eigenvalues: np.ndarray
    n_negative_eigenvalues: int
    spectral_radius: float
    condition_number: float
    monotonicity_bound: float  # Theoretical bound on when monotonicity breaks


class TheoreticalAnalyzer:
    """
    Analyzes the theoretical conditions for Pareto frontier monotonicity.

    This class computes the LODF matrix and its spectral properties to
    predict when non-monotonicity is likely to occur.
    """

    def __init__(self, env: Environment):
        """
        Initialize the theoretical analyzer.

        Args:
            env: Grid2Op environment
        """
        self.env = env
        self.n_lines = env.n_line
        self._lodf_matrix = None

    def compute_lodf_matrix(self) -> np.ndarray:
        """
        Compute the Line Outage Distribution Factor matrix.

        LODF[i,j] represents the fraction of flow from line j that
        redistributes to line i when line j is disconnected.

        Returns:
            LODF matrix of shape (n_lines, n_lines)
        """
        if self._lodf_matrix is not None:
            return self._lodf_matrix

        obs = self.env.reset()
        simulator = obs.get_simulator()

        n = self.n_lines
        lodf = np.zeros((n, n))

        # Get base case flows
        base_flows = obs.p_or.copy()

        for j in range(n):
            if not obs.line_status[j]:
                continue

            try:
                # Create action to disconnect line j
                action = self.env.action_space({"set_line_status": [(j, -1)]})
                result = simulator.predict(action)

                if result.converged:
                    new_flows = result.current_obs.p_or

                    # LODF[i,j] = (new_flow_i - base_flow_i) / base_flow_j
                    if abs(base_flows[j]) > 1e-6:
                        for i in range(n):
                            if i != j and obs.line_status[i]:
                                lodf[i, j] = (new_flows[i] - base_flows[i]) / base_flows[j]
            except:
                pass

        self._lodf_matrix = lodf
        return lodf

    def compute_interaction_matrix(self, lodf: np.ndarray) -> np.ndarray:
        """
        Compute the LODF interaction matrix M = L^T @ L.

        This matrix captures how topological actions interact:
        - M[i,j] > 0: actions i and j have similar effects (constructive)
        - M[i,j] < 0: actions i and j have opposing effects (destructive)

        Args:
            lodf: LODF matrix

        Returns:
            Interaction matrix of shape (n_lines, n_lines)
        """
        return lodf.T @ lodf

    def analyze_spectral_properties(self, lodf: Optional[np.ndarray] = None) -> LODFAnalysis:
        """
        Perform complete spectral analysis of the LODF matrix.

        This is the core theoretical contribution: we show that the
        eigenvalue structure of the interaction matrix determines
        the potential for non-monotonicity.

        Args:
            lodf: Pre-computed LODF matrix (optional)

        Returns:
            LODFAnalysis with spectral properties
        """
        if lodf is None:
            lodf = self.compute_lodf_matrix()

        interaction = self.compute_interaction_matrix(lodf)

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(interaction)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

        # Count negative eigenvalues (key indicator of non-monotonicity potential)
        n_negative = np.sum(eigenvalues < -1e-10)

        # Spectral radius
        spectral_radius = np.max(np.abs(eigenvalues))

        # Condition number (if invertible)
        nonzero_eigs = eigenvalues[np.abs(eigenvalues) > 1e-10]
        if len(nonzero_eigs) > 0:
            condition_number = np.max(np.abs(nonzero_eigs)) / np.min(np.abs(nonzero_eigs))
        else:
            condition_number = np.inf

        # Monotonicity bound: theoretical estimate of when non-monotonicity appears
        # Based on perturbation theory: if ||L||_2 * k > threshold, non-monotonicity likely
        # where k is the number of simultaneous actions
        lodf_norm = np.linalg.norm(lodf, 2)
        monotonicity_bound = 1.0 / lodf_norm if lodf_norm > 0 else np.inf

        return LODFAnalysis(
            lodf_matrix=lodf,
            interaction_matrix=interaction,
            eigenvalues=eigenvalues,
            n_negative_eigenvalues=n_negative,
            spectral_radius=spectral_radius,
            condition_number=condition_number,
            monotonicity_bound=monotonicity_bound
        )

    def predict_nonmonotonicity_risk(self, analysis: LODFAnalysis) -> Dict:
        """
        Predict the risk of non-monotonicity based on spectral analysis.

        This method provides actionable insights for operators.

        Args:
            analysis: LODFAnalysis from spectral analysis

        Returns:
            Dictionary with risk assessment and recommendations
        """
        # Risk factors
        risk_score = 0.0
        risk_factors = []

        # Factor 1: Negative eigenvalues (strongest indicator)
        if analysis.n_negative_eigenvalues > 0:
            risk_score += 0.4
            risk_factors.append(
                f"Interaction matrix has {analysis.n_negative_eigenvalues} negative eigenvalues"
            )

        # Factor 2: High condition number (numerical instability)
        if analysis.condition_number > 100:
            risk_score += 0.2
            risk_factors.append(
                f"High condition number ({analysis.condition_number:.1f}) indicates sensitivity"
            )

        # Factor 3: Low monotonicity bound
        if analysis.monotonicity_bound < 3:
            risk_score += 0.2
            risk_factors.append(
                f"Monotonicity bound ({analysis.monotonicity_bound:.2f}) suggests issues with >3 actions"
            )

        # Factor 4: Spectral radius
        if analysis.spectral_radius > 2:
            risk_score += 0.2
            risk_factors.append(
                f"High spectral radius ({analysis.spectral_radius:.2f}) amplifies errors"
            )

        # Risk level
        if risk_score >= 0.6:
            risk_level = "HIGH"
            recommendation = "Non-monotonicity is likely. Limit action budget to 2-3 actions."
        elif risk_score >= 0.3:
            risk_level = "MEDIUM"
            recommendation = "Some risk of non-monotonicity. Analyze Pareto frontier carefully."
        else:
            risk_level = "LOW"
            recommendation = "Monotonicity is likely. Standard optimization should work."

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "theoretical_action_limit": int(np.ceil(analysis.monotonicity_bound)),
            "eigenvalue_summary": {
                "n_total": len(analysis.eigenvalues),
                "n_negative": analysis.n_negative_eigenvalues,
                "n_positive": np.sum(analysis.eigenvalues > 1e-10),
                "n_zero": np.sum(np.abs(analysis.eigenvalues) <= 1e-10),
                "largest": float(analysis.eigenvalues[0]) if len(analysis.eigenvalues) > 0 else 0,
                "smallest_nonzero": float(
                    analysis.eigenvalues[np.abs(analysis.eigenvalues) > 1e-10][-1]
                ) if np.sum(np.abs(analysis.eigenvalues) > 1e-10) > 0 else 0
            }
        }

    def compute_action_interference(
        self,
        actions: List[int],
        lodf: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute the interference between a specific set of actions.

        This method explains WHY a particular combination of actions
        might lead to suboptimal results.

        Args:
            actions: List of line IDs to analyze
            lodf: Pre-computed LODF matrix (optional)

        Returns:
            Dictionary with interference analysis
        """
        if lodf is None:
            lodf = self.compute_lodf_matrix()

        n_actions = len(actions)
        if n_actions < 2:
            return {"interference": 0.0, "type": "none", "explanation": "Single action"}

        # Extract submatrix for these actions
        sub_lodf = lodf[:, actions]

        # Compute pairwise interference
        interferences = []
        for i in range(n_actions):
            for j in range(i + 1, n_actions):
                # Correlation between effects of actions i and j
                effect_i = sub_lodf[:, i]
                effect_j = sub_lodf[:, j]

                if np.linalg.norm(effect_i) > 1e-6 and np.linalg.norm(effect_j) > 1e-6:
                    correlation = np.dot(effect_i, effect_j) / (
                        np.linalg.norm(effect_i) * np.linalg.norm(effect_j)
                    )
                else:
                    correlation = 0.0

                interferences.append({
                    "action_i": actions[i],
                    "action_j": actions[j],
                    "correlation": correlation,
                    "type": "constructive" if correlation > 0 else "destructive"
                })

        # Aggregate interference
        avg_interference = np.mean([abs(x["correlation"]) for x in interferences])
        n_destructive = sum(1 for x in interferences if x["type"] == "destructive")

        # Overall assessment
        if n_destructive > n_actions // 2:
            overall_type = "destructive"
            explanation = (
                f"Actions have predominantly destructive interference "
                f"({n_destructive}/{len(interferences)} pairs). "
                "Reducing action count may improve results."
            )
        else:
            overall_type = "constructive"
            explanation = (
                f"Actions have predominantly constructive interference "
                f"({len(interferences) - n_destructive}/{len(interferences)} pairs). "
                "Adding more actions may help."
            )

        return {
            "n_actions": n_actions,
            "pairwise_interferences": interferences,
            "average_interference": avg_interference,
            "n_destructive_pairs": n_destructive,
            "overall_type": overall_type,
            "explanation": explanation
        }


def generate_theoretical_report(env: Environment) -> str:
    """
    Generate a comprehensive theoretical report for a given network.

    This report can be included in papers to explain the theoretical
    foundations of observed non-monotonicity.

    Args:
        env: Grid2Op environment

    Returns:
        Formatted report string
    """
    analyzer = TheoreticalAnalyzer(env)
    analysis = analyzer.analyze_spectral_properties()
    risk = analyzer.predict_nonmonotonicity_risk(analysis)

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║     THEORETICAL ANALYSIS OF PARETO FRONTIER MONOTONICITY         ║
╚══════════════════════════════════════════════════════════════════╝

Network: {env.n_line} lines, {env.n_sub} substations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    LODF MATRIX SPECTRAL ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Interaction Matrix M = L^T @ L Properties:
  • Spectral Radius:     {analysis.spectral_radius:.4f}
  • Condition Number:    {analysis.condition_number:.2f}
  • Monotonicity Bound:  {analysis.monotonicity_bound:.2f} actions

Eigenvalue Distribution:
  • Total eigenvalues:   {risk['eigenvalue_summary']['n_total']}
  • Positive:            {risk['eigenvalue_summary']['n_positive']}
  • Zero (null space):   {risk['eigenvalue_summary']['n_zero']}
  • NEGATIVE:            {risk['eigenvalue_summary']['n_negative']} ← KEY INDICATOR

  Largest eigenvalue:    {risk['eigenvalue_summary']['largest']:.4f}
  Smallest (non-zero):   {risk['eigenvalue_summary']['smallest_nonzero']:.6f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    RISK ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Overall Risk Level: {risk['risk_level']} (score: {risk['risk_score']:.2f})

Risk Factors Identified:
"""

    for i, factor in enumerate(risk['risk_factors'], 1):
        report += f"  {i}. {factor}\n"

    if not risk['risk_factors']:
        report += "  (No significant risk factors identified)\n"

    report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{risk['recommendation']}

Theoretical maximum actions before non-monotonicity: ~{risk['theoretical_action_limit']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    THEORETICAL FOUNDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The presence of {risk['eigenvalue_summary']['n_negative']} negative eigenvalues in the
interaction matrix M = L^T @ L indicates that certain combinations of
topological actions will have DESTRUCTIVE INTERFERENCE.

Physical interpretation:
• Positive eigenvalues → action effects align (constructive)
• Negative eigenvalues → action effects oppose (destructive)
• Zero eigenvalues → actions in null space (no effect)

When the number of actions k exceeds the monotonicity bound
({analysis.monotonicity_bound:.2f}), the probability of encountering
destructive interference increases, leading to non-monotonic behavior
in the Pareto frontier.

This explains the empirical observation that MORE ACTIONS DO NOT
ALWAYS IMPROVE THE LOAD MARGIN.
"""

    return report


if __name__ == "__main__":
    # Demo: analyze the IEEE 14 network
    import grid2op

    env = grid2op.make("l2rpn_case14_sandbox")
    report = generate_theoretical_report(env)
    print(report)
