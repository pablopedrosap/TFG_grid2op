"""
Visualization Module
====================
Módulo para generar gráficas y visualizaciones de los resultados
del análisis de margen de carga.

Genera las curvas/métricas requeridas:
- Vmin(λ): Tensión mínima vs factor de carga
- ρmax(λ): Sobrecarga máxima vs factor de carga
- Comparativas entre métodos y contingencias

Author: Pablo Pedrosa Prats
TFG - ICAI 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .load_margin import LoadMarginResult


# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

COLORS = {
    'base': '#2E86AB',
    'n1': '#A23B72',
    'greedy': '#F18F01',
    'sensitivity': '#C73E1D',
    'limit': '#E63946',
    'safe': '#2A9D8F'
}


def setup_figure(
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 100
) -> Tuple[plt.Figure, plt.Axes]:
    """Configura una figura con estilo consistente."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_load_margin_curve(
    result: LoadMarginResult,
    title: str = "Load Margin Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera la curva de margen de carga con Vmin y ρmax.

    Args:
        result: Resultado del análisis de margen
        title: Título del gráfico
        save_path: Ruta para guardar la figura

    Returns:
        Figura de matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    lambda_vals = result.lambda_values

    # Gráfica 1: Vmin(λ)
    ax1.plot(lambda_vals, result.v_min_values, 'b-', linewidth=2, label='V_min')
    ax1.plot(lambda_vals, result.v_max_values, 'r--', linewidth=2, label='V_max')

    # Límites de tensión
    ax1.axhline(y=0.9, color=COLORS['limit'], linestyle=':', label='V_min limit (0.9 p.u.)')
    ax1.axhline(y=1.1, color=COLORS['limit'], linestyle=':', label='V_max limit (1.1 p.u.)')

    # Marcar λ*
    ax1.axvline(x=result.lambda_max, color=COLORS['safe'], linestyle='--',
                label=f'λ* = {result.lambda_max:.3f}')

    ax1.set_xlabel('Load Factor λ', fontsize=12)
    ax1.set_ylabel('Voltage (p.u.)', fontsize=12)
    ax1.set_title('Voltage Profile vs Load Factor', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Gráfica 2: ρmax(λ)
    ax2.plot(lambda_vals, result.rho_max_values, 'g-', linewidth=2, label='ρ_max')

    # Límite térmico
    ax2.axhline(y=1.0, color=COLORS['limit'], linestyle=':', label='Thermal limit (100%)')

    # Marcar λ*
    ax2.axvline(x=result.lambda_max, color=COLORS['safe'], linestyle='--',
                label=f'λ* = {result.lambda_max:.3f}')

    ax2.set_xlabel('Load Factor λ', fontsize=12)
    ax2.set_ylabel('Maximum Line Loading ρ', fontsize=12)
    ax2.set_title('Line Loading vs Load Factor', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Formatear eje Y como porcentaje
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_n1_comparison(
    results: Dict[str, LoadMarginResult],
    title: str = "N-1 Contingency Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera gráfica comparativa de contingencias N-1.

    Args:
        results: Diccionario de resultados por contingencia
        title: Título
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extraer datos
    contingencies = []
    lambda_max_values = []
    colors = []

    for name, result in results.items():
        contingencies.append(name)
        lambda_max_values.append(result.lambda_max)

        if name == "base_case":
            colors.append(COLORS['base'])
        else:
            colors.append(COLORS['n1'])

    # Gráfica 1: Bar chart de λ*
    y_pos = np.arange(len(contingencies))
    ax1.barh(y_pos, lambda_max_values, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(contingencies, fontsize=9)
    ax1.set_xlabel('Load Margin λ*', fontsize=12)
    ax1.set_title('Load Margin by Contingency', fontsize=14)

    # Añadir valores
    for i, v in enumerate(lambda_max_values):
        ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    ax1.axvline(x=lambda_max_values[0], color='gray', linestyle='--', alpha=0.5)

    # Gráfica 2: Curvas de ρmax superpuestas
    for name, result in results.items():
        if len(result.lambda_values) > 0:
            label = f"{name} (λ*={result.lambda_max:.2f})"
            if name == "base_case":
                ax2.plot(result.lambda_values, result.rho_max_values,
                        linewidth=2.5, label=label)
            else:
                ax2.plot(result.lambda_values, result.rho_max_values,
                        linewidth=1, alpha=0.6, label=label)

    ax2.axhline(y=1.0, color=COLORS['limit'], linestyle=':', label='Limit')
    ax2.set_xlabel('Load Factor λ', fontsize=12)
    ax2.set_ylabel('Maximum Line Loading ρ', fontsize=12)
    ax2.set_title('Line Loading Evolution', fontsize=14)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_method_comparison(
    base_result: LoadMarginResult,
    greedy_result: Dict,
    sensitivity_result: Dict,
    title: str = "Method Comparison: Base vs Greedy vs Sensitivity",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compara los tres métodos: caso base, Greedy y Sensibilidades.

    Args:
        base_result: Resultado sin control topológico
        greedy_result: Resultado con agente Greedy
        sensitivity_result: Resultado con agente de sensibilidades
        title: Título
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Datos
    methods = ['Base Case', 'Greedy Agent', 'Sensitivity Agent']
    lambda_max = [
        base_result.lambda_max,
        greedy_result.get('lambda_max', 0),
        sensitivity_result.get('lambda_max', 0)
    ]
    colors_bar = [COLORS['base'], COLORS['greedy'], COLORS['sensitivity']]

    # Gráfica 1: Comparación de λ*
    ax1 = axes[0, 0]
    bars = ax1.bar(methods, lambda_max, color=colors_bar, alpha=0.8)
    ax1.set_ylabel('Load Margin λ*', fontsize=12)
    ax1.set_title('Load Margin Comparison', fontsize=14)

    for bar, val in zip(bars, lambda_max):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    # Gráfica 2: Mejora porcentual
    ax2 = axes[0, 1]
    if base_result.lambda_max > 0:
        improvements = [(lm - base_result.lambda_max) / base_result.lambda_max * 100
                       for lm in lambda_max]
    else:
        improvements = [0, 0, 0]

    bars2 = ax2.bar(methods[1:], improvements[1:],
                   color=[COLORS['greedy'], COLORS['sensitivity']], alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Improvement (%)', fontsize=12)
    ax2.set_title('Improvement over Base Case', fontsize=14)

    for bar, val in zip(bars2, improvements[1:]):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f'{val:+.1f}%', ha='center', fontsize=11, fontweight='bold')

    # Gráfica 3: Evolución de ρmax
    ax3 = axes[1, 0]

    ax3.plot(base_result.lambda_values, base_result.rho_max_values,
            color=COLORS['base'], linewidth=2, label='Base Case')

    if greedy_result.get('lambda_values'):
        ax3.plot(greedy_result['lambda_values'], greedy_result['rho_max_values'],
                color=COLORS['greedy'], linewidth=2, label='Greedy')

    if sensitivity_result.get('lambda_values'):
        ax3.plot(sensitivity_result['lambda_values'], sensitivity_result['rho_max_values'],
                color=COLORS['sensitivity'], linewidth=2, label='Sensitivity')

    ax3.axhline(y=1.0, color=COLORS['limit'], linestyle=':', label='Limit')
    ax3.set_xlabel('Load Factor λ', fontsize=12)
    ax3.set_ylabel('Maximum Line Loading ρ', fontsize=12)
    ax3.set_title('Line Loading Evolution', fontsize=14)
    ax3.legend(loc='upper left')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    # Gráfica 4: Número de acciones tomadas
    ax4 = axes[1, 1]

    actions_count = [
        0,
        len(greedy_result.get('actions_taken', [])),
        len(sensitivity_result.get('actions_taken', []))
    ]

    bars4 = ax4.bar(methods, actions_count, color=colors_bar, alpha=0.8)
    ax4.set_ylabel('Number of Actions', fontsize=12)
    ax4.set_title('Corrective Actions Taken', fontsize=14)

    for bar, val in zip(bars4, actions_count):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                str(int(val)), ha='center', fontsize=11, fontweight='bold')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_pv_curve(
    result: LoadMarginResult,
    title: str = "PV Curve (Nose Curve)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera la curva PV (curva de nariz) clásica de análisis de estabilidad.

    La curva PV muestra la tensión en función de la potencia demandada.

    Args:
        result: Resultado del análisis
        title: Título
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, ax = setup_figure(figsize=(10, 7))

    # Calcular potencia total
    # Asumimos que la potencia base es proporcional a λ
    p_values = np.array(result.lambda_values) * 100  # Normalizado a 100%

    ax.plot(p_values, result.v_min_values, 'b-', linewidth=2.5,
           marker='o', markersize=4, label='V_min')

    # Punto de colapso
    collapse_p = result.lambda_max * 100
    collapse_v = result.v_min_values[-1] if result.v_min_values else 0

    ax.scatter([collapse_p], [collapse_v], color=COLORS['limit'],
              s=150, zorder=5, marker='X', label=f'Collapse point (λ*={result.lambda_max:.3f})')

    # Límites
    ax.axhline(y=0.9, color=COLORS['limit'], linestyle=':', alpha=0.7,
              label='V_min limit (0.9 p.u.)')

    # Región segura
    ax.fill_between(p_values, 0.9, result.v_min_values,
                   where=np.array(result.v_min_values) >= 0.9,
                   alpha=0.2, color=COLORS['safe'], label='Safe region')

    ax.set_xlabel('Total Load (% of base case)', fontsize=12)
    ax.set_ylabel('Minimum Voltage (p.u.)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')

    ax.set_xlim(left=95)
    ax.set_ylim(0.85, 1.15)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_summary_table(
    results: Dict[str, LoadMarginResult],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Crea una tabla resumen con los resultados de todas las contingencias.

    Args:
        results: Diccionario de resultados
        save_path: Ruta para guardar CSV

    Returns:
        DataFrame con el resumen
    """
    data = []

    for name, result in results.items():
        row = {
            'Scenario': name,
            'λ*': result.lambda_max,
            'Failure Reason': result.failure_reason or 'Max λ reached',
            'Failure λ': result.failure_lambda,
            'Final V_min (p.u.)': result.v_min_values[-1] if result.v_min_values else None,
            'Final V_max (p.u.)': result.v_max_values[-1] if result.v_max_values else None,
            'Final ρ_max (%)': result.rho_max_values[-1] * 100 if result.rho_max_values else None,
            'Overloaded Lines': len(result.overloaded_lines),
            'Undervoltage Buses': len(result.undervoltage_buses),
        }
        data.append(row)

    df = pd.DataFrame(data)
    df = df.sort_values('λ*', ascending=True)

    if save_path:
        df.to_csv(save_path, index=False)

    return df


def plot_sensitivity_analysis(
    results: List[Dict],
    parameter_name: str,
    parameter_values: List[float],
    title: str = "Sensitivity Analysis",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera gráfica de análisis de sensibilidad para un parámetro.

    Args:
        results: Lista de resultados para cada valor del parámetro
        parameter_name: Nombre del parámetro
        parameter_values: Valores probados
        title: Título
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, ax = setup_figure(figsize=(10, 6))

    lambda_max_values = [r.get('lambda_max', r.lambda_max if hasattr(r, 'lambda_max') else 0)
                        for r in results]

    ax.plot(parameter_values, lambda_max_values, 'o-', linewidth=2,
           markersize=8, color=COLORS['base'])

    ax.set_xlabel(parameter_name, fontsize=12)
    ax.set_ylabel('Load Margin λ*', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Añadir grid más detallado
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def generate_full_report(
    base_results: Dict[str, LoadMarginResult],
    greedy_results: Dict = None,
    sensitivity_results: Dict = None,
    output_dir: str = "results"
) -> None:
    """
    Genera un informe completo con todas las gráficas.

    Args:
        base_results: Resultados del análisis base (caso base + N-1)
        greedy_results: Resultados con agente Greedy
        sensitivity_results: Resultados con agente de sensibilidades
        output_dir: Directorio de salida
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Generando informe completo...")

    # 1. Curva de margen de carga (caso base)
    if "base_case" in base_results:
        fig = plot_load_margin_curve(
            base_results["base_case"],
            title="Base Case - Load Margin Analysis",
            save_path=output_path / "01_load_margin_base.png"
        )
        plt.close(fig)
        print("  - Curva de margen base generada")

    # 2. Curva PV
    if "base_case" in base_results:
        fig = plot_pv_curve(
            base_results["base_case"],
            save_path=output_path / "02_pv_curve.png"
        )
        plt.close(fig)
        print("  - Curva PV generada")

    # 3. Comparación N-1
    fig = plot_n1_comparison(
        base_results,
        save_path=output_path / "03_n1_comparison.png"
    )
    plt.close(fig)
    print("  - Comparación N-1 generada")

    # 4. Comparación de métodos
    if greedy_results and sensitivity_results:
        fig = plot_method_comparison(
            base_results["base_case"],
            greedy_results,
            sensitivity_results,
            save_path=output_path / "04_method_comparison.png"
        )
        plt.close(fig)
        print("  - Comparación de métodos generada")

    # 5. Tabla resumen
    df = create_summary_table(
        base_results,
        save_path=output_path / "05_summary_table.csv"
    )
    print("  - Tabla resumen generada")

    print(f"\nInforme completo guardado en: {output_path}")

    return df


def plot_pareto_frontier(
    pareto_results: Dict,
    title: str = "Pareto Frontier: Load Margin vs Number of Actions",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera la grafica de la frontera de Pareto.

    Muestra el trade-off entre margen de carga (lambda*) y numero de acciones.

    Args:
        pareto_results: Diccionario con resultados del analisis de Pareto
        title: Titulo del grafico
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    points = pareto_results['points']
    n_actions = [p['n_actions'] for p in points]
    lambda_max = [p['lambda_max'] for p in points]
    improvements = [p['improvement'] for p in points]
    marginals = [p['marginal'] for p in points]

    knee_point = pareto_results.get('knee_point', 0)
    optimal_budget = pareto_results.get('optimal_budget', 0)

    # Graph 1: Pareto Frontier (lambda* vs n_actions)
    ax1 = axes[0, 0]
    ax1.plot(n_actions, lambda_max, 'o-', linewidth=2, markersize=8,
             color=COLORS['base'], label='Pareto Frontier')

    # Mark knee point
    knee_idx = next((i for i, p in enumerate(points) if p['n_actions'] == knee_point), 0)
    ax1.scatter([knee_point], [lambda_max[knee_idx]], color=COLORS['limit'],
                s=200, zorder=5, marker='*', label=f'Knee point ({knee_point} actions)')

    # Mark optimal
    opt_idx = next((i for i, p in enumerate(points) if p['n_actions'] == optimal_budget), 0)
    ax1.scatter([optimal_budget], [lambda_max[opt_idx]], color=COLORS['safe'],
                s=200, zorder=5, marker='D', label=f'Optimal ({optimal_budget} actions)')

    ax1.set_xlabel('Number of Actions', fontsize=12)
    ax1.set_ylabel('Load Margin λ*', fontsize=12)
    ax1.set_title('Pareto Frontier', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Graph 2: Improvement vs base case
    ax2 = axes[0, 1]
    bars = ax2.bar(n_actions, improvements, color=COLORS['greedy'], alpha=0.8)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Number of Actions', fontsize=12)
    ax2.set_ylabel('Improvement vs Base (%)', fontsize=12)
    ax2.set_title('Improvement over Base Case', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Highlight knee point
    if knee_idx < len(bars):
        bars[knee_idx].set_color(COLORS['limit'])
    if opt_idx < len(bars):
        bars[opt_idx].set_edgecolor(COLORS['safe'])
        bars[opt_idx].set_linewidth(3)

    # Graph 3: Marginal improvement
    ax3 = axes[1, 0]
    ax3.bar(n_actions[1:], marginals[1:], color=COLORS['sensitivity'], alpha=0.8)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Number of Actions', fontsize=12)
    ax3.set_ylabel('Marginal Improvement (%)', fontsize=12)
    ax3.set_title('Marginal Benefit per Additional Action', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Graph 4: Cumulative improvement (area chart)
    ax4 = axes[1, 1]
    ax4.fill_between(n_actions, 0, improvements, alpha=0.3, color=COLORS['base'])
    ax4.plot(n_actions, improvements, 'o-', linewidth=2, color=COLORS['base'])

    # Mark 20% budget line
    max_budget = max(n_actions)
    twenty_pct = int(max_budget * 0.2) if max_budget > 0 else 1
    ax4.axvline(x=twenty_pct, color=COLORS['limit'], linestyle='--',
                label=f'20% of budget ({twenty_pct} actions)')

    # Calculate and show Pareto efficiency
    pareto_eff = pareto_results.get('pareto_efficiency', 0)
    ax4.text(0.95, 0.05, f'Pareto Efficiency: {pareto_eff:.1f}%\n(benefit with 20% of actions)',
             transform=ax4.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax4.set_xlabel('Number of Actions', fontsize=12)
    ax4.set_ylabel('Cumulative Improvement (%)', fontsize=12)
    ax4.set_title('Cumulative Benefit Analysis', fontsize=14)
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_pareto_with_time(
    pareto_results: Dict,
    title: str = "Pareto Analysis with Computation Time",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera grafica de Pareto incluyendo tiempo de computo.

    Args:
        pareto_results: Diccionario con resultados
        title: Titulo
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    points = pareto_results['points']
    n_actions = [p['n_actions'] for p in points]
    lambda_max = [p['lambda_max'] for p in points]
    times = [p.get('time', 0) for p in points]

    # Graph 1: Lambda vs Actions
    ax1 = axes[0]
    ax1.plot(n_actions, lambda_max, 'o-', linewidth=2, markersize=8, color=COLORS['base'])
    ax1.set_xlabel('Number of Actions', fontsize=12)
    ax1.set_ylabel('Load Margin λ*', fontsize=12)
    ax1.set_title('Lambda* vs Actions', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Graph 2: Time vs Actions
    ax2 = axes[1]
    ax2.bar(n_actions, times, color=COLORS['greedy'], alpha=0.8)
    ax2.set_xlabel('Number of Actions', fontsize=12)
    ax2.set_ylabel('Computation Time (s)', fontsize=12)
    ax2.set_title('Computation Time vs Actions', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Graph 3: Efficiency (lambda per second)
    ax3 = axes[2]
    efficiency = [(lm / t if t > 0 else 0) for lm, t in zip(lambda_max, times)]
    ax3.plot(n_actions, efficiency, 's-', linewidth=2, markersize=8, color=COLORS['sensitivity'])
    ax3.set_xlabel('Number of Actions', fontsize=12)
    ax3.set_ylabel('λ* per second', fontsize=12)
    ax3.set_title('Computational Efficiency', fontsize=14)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_statistical_pareto_frontier(
    stat_results: Dict,
    title: str = "Statistical Pareto Frontier (Paper Quality)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Genera grafica de Pareto con barras de error y intervalos de confianza.

    Esta visualizacion es de calidad de paper: incluye estadisticas
    de multiples runs con intervalos de confianza del 95%.

    Args:
        stat_results: Diccionario con resultados estadisticos
        title: Titulo del grafico
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    points = stat_results['points']
    n_actions = [p['n_actions'] for p in points]
    lambda_mean = [p['lambda_mean'] for p in points]
    lambda_std = [p['lambda_std'] for p in points]
    ci_lower = [p['ci_lower'] for p in points]
    ci_upper = [p['ci_upper'] for p in points]
    improvement_mean = [p['improvement_mean'] for p in points]
    improvement_std = [p['improvement_std'] for p in points]

    summary = stat_results.get('summary', {})
    n_runs = stat_results.get('metadata', {}).get('n_runs', 1)

    # Graph 1: Lambda* with confidence intervals
    ax1 = axes[0, 0]
    ax1.errorbar(n_actions, lambda_mean, yerr=lambda_std, fmt='o-',
                 linewidth=2, markersize=8, capsize=5, capthick=2,
                 color=COLORS['base'], label=f'Mean (n={n_runs} runs)')
    ax1.fill_between(n_actions, ci_lower, ci_upper, alpha=0.2,
                     color=COLORS['base'], label='95% CI')

    # Mark optimal and knee points
    optimal = summary.get('optimal_n_actions', 0)
    knee = summary.get('knee_point_mode', 0)

    if optimal < len(lambda_mean):
        ax1.scatter([optimal], [lambda_mean[optimal]], color=COLORS['safe'],
                    s=200, zorder=5, marker='D', label=f'Optimal ({optimal} actions)')
    if knee < len(lambda_mean):
        ax1.scatter([knee], [lambda_mean[knee]], color=COLORS['limit'],
                    s=200, zorder=5, marker='*', label=f'Knee ({knee} actions)')

    ax1.set_xlabel('Number of Actions', fontsize=12)
    ax1.set_ylabel('Load Margin $\\lambda^*$', fontsize=12)
    ax1.set_title('Pareto Frontier with Statistical Bounds', fontsize=14)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Graph 2: Improvement with error bars
    ax2 = axes[0, 1]
    bars = ax2.bar(n_actions, improvement_mean, yerr=improvement_std,
                   color=COLORS['greedy'], alpha=0.8, capsize=3)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Highlight significant improvements (CI doesn't include 0)
    for i, (imp, std) in enumerate(zip(improvement_mean, improvement_std)):
        if imp - 1.96 * std > 0:  # Significantly positive
            bars[i].set_color(COLORS['safe'])
        elif imp + 1.96 * std < 0:  # Significantly negative
            bars[i].set_color(COLORS['limit'])

    ax2.set_xlabel('Number of Actions', fontsize=12)
    ax2.set_ylabel('Improvement vs Base (%)', fontsize=12)
    ax2.set_title('Improvement with Standard Deviation', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Graph 3: Box plots of all lambdas per action count
    ax3 = axes[1, 0]
    all_lambdas_data = [p.get('all_lambdas', [p['lambda_mean']]) for p in points]

    bp = ax3.boxplot(all_lambdas_data, positions=n_actions, widths=0.6,
                     patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['base'])
        patch.set_alpha(0.6)

    ax3.set_xlabel('Number of Actions', fontsize=12)
    ax3.set_ylabel('Load Margin $\\lambda^*$', fontsize=12)
    ax3.set_title(f'Distribution Across {n_runs} Runs', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Graph 4: Key metrics summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create text summary
    mono_rate = summary.get('monotonicity_rate', 0)
    pareto_eff = summary.get('pareto_efficiency_mean', 0)
    pareto_std = summary.get('pareto_efficiency_std', 0)

    text = f"""
    STATISTICAL SUMMARY
    ═══════════════════════════════════════

    Number of runs: {n_runs}

    Base case λ*: {summary.get('base_lambda_mean', 0):.3f} ± {summary.get('base_lambda_std', 0):.3f}

    Maximum λ*: {summary.get('max_lambda_mean', 0):.3f}

    Optimal budget: {optimal} actions

    Knee point: {knee} actions

    Pareto efficiency: {pareto_eff:.1f}% ± {pareto_std:.1f}%
    (% of max improvement with 20% of actions)

    Monotonicity rate: {mono_rate:.1f}%

    ═══════════════════════════════════════
    """

    ax4.text(0.1, 0.5, text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')  # Higher DPI for paper

    return fig


def plot_monotonicity_analysis(
    stat_results: Dict,
    title: str = "Monotonicity Analysis of Pareto Frontier",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza el analisis de monotonicidad para paper.

    Args:
        stat_results: Resultados estadisticos con analisis
        title: Titulo
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    points = stat_results['points']
    n_actions = [p['n_actions'] for p in points]
    lambda_mean = [p['lambda_mean'] for p in points]

    # Graph 1: Frontier with monotonicity violations highlighted
    ax1 = axes[0]
    ax1.plot(n_actions, lambda_mean, 'o-', linewidth=2, markersize=8,
             color=COLORS['base'], label='Mean λ*')

    # Highlight violations
    violations = stat_results.get('analysis', {}).get('monotonicity', {}).get('violations', [])

    for v in violations:
        from_n = v['from_actions']
        to_n = v['to_actions']
        ax1.annotate('', xy=(to_n, lambda_mean[to_n]),
                     xytext=(from_n, lambda_mean[from_n]),
                     arrowprops=dict(arrowstyle='->', color=COLORS['limit'],
                                    lw=2, connectionstyle='arc3,rad=0.2'))
        ax1.scatter([from_n, to_n], [lambda_mean[from_n], lambda_mean[to_n]],
                   color=COLORS['limit'], s=150, zorder=5, marker='X')

    ax1.set_xlabel('Number of Actions', fontsize=12)
    ax1.set_ylabel('Load Margin λ*', fontsize=12)
    ax1.set_title('Monotonicity Violations in Pareto Frontier', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Add annotation about violations
    if violations:
        ax1.text(0.95, 0.05, f'{len(violations)} violations detected',
                 transform=ax1.transAxes, fontsize=10, color=COLORS['limit'],
                 ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Graph 2: Marginal improvement analysis
    ax2 = axes[1]

    marginal_improvements = []
    for i in range(1, len(lambda_mean)):
        marginal = (lambda_mean[i] - lambda_mean[i-1]) / lambda_mean[0] * 100
        marginal_improvements.append(marginal)

    colors = [COLORS['safe'] if m > 0 else COLORS['limit'] for m in marginal_improvements]
    ax2.bar(n_actions[1:], marginal_improvements, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax2.set_xlabel('Number of Actions', fontsize=12)
    ax2.set_ylabel('Marginal Improvement (%)', fontsize=12)
    ax2.set_title('Marginal Benefit per Additional Action', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Add statistics
    positive_marginals = sum(1 for m in marginal_improvements if m > 0)
    ax2.text(0.95, 0.95, f'{positive_marginals}/{len(marginal_improvements)} positive',
             transform=ax2.transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
