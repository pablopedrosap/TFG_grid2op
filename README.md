# Load Margin Analysis using Grid2Op

**TFG - Trabajo Fin de Grado**
**Pablo Pedrosa Prats**
**ICAI - Universidad Pontificia Comillas**

Directores:
- Javier García González (javiergg@comillas.edu)
- Erik Francisco Álvarez Quispe (erik.alvarez@ri.se)

---

## Key Finding: Non-Monotonicity in Topological Control

> **"More control actions do not always improve the load margin."**

This work demonstrates that the Pareto frontier (λ* vs. number of topological actions) is **non-monotonic**: adding more control actions can *decrease* the load margin. This counterintuitive result was validated across multiple networks with **0% monotonicity rate** in all experiments.

```
         λ*
         │      ●────●
         │     /      \
         │    /        ●────●
         │   ●
         │  /
         │ ●
         └──────────────────── # Actions
              Non-monotonic Pareto frontier
```

**Theoretical contribution**: We develop a predictive framework based on the spectral analysis of the LODF interaction matrix M = LᵀL, with a monotonicity bound η = 1/‖L‖₂ that predicts when violations will occur.

---

## Objective

Approximate the **load margin (λ*)** of a power system by incrementally scaling demand and observing system behavior using the open-source tool [Grid2Op](https://github.com/Grid2op/grid2op).

## Metodología

1. Aplicar un factor de escalado de demanda λ (P y Q)
2. Ejecutar un flujo de potencia para cada paso
3. Registrar variables relevantes (tensión mínima en nudos, cargas de líneas, etc.)
4. El margen λ* se define como el mayor λ para el que el sistema permanece operable

### Criterios de Límite (No Operable)

- No convergencia/inviabilidad del flujo de potencia
- Tensiones fuera de rango (0.9-1.1 p.u.)
- Sobrecargas térmicas por encima del 100%

### Escenarios de Análisis

1. **Caso Base**: Sistema sin contingencias
2. **Contingencias N-1**: Desconexión de líneas representativas
3. **Control Topológico**: Evaluación del impacto de acciones de control topológico sobre el margen de carga

## Project Structure

```
TFG_grid2op/
├── src/
│   ├── __init__.py
│   ├── load_margin.py           # Load margin analysis
│   ├── visualization.py         # Plot generation
│   ├── theoretical_analysis.py  # LODF spectral analysis (NEW)
│   └── agents/
│       ├── __init__.py
│       ├── greedy_agent.py      # Greedy Look-Ahead agent
│       └── sensitivity_agent.py # PTDF/LODF-based agent
├── notebooks/
│   └── 01_load_margin_analysis.ipynb
├── results_paper_ieee14/        # IEEE 14 results (10 runs)
├── results_paper_large/         # Large network results (30 runs)
├── docs/
│   └── memoria_tfg.tex          # Full thesis document
├── run_analysis.py              # Basic analysis script
├── run_paper_experiment.py      # Statistical validation script
├── requirements.txt
└── README.md
```

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/TFG_grid2op.git
cd TFG_grid2op

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Ejecución Rápida

```bash
python run_analysis.py
```

### Opciones Disponibles

```bash
python run_analysis.py --help

Options:
  --env ENV_NAME        Entorno Grid2Op (default: l2rpn_case14_sandbox)
  --agent TYPE          Tipo de agente: 'baseline', 'greedy', 'sensitivity' (default: baseline)
  --lambda-max LAMBDA   Máximo λ a probar (default: 1.5)
  --lambda-step STEP    Incremento de λ (default: 0.01)
  --output DIR          Directorio de salida (default: results)
  --n1-lines N          Líneas a analizar en N-1 (default: 5)
  --max-actions N       Máximo de acciones correctivas (default: 10)
```

### Ejemplos de Uso

```bash
# Análisis básico (solo caso base y N-1)
python run_analysis.py --agent baseline --output results_baseline

# Análisis con agente Greedy (más lento, más exhaustivo)
python run_analysis.py --agent greedy --output results_greedy

# Análisis con agente de Sensibilidades (rápido, basado en PTDF/LODF)
python run_analysis.py --agent sensitivity --output results_sensitivity

# Comparar todos los métodos
python run_analysis.py --agent all --output results_comparison
```

### Usando el Notebook

```bash
cd notebooks
jupyter notebook 01_load_margin_analysis.ipynb
```

## Entregables

- **Código reproducible en Python**
- **Memoria con curvas/métricas comparativas**:
  - Vmin(λ): Tensión mínima vs factor de carga
  - ρmax(λ): Sobrecarga máxima vs factor de carga
  - λ*: Margen de carga para cada escenario

## Métodos Implementados

### 1. Análisis de Margen de Carga Base
Incremento progresivo de la demanda hasta encontrar el punto de colapso.

### 2. Análisis de Contingencias N-1
Evaluación del margen bajo diferentes contingencias (desconexión de líneas).

### 3. Agente Greedy con Look-Ahead (Baseline)
Heurística que evalúa acciones topológicas de forma greedy, simulando N pasos hacia el futuro.

### 4. Agente basado en Sensibilidades (PTDF/LODF)
Utiliza factores de distribución de transferencia de potencia para filtrar el espacio de acciones antes de simular, reduciendo el coste computacional.

## Main Results

### Non-Monotonicity Validation (Statistical)

| Network | Lines | Runs | Monotonicity Rate | Violations |
|---------|-------|------|-------------------|------------|
| IEEE 14 | 20 | 10 | **0%** | 5 |
| Large (IEEE 118-like) | 59 | 30 | **0%** | 4 |

### Load Margin Improvement

| Network | Base λ* | Max λ* | Improvement |
|---------|---------|--------|-------------|
| IEEE 14 | 1.252 ± 0.066 | 1.417 ± 0.115 | +13.2% |
| Large | 1.377 ± 0.201 | 1.662 ± 0.280 | +20.7% |

### Theoretical Analysis (LODF Spectral Properties)

| Metric | IEEE 14 |
|--------|---------|
| Spectral radius ‖L‖₂ | 12.10 |
| Condition number | 20.92 |
| **Monotonicity bound η** | **0.29** |
| Risk level | MEDIUM |

The theoretical bound η = 0.29 correctly predicts that non-monotonicity can appear even with a single action.

### Agent Performance Comparison

| Method | λ* (Margin) | Time | Speedup |
|--------|-------------|------|---------|
| Greedy (Brute Force) | 1.50 | 13.24s | 1.0x |
| Sensitivity (PTDF/LODF) | 1.50 | 8.09s | **1.6x** |

The sensitivity agent achieves **identical accuracy** with **1.6x speedup** on large networks.

## Reproducing the Experiments

```bash
# Install dependencies
pip install -r requirements.txt

# Run paper-quality experiment (IEEE 14, 10 runs)
python run_paper_experiment.py --env l2rpn_case14_sandbox \
                               --n-runs 10 \
                               --budget 10 \
                               --output results_paper_ieee14

# Run large network experiment (30 runs)
python run_paper_experiment.py --env l2rpn_wcci_2022 \
                               --n-runs 30 \
                               --budget 10 \
                               --output results_paper_large
```

## References

- [Grid2Op Documentation](https://grid2op.readthedocs.io/)
- [Grid2Op GitHub](https://github.com/Grid2op/grid2op)
- [L2RPN Competition](https://l2rpn.chalearn.org/)

## Citation

If you use this work, please cite:

```bibtex
@thesis{pedrosa2025loadmargin,
  title={Load Margin Analysis with Topological Control:
         Non-Monotonicity and Theoretical Bounds},
  author={Pedrosa Prats, Pablo},
  school={ICAI - Universidad Pontificia Comillas},
  year={2025},
  type={Bachelor's Thesis}
}
```

## License

This project is part of a Bachelor's Thesis (TFG) at ICAI - Universidad Pontificia Comillas.
