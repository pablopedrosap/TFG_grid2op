"""
Load Margin Analysis Module
===========================
Módulo principal para el cálculo del margen de carga (λ*) en sistemas eléctricos
usando Grid2Op.

El margen de carga λ* se define como el mayor factor de escalado de demanda
para el que el sistema permanece operable.

Criterios de límite (no operable):
    - No convergencia del flujo de potencia
    - Tensiones fuera de rango (0.9-1.1 p.u.)
    - Sobrecargas térmicas > 100%

Author: Pablo Pedrosa Prats
TFG - ICAI 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, field
import grid2op
from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation


@dataclass
class LoadMarginResult:
    """Resultado del análisis de margen de carga."""
    lambda_max: float  # Máximo λ alcanzado (margen de carga)
    lambda_values: List[float] = field(default_factory=list)  # Valores de λ probados
    v_min_values: List[float] = field(default_factory=list)  # Tensión mínima por paso
    v_max_values: List[float] = field(default_factory=list)  # Tensión máxima por paso
    rho_max_values: List[float] = field(default_factory=list)  # Carga máxima de línea por paso
    converged: List[bool] = field(default_factory=list)  # Convergencia por paso
    failure_reason: Optional[str] = None  # Razón del fallo
    failure_lambda: Optional[float] = None  # λ en el que falló
    overloaded_lines: List[int] = field(default_factory=list)  # Líneas sobrecargadas
    undervoltage_buses: List[int] = field(default_factory=list)  # Buses con subtensión


@dataclass
class SystemState:
    """Estado del sistema en un punto de operación."""
    lambda_factor: float
    converged: bool
    v_min: float
    v_max: float
    v_mean: float
    rho_max: float
    rho_mean: float
    total_load_p: float
    total_load_q: float
    total_gen_p: float
    losses: float
    overloaded_lines: List[int]
    undervoltage_buses: List[int]
    overvoltage_buses: List[int]


class LoadMarginAnalyzer:
    """
    Analizador de margen de carga para sistemas eléctricos.

    Incrementa la demanda de forma escalonada y observa el comportamiento
    del sistema hasta encontrar el punto de colapso.

    Attributes:
        env: Entorno Grid2Op
        v_min_pu: Límite inferior de tensión (p.u.)
        v_max_pu: Límite superior de tensión (p.u.)
        rho_max: Límite de carga de línea (1.0 = 100%)
    """

    def __init__(
        self,
        env: Environment,
        v_min_pu: float = 0.9,
        v_max_pu: float = 1.1,
        rho_max: float = 1.0
    ):
        """
        Inicializa el analizador.

        Args:
            env: Entorno Grid2Op configurado
            v_min_pu: Tensión mínima permitida (p.u.)
            v_max_pu: Tensión máxima permitida (p.u.)
            rho_max: Carga máxima permitida en líneas (1.0 = 100%)
        """
        self.env = env
        self.v_min_pu = v_min_pu
        self.v_max_pu = v_max_pu
        self.rho_max = rho_max

        # Obtener voltaje nominal para convertir a p.u.
        self.v_nominal = self._get_nominal_voltages()

    def _get_nominal_voltages(self) -> Dict[int, float]:
        """
        Obtiene los voltajes nominales de cada subestación.

        En Grid2Op, los voltajes nominales se pueden obtener de la
        observación inicial o del backend.
        """
        obs = self.env.reset()

        # Obtener voltajes nominales por subestación
        # Usamos los voltajes de generadores como referencia (suelen estar a 1.0 p.u.)
        nominal_voltages = {}

        # Obtener información de subestaciones desde la topología
        # Los voltajes en Grid2Op están en kV
        # Para l2rpn_case14_sandbox, los niveles típicos son ~20kV y ~100-150kV

        # Usamos los voltajes iniciales de los generadores como referencia
        for gen_id in range(obs.n_gen):
            sub_id = obs.gen_to_subid[gen_id]
            if sub_id not in nominal_voltages:
                v_gen = obs.gen_v[gen_id]  # Voltaje del generador en kV
                if v_gen > 0:
                    nominal_voltages[sub_id] = v_gen

        # Para las subestaciones sin generador, usar interpolación
        # basada en voltajes de líneas conectadas
        for sub_id in range(obs.n_sub):
            if sub_id not in nominal_voltages:
                # Buscar voltaje de líneas conectadas a esta subestación
                for line_id in range(obs.n_line):
                    if obs.line_or_to_subid[line_id] == sub_id and obs.v_or[line_id] > 0:
                        nominal_voltages[sub_id] = obs.v_or[line_id]
                        break
                    elif obs.line_ex_to_subid[line_id] == sub_id and obs.v_ex[line_id] > 0:
                        nominal_voltages[sub_id] = obs.v_ex[line_id]
                        break

        return nominal_voltages

    def _get_voltages_pu(self, obs: BaseObservation) -> np.ndarray:
        """
        Obtiene todos los voltajes de la red en p.u.

        En Grid2Op, los voltajes se devuelven en kV. Para convertir a p.u.,
        dividimos por el voltaje nominal de cada subestación.

        Args:
            obs: Observación actual

        Returns:
            Array con voltajes en p.u.
        """
        v_pu_list = []

        # Voltajes en el origen de las líneas
        for line_id in range(obs.n_line):
            if obs.line_status[line_id] and obs.v_or[line_id] > 0:
                sub_id = obs.line_or_to_subid[line_id]
                v_nominal = self.v_nominal.get(sub_id, obs.v_or[line_id])
                if v_nominal > 0:
                    v_pu = obs.v_or[line_id] / v_nominal
                    v_pu_list.append(v_pu)

        # Voltajes en el extremo de las líneas
        for line_id in range(obs.n_line):
            if obs.line_status[line_id] and obs.v_ex[line_id] > 0:
                sub_id = obs.line_ex_to_subid[line_id]
                v_nominal = self.v_nominal.get(sub_id, obs.v_ex[line_id])
                if v_nominal > 0:
                    v_pu = obs.v_ex[line_id] / v_nominal
                    v_pu_list.append(v_pu)

        if len(v_pu_list) == 0:
            return np.array([1.0])

        return np.array(v_pu_list)

    def _check_voltage_limits(self, obs: BaseObservation) -> Tuple[bool, List[int], List[int]]:
        """
        Verifica si los voltajes están dentro de límites.

        Returns:
            Tuple: (dentro_de_limites, buses_subtension, buses_sobretension)
        """
        v_pu = self._get_voltages_pu(obs)

        undervoltage = np.where(v_pu < self.v_min_pu)[0].tolist()
        overvoltage = np.where(v_pu > self.v_max_pu)[0].tolist()

        within_limits = len(undervoltage) == 0 and len(overvoltage) == 0

        return within_limits, undervoltage, overvoltage

    def _check_thermal_limits(self, obs: BaseObservation) -> Tuple[bool, List[int]]:
        """
        Verifica si las líneas están dentro de límites térmicos.

        Returns:
            Tuple: (dentro_de_limites, lineas_sobrecargadas)
        """
        rho = obs.rho
        overloaded = np.where(rho > self.rho_max)[0].tolist()
        within_limits = len(overloaded) == 0

        return within_limits, overloaded

    def _get_system_state(
        self,
        obs: BaseObservation,
        lambda_factor: float,
        converged: bool
    ) -> SystemState:
        """Obtiene el estado completo del sistema."""
        v_pu = self._get_voltages_pu(obs)
        _, undervoltage, overvoltage = self._check_voltage_limits(obs)
        _, overloaded = self._check_thermal_limits(obs)

        return SystemState(
            lambda_factor=lambda_factor,
            converged=converged,
            v_min=float(np.min(v_pu)) if len(v_pu) > 0 else 0.0,
            v_max=float(np.max(v_pu)) if len(v_pu) > 0 else 0.0,
            v_mean=float(np.mean(v_pu)) if len(v_pu) > 0 else 0.0,
            rho_max=float(np.max(obs.rho)) if len(obs.rho) > 0 else 0.0,
            rho_mean=float(np.mean(obs.rho)) if len(obs.rho) > 0 else 0.0,
            total_load_p=float(np.sum(obs.load_p)),
            total_load_q=float(np.sum(obs.load_q)),
            total_gen_p=float(np.sum(obs.gen_p)),
            losses=float(np.sum(obs.gen_p) - np.sum(obs.load_p)),
            overloaded_lines=overloaded,
            undervoltage_buses=undervoltage,
            overvoltage_buses=overvoltage
        )

    def calculate_load_margin(
        self,
        lambda_start: float = 1.0,
        lambda_end: float = 2.0,
        lambda_step: float = 0.01,
        action=None,
        verbose: bool = True
    ) -> LoadMarginResult:
        """
        Calcula el margen de carga incrementando λ hasta el colapso.

        El método escala la demanda (P y Q) por un factor λ y ejecuta
        un flujo de potencia para cada paso, registrando variables
        relevantes.

        Args:
            lambda_start: Factor de escala inicial
            lambda_end: Factor de escala máximo a probar
            lambda_step: Incremento de λ entre pasos
            action: Acción topológica a aplicar (None = do nothing)
            verbose: Si True, imprime progreso

        Returns:
            LoadMarginResult con el margen encontrado y métricas
        """
        # Resetear el entorno
        obs = self.env.reset()

        # Obtener el simulador para poder modificar cargas
        simulator = obs.get_simulator()

        # Guardar valores iniciales
        init_load_p = obs.load_p.copy()
        init_load_q = obs.load_q.copy()
        init_gen_p = obs.gen_p.copy()

        # Inicializar resultado
        result = LoadMarginResult(lambda_max=lambda_start)

        # Generar valores de λ
        lambda_values = np.arange(lambda_start, lambda_end + lambda_step, lambda_step)

        if verbose:
            print(f"Calculando margen de carga de λ={lambda_start:.2f} a λ={lambda_end:.2f}")
            print(f"Límites: V=[{self.v_min_pu:.2f}, {self.v_max_pu:.2f}] p.u., ρ_max={self.rho_max:.0%}")
            print("-" * 60)

        last_valid_lambda = lambda_start

        for lam in lambda_values:
            # Escalar cargas y generación
            new_load_p = init_load_p * lam
            new_load_q = init_load_q * lam
            new_gen_p = init_gen_p * lam

            # Simular con el factor de escala
            try:
                if action is None:
                    action = self.env.action_space()  # Do nothing

                sim_result = simulator.predict(
                    action,
                    new_gen_p=new_gen_p,
                    new_load_p=new_load_p,
                    new_load_q=new_load_q
                )

                converged = sim_result.converged
                sim_obs = sim_result.current_obs

            except Exception as e:
                if verbose:
                    print(f"λ={lam:.3f}: Error en simulación - {str(e)}")
                result.failure_reason = f"Simulation error: {str(e)}"
                result.failure_lambda = lam
                break

            # Verificar convergencia
            if not converged:
                if verbose:
                    print(f"λ={lam:.3f}: No convergencia del flujo de potencia")
                result.failure_reason = "Power flow did not converge"
                result.failure_lambda = lam
                break

            # Obtener estado del sistema
            state = self._get_system_state(sim_obs, lam, converged)

            # Registrar métricas
            result.lambda_values.append(lam)
            result.v_min_values.append(state.v_min)
            result.v_max_values.append(state.v_max)
            result.rho_max_values.append(state.rho_max)
            result.converged.append(converged)

            # Verificar límites de tensión
            v_ok, undervoltage, overvoltage = self._check_voltage_limits(sim_obs)
            if not v_ok:
                if verbose:
                    print(f"λ={lam:.3f}: Violación de tensión - "
                          f"V_min={state.v_min:.3f}, V_max={state.v_max:.3f} p.u.")
                result.failure_reason = "Voltage limits violated"
                result.failure_lambda = lam
                result.undervoltage_buses = undervoltage
                break

            # Verificar límites térmicos
            thermal_ok, overloaded = self._check_thermal_limits(sim_obs)
            if not thermal_ok:
                if verbose:
                    print(f"λ={lam:.3f}: Sobrecarga térmica - "
                          f"ρ_max={state.rho_max:.1%}, líneas: {overloaded}")
                result.failure_reason = "Thermal limits violated"
                result.failure_lambda = lam
                result.overloaded_lines = overloaded
                break

            # Este λ es válido
            last_valid_lambda = lam

            if verbose and (lam * 100) % 5 == 0:  # Imprimir cada 5%
                print(f"λ={lam:.3f}: OK - V=[{state.v_min:.3f}, {state.v_max:.3f}], "
                      f"ρ_max={state.rho_max:.1%}")

        result.lambda_max = last_valid_lambda

        if verbose:
            print("-" * 60)
            print(f"Margen de carga λ* = {result.lambda_max:.3f}")
            if result.failure_reason:
                print(f"Razón de fallo: {result.failure_reason}")

        return result

    def analyze_contingency(
        self,
        line_id: int,
        lambda_start: float = 1.0,
        lambda_end: float = 2.0,
        lambda_step: float = 0.01,
        verbose: bool = True
    ) -> LoadMarginResult:
        """
        Analiza el margen de carga bajo contingencia N-1 (línea desconectada).

        Args:
            line_id: ID de la línea a desconectar
            lambda_start: Factor de escala inicial
            lambda_end: Factor de escala máximo
            lambda_step: Incremento de λ
            verbose: Imprimir progreso

        Returns:
            LoadMarginResult para la contingencia
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"CONTINGENCIA N-1: Desconexión de línea {line_id}")
            print(f"{'='*60}")

        # Crear acción de desconexión
        disconnect_action = self.env.action_space({"set_line_status": [(line_id, -1)]})

        return self.calculate_load_margin(
            lambda_start=lambda_start,
            lambda_end=lambda_end,
            lambda_step=lambda_step,
            action=disconnect_action,
            verbose=verbose
        )

    def run_n1_analysis(
        self,
        line_ids: Optional[List[int]] = None,
        lambda_end: float = 2.0,
        lambda_step: float = 0.01,
        verbose: bool = True
    ) -> Dict[str, LoadMarginResult]:
        """
        Ejecuta análisis N-1 completo para múltiples contingencias.

        Args:
            line_ids: Lista de IDs de líneas a analizar (None = todas)
            lambda_end: Factor de escala máximo
            lambda_step: Incremento de λ
            verbose: Imprimir progreso

        Returns:
            Diccionario con resultados por contingencia
        """
        obs = self.env.reset()

        if line_ids is None:
            line_ids = list(range(obs.n_line))

        results = {}

        # Primero el caso base
        if verbose:
            print("\n" + "="*60)
            print("CASO BASE (Sin contingencias)")
            print("="*60)

        results["base_case"] = self.calculate_load_margin(
            lambda_end=lambda_end,
            lambda_step=lambda_step,
            verbose=verbose
        )

        # Luego cada contingencia
        for line_id in line_ids:
            results[f"N-1_line_{line_id}"] = self.analyze_contingency(
                line_id=line_id,
                lambda_end=lambda_end,
                lambda_step=lambda_step,
                verbose=verbose
            )

        return results


def create_results_dataframe(results: Dict[str, LoadMarginResult]) -> pd.DataFrame:
    """
    Crea un DataFrame con los resultados del análisis N-1.

    Args:
        results: Diccionario de resultados

    Returns:
        DataFrame con métricas comparativas
    """
    data = []

    for name, result in results.items():
        data.append({
            "Contingency": name,
            "λ*": result.lambda_max,
            "Failure Reason": result.failure_reason or "Max λ reached",
            "Failure λ": result.failure_lambda,
            "Final V_min": result.v_min_values[-1] if result.v_min_values else None,
            "Final ρ_max": result.rho_max_values[-1] if result.rho_max_values else None,
            "Overloaded Lines": result.overloaded_lines,
        })

    df = pd.DataFrame(data)
    df = df.sort_values("λ*", ascending=True)

    return df
