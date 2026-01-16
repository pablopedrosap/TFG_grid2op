"""
Sensitivity-Based Agent (PTDF/LODF)
===================================
Agente que utiliza factores de distribución de transferencia de potencia
(PTDF) y factores de distribución de desconexión de líneas (LODF) para
filtrar el espacio de acciones antes de la simulación.

Este enfoque "Physics-Informed" reduce drásticamente el número de
simulaciones necesarias al pre-calcular analíticamente el impacto
de las acciones topológicas.

Fundamento matemático:
- PTDF: Sensibilidad del flujo de potencia en una línea respecto a
        inyecciones de potencia en los nodos.
- LODF: Redistribución de flujo cuando una línea se desconecta.

Author: Pablo Pedrosa Prats
TFG - ICAI 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import grid2op
from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction
from grid2op.Observation import BaseObservation
from grid2op.Environment import Environment


@dataclass
class SensitivityResult:
    """Resultado del análisis de sensibilidad para una acción."""
    action: BaseAction
    description: str
    theoretical_impact: float  # Impacto teórico calculado
    lines_relieved: List[int]  # Líneas que se alivian
    lines_worsened: List[int]  # Líneas que empeoran
    priority_score: float  # Puntuación de prioridad


class NetworkSensitivityAnalyzer:
    """
    Calcula y gestiona las matrices de sensibilidad de la red.

    PTDF (Power Transfer Distribution Factors):
        - Matriz que indica cómo se distribuye una inyección de potencia
          entre las líneas de la red.
        - PTDF[l,n] = cambio en flujo de línea l por MW inyectado en nodo n

    LODF (Line Outage Distribution Factors):
        - Matriz que indica cómo se redistribuye el flujo cuando una línea
          se desconecta.
        - LODF[l,k] = factor de redistribución del flujo de línea k
                      hacia línea l cuando k se desconecta

    Estas matrices se calculan a partir de la matriz de admitancia
    de la red (modelo DC simplificado).
    """

    def __init__(self, env: Environment):
        """
        Inicializa el analizador de sensibilidades.

        Args:
            env: Entorno Grid2Op
        """
        self.env = env
        self.n_line = None
        self.n_sub = None
        self.ptdf = None
        self.lodf = None
        self._initialized = False

    def initialize(self, obs: BaseObservation):
        """
        Inicializa las matrices de sensibilidad desde una observación.

        Args:
            obs: Observación inicial del entorno
        """
        self.n_line = obs.n_line
        self.n_sub = obs.n_sub

        # Calcular matrices de sensibilidad
        self._compute_ptdf(obs)
        self._compute_lodf(obs)

        self._initialized = True

    def _compute_ptdf(self, obs: BaseObservation):
        """
        Calcula la matriz PTDF (Power Transfer Distribution Factors).

        El PTDF se calcula usando la aproximación DC:
        PTDF = B_line * X_bus

        donde:
        - B_line: matriz de susceptancia de líneas
        - X_bus: inversa de la matriz B de bus (reactancia)

        Para simplificar, usamos una aproximación basada en la
        topología actual de la red.
        """
        # Obtener la matriz de conectividad
        # En Grid2Op, podemos aproximar el PTDF usando los flujos actuales

        # Aproximación simplificada basada en reactancias
        n_lines = obs.n_line
        n_buses = obs.n_sub

        # Inicializar PTDF con valores aproximados
        # En una implementación completa, se calcularía desde los parámetros de línea
        self.ptdf = np.zeros((n_lines, n_buses))

        # Usar la información de flujos para estimar sensibilidades
        # Los flujos actuales dan una indicación de la distribución
        p_or = obs.p_or
        total_gen = np.sum(obs.gen_p)

        if total_gen > 0:
            # Normalizar flujos como aproximación de PTDF
            for l in range(n_lines):
                if obs.line_status[l]:
                    # Distribuir proporcionalmente a los flujos
                    self.ptdf[l, :] = p_or[l] / total_gen

    def _compute_lodf(self, obs: BaseObservation):
        """
        Calcula la matriz LODF (Line Outage Distribution Factors).

        LODF indica cómo se redistribuye el flujo cuando una línea
        se desconecta. Se calcula como:

        LODF[l,k] = PTDF[l,i] - PTDF[l,j] / (1 - PTDF[k,i] + PTDF[k,j])

        donde i,j son los nodos de la línea k.

        Para simplificar, usamos una aproximación basada en la
        impedancia relativa de las líneas paralelas.
        """
        n_lines = obs.n_line

        # Inicializar LODF
        self.lodf = np.zeros((n_lines, n_lines))

        # Calcular LODF aproximado
        # Cuando una línea k se desconecta, su flujo se redistribuye
        # proporcionalmente a la impedancia de las líneas paralelas

        rho = obs.rho.copy()
        rho[rho == 0] = 0.001  # Evitar división por cero

        for k in range(n_lines):
            if not obs.line_status[k]:
                continue

            # El flujo de k se redistribuye a otras líneas
            # Aproximación: proporcionalmente al flujo actual normalizado
            for l in range(n_lines):
                if l == k or not obs.line_status[l]:
                    continue

                # Factor de redistribución simplificado
                # En una implementación real, esto vendría de la matriz de admitancia
                self.lodf[l, k] = rho[l] / np.sum(rho[obs.line_status])

    def estimate_disconnection_impact(
        self,
        line_to_disconnect: int,
        obs: BaseObservation
    ) -> Dict[int, float]:
        """
        Estima el impacto de desconectar una línea en las demás.

        Args:
            line_to_disconnect: ID de la línea a desconectar
            obs: Observación actual

        Returns:
            Diccionario {línea: nuevo_flujo_estimado}
        """
        if not self._initialized:
            self.initialize(obs)

        impacts = {}
        p_or = obs.p_or
        flow_disconnected = p_or[line_to_disconnect]

        for l in range(self.n_line):
            if l == line_to_disconnect or not obs.line_status[l]:
                continue

            # Nuevo flujo = flujo actual + LODF * flujo de la línea desconectada
            lodf_factor = self.lodf[l, line_to_disconnect]
            new_flow = p_or[l] + lodf_factor * flow_disconnected
            impacts[l] = new_flow

        return impacts

    def rank_disconnection_actions(
        self,
        overloaded_line: int,
        obs: BaseObservation,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Rankea las líneas cuya desconexión más aliviaría una sobrecarga.

        Usa LODF para calcular analíticamente qué desconexiones
        tienen mayor impacto negativo (reducen flujo) en la línea
        sobrecargada.

        Args:
            overloaded_line: Línea con sobrecarga
            obs: Observación actual
            top_k: Número de mejores candidatas a retornar

        Returns:
            Lista de (línea_a_desconectar, impacto_estimado)
        """
        if not self._initialized:
            self.initialize(obs)

        impacts = []

        for k in range(self.n_line):
            if k == overloaded_line or not obs.line_status[k]:
                continue

            # LODF negativo significa que desconectar k reduce el flujo
            # en la línea sobrecargada
            lodf = self.lodf[overloaded_line, k]
            flow_k = obs.p_or[k]

            # Impacto = LODF * flujo_de_k
            # Queremos valores negativos (reducción de flujo)
            impact = lodf * flow_k
            impacts.append((k, impact))

        # Ordenar por impacto (más negativo = mejor)
        impacts.sort(key=lambda x: x[1])

        return impacts[:top_k]


class SensitivityAgent(BaseAgent):
    """
    Agente basado en análisis de sensibilidades (PTDF/LODF).

    A diferencia del agente Greedy que prueba todas las acciones,
    este agente usa análisis matemático para pre-filtrar las
    acciones más prometedoras.

    Proceso:
    1. Detectar líneas sobrecargadas
    2. Usar LODF para identificar desconexiones que alivian la sobrecarga
    3. Filtrar top K candidatas
    4. Simular solo esas K acciones en Grid2Op
    5. Seleccionar la mejor basada en resultados reales

    Ventajas:
    - Mucho más rápido que fuerza bruta
    - Decisiones explicables (basadas en física)
    - Escala mejor a redes grandes
    """

    def __init__(
        self,
        action_space,
        top_k_candidates: int = 5,
        rho_threshold: float = 0.9,
        stress_factors: List[float] = None
    ):
        """
        Inicializa el agente de sensibilidades.

        Args:
            action_space: Espacio de acciones
            top_k_candidates: Candidatas a simular después del filtrado
            rho_threshold: Umbral de sobrecarga
            stress_factors: Factores de estrés para validación
        """
        super().__init__(action_space)
        self.top_k_candidates = top_k_candidates
        self.rho_threshold = rho_threshold
        self.stress_factors = stress_factors or [1.0, 1.02, 1.05]

        self.sensitivity_analyzer = None
        self._env = None

    def initialize_analyzer(self, env: Environment, obs: BaseObservation):
        """Inicializa el analizador de sensibilidades."""
        self._env = env
        self.sensitivity_analyzer = NetworkSensitivityAnalyzer(env)
        self.sensitivity_analyzer.initialize(obs)

    def _get_candidate_actions(
        self,
        obs: BaseObservation
    ) -> List[Tuple[BaseAction, str, float]]:
        """
        Obtiene acciones candidatas usando análisis de sensibilidad.

        Args:
            obs: Observación actual

        Returns:
            Lista de (acción, descripción, score_teórico)
        """
        if self.sensitivity_analyzer is None:
            return []

        candidates = []

        # Identificar líneas sobrecargadas
        overloaded_lines = np.where(obs.rho > self.rho_threshold)[0]

        if len(overloaded_lines) == 0:
            return [(self.action_space(), "Do nothing", 0.0)]

        # Para cada línea sobrecargada, encontrar las mejores desconexiones
        all_candidates = {}

        for ol in overloaded_lines:
            ranked = self.sensitivity_analyzer.rank_disconnection_actions(
                ol, obs, self.top_k_candidates
            )

            for line_id, impact in ranked:
                if line_id not in all_candidates:
                    all_candidates[line_id] = impact
                else:
                    # Sumar impactos si ayuda a múltiples líneas
                    all_candidates[line_id] += impact

        # Ordenar por impacto total y crear acciones
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1])

        for line_id, impact in sorted_candidates[:self.top_k_candidates]:
            try:
                action = self.action_space({"set_line_status": [(line_id, -1)]})
                candidates.append((
                    action,
                    f"Disconnect line {line_id} (LODF score: {impact:.3f})",
                    impact
                ))
            except Exception:
                pass

        # Añadir también acciones de cambio de bus
        candidates.extend(self._get_bus_change_candidates(obs, overloaded_lines))

        return candidates

    def _get_bus_change_candidates(
        self,
        obs: BaseObservation,
        overloaded_lines: np.ndarray
    ) -> List[Tuple[BaseAction, str, float]]:
        """
        Genera candidatas de cambio de bus para líneas sobrecargadas.

        El cambio de bus puede redistribuir flujos de forma efectiva
        sin desconectar elementos.

        Args:
            obs: Observación actual
            overloaded_lines: Índices de líneas sobrecargadas

        Returns:
            Lista de candidatas (acción, descripción, score)
        """
        candidates = []

        for line_id in overloaded_lines:
            # Cambio de bus en origen
            try:
                action = self.action_space()
                action.line_or_change_bus = [int(line_id)]
                score = -obs.rho[line_id]  # Score negativo = prioridad
                candidates.append((
                    action,
                    f"Change origin bus line {line_id}",
                    score
                ))
            except Exception:
                pass

            # Cambio de bus en extremo
            try:
                action = self.action_space()
                action.line_ex_change_bus = [int(line_id)]
                score = -obs.rho[line_id]
                candidates.append((
                    action,
                    f"Change extremity bus line {line_id}",
                    score
                ))
            except Exception:
                pass

        return candidates

    def _validate_action(
        self,
        obs: BaseObservation,
        action: BaseAction
    ) -> Tuple[bool, float, float]:
        """
        Valida una acción mediante simulación.

        Args:
            obs: Observación actual
            action: Acción a validar

        Returns:
            (válida, rho_max, lambda_margin)
        """
        try:
            sim_obs, _, sim_done, _ = obs.simulate(action)

            if sim_done:
                return False, float('inf'), 0.0

            rho_max = np.max(sim_obs.rho)

            # Estimar margen de carga con stress testing
            lambda_margin = 1.0
            simulator = obs.get_simulator()

            init_load_p = obs.load_p.copy()
            init_load_q = obs.load_q.copy()
            init_gen_p = obs.gen_p.copy()

            for stress in self.stress_factors:
                try:
                    result = simulator.predict(
                        action,
                        new_gen_p=init_gen_p * stress,
                        new_load_p=init_load_p * stress,
                        new_load_q=init_load_q * stress
                    )

                    if not result.converged:
                        break
                    if np.any(result.current_obs.rho > 1.0):
                        break
                    lambda_margin = stress

                except Exception:
                    break

            return True, rho_max, lambda_margin

        except Exception:
            return False, float('inf'), 0.0

    def act(
        self,
        obs: BaseObservation,
        reward: float = 0,
        done: bool = False
    ) -> BaseAction:
        """
        Decide la mejor acción usando análisis de sensibilidades.

        Args:
            obs: Observación actual
            reward: Reward anterior
            done: Si terminó el episodio

        Returns:
            Mejor acción encontrada
        """
        if done:
            return self.action_space()

        # Verificar si hay sobrecarga
        rho_max = np.max(obs.rho)
        if rho_max < self.rho_threshold:
            return self.action_space()

        # Inicializar analizador si es necesario
        if self.sensitivity_analyzer is None:
            self.sensitivity_analyzer = NetworkSensitivityAnalyzer(self._env)
            self.sensitivity_analyzer.initialize(obs)

        # Obtener candidatas filtradas por sensibilidad
        candidates = self._get_candidate_actions(obs)

        if not candidates:
            return self.action_space()

        # Validar candidatas mediante simulación
        best_action = self.action_space()
        best_lambda = 0.0
        best_rho = float('inf')

        for action, description, theoretical_score in candidates:
            valid, rho, lambda_margin = self._validate_action(obs, action)

            if valid and lambda_margin > best_lambda:
                best_action = action
                best_lambda = lambda_margin
                best_rho = rho

            elif valid and lambda_margin == best_lambda and rho < best_rho:
                best_action = action
                best_rho = rho

        return best_action

    def get_analysis_report(
        self,
        obs: BaseObservation
    ) -> Dict:
        """
        Genera un informe detallado del análisis de sensibilidades.

        Útil para explicar las decisiones del agente.

        Args:
            obs: Observación actual

        Returns:
            Diccionario con análisis detallado
        """
        if self.sensitivity_analyzer is None:
            return {"error": "Analyzer not initialized"}

        overloaded = np.where(obs.rho > self.rho_threshold)[0]

        report = {
            "overloaded_lines": overloaded.tolist(),
            "rho_values": {int(l): float(obs.rho[l]) for l in overloaded},
            "recommended_actions": [],
            "lodf_analysis": {}
        }

        for ol in overloaded:
            ranked = self.sensitivity_analyzer.rank_disconnection_actions(
                ol, obs, self.top_k_candidates
            )
            report["lodf_analysis"][int(ol)] = [
                {"line": int(l), "impact": float(i)} for l, i in ranked
            ]

        candidates = self._get_candidate_actions(obs)
        for action, desc, score in candidates[:5]:
            valid, rho, lam = self._validate_action(obs, action)
            report["recommended_actions"].append({
                "description": desc,
                "theoretical_score": float(score),
                "valid": valid,
                "rho_after": float(rho) if valid else None,
                "lambda_margin": float(lam) if valid else None
            })

        return report


class SensitivityLoadMarginOptimizer:
    """
    Optimizador de margen de carga usando análisis de sensibilidades.

    Combina el análisis PTDF/LODF con la búsqueda de margen de carga
    para maximizar λ* de forma eficiente.
    """

    def __init__(
        self,
        env: Environment,
        top_k_candidates: int = 5,
        rho_threshold: float = 0.9
    ):
        """
        Inicializa el optimizador.

        Args:
            env: Entorno Grid2Op
            top_k_candidates: Candidatas a evaluar
            rho_threshold: Umbral de sobrecarga
        """
        self.env = env
        self.agent = SensitivityAgent(
            env.action_space,
            top_k_candidates=top_k_candidates,
            rho_threshold=rho_threshold
        )
        self.agent._env = env

    def optimize_load_margin(
        self,
        lambda_start: float = 1.0,
        lambda_end: float = 2.0,
        lambda_step: float = 0.01,
        max_actions: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Optimiza el margen de carga usando acciones basadas en sensibilidades.

        Args:
            lambda_start: Factor inicial
            lambda_end: Factor máximo
            lambda_step: Incremento
            max_actions: Máximo de acciones
            verbose: Imprimir progreso

        Returns:
            Resultados del análisis
        """
        obs = self.env.reset()

        # Inicializar analizador
        self.agent.initialize_analyzer(self.env, obs)

        simulator = obs.get_simulator()

        init_load_p = obs.load_p.copy()
        init_load_q = obs.load_q.copy()
        init_gen_p = obs.gen_p.copy()

        lambda_values = np.arange(lambda_start, lambda_end + lambda_step, lambda_step)

        results = {
            "lambda_values": [],
            "rho_max_values": [],
            "actions_taken": [],
            "lambda_max": lambda_start,
            "analysis_reports": []
        }

        actions_count = 0
        current_action = self.env.action_space()

        for lam in lambda_values:
            new_load_p = init_load_p * lam
            new_load_q = init_load_q * lam
            new_gen_p = init_gen_p * lam

            try:
                result = simulator.predict(
                    current_action,
                    new_gen_p=new_gen_p,
                    new_load_p=new_load_p,
                    new_load_q=new_load_q
                )

                if not result.converged:
                    if verbose:
                        print(f"λ={lam:.3f}: No convergencia")
                    break

                sim_obs = result.current_obs
                rho_max = np.max(sim_obs.rho)

                results["lambda_values"].append(lam)
                results["rho_max_values"].append(rho_max)

                if rho_max > 1.0 and actions_count < max_actions:
                    # Usar análisis de sensibilidad
                    self.agent.sensitivity_analyzer.initialize(sim_obs)
                    report = self.agent.get_analysis_report(sim_obs)

                    correction = self.agent.act(sim_obs)

                    if correction != self.env.action_space():
                        actions_count += 1
                        results["actions_taken"].append({
                            "lambda": lam,
                            "rho_before": rho_max,
                            "analysis": report
                        })

                        if verbose:
                            print(f"λ={lam:.3f}: Acción correctiva basada en LODF")

                elif rho_max > 1.0:
                    if verbose:
                        print(f"λ={lam:.3f}: Sobrecarga, sin acciones disponibles")
                    break

                results["lambda_max"] = lam

                if verbose and int(lam * 100) % 10 == 0:
                    print(f"λ={lam:.3f}: ρ_max={rho_max:.1%}")

            except Exception as e:
                if verbose:
                    print(f"λ={lam:.3f}: Error - {str(e)}")
                break

        return results
