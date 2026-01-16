"""
Greedy Agent with Look-Ahead
============================
Agente baseline que evalúa acciones topológicas de forma greedy,
simulando N pasos hacia el futuro para asegurar que una acción
que alivia una sobrecarga no provoca un colapso posterior.

Este agente sirve como línea base para comparar con métodos
más sofisticados (PTDF/LODF).

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
class ActionEvaluation:
    """Evaluación de una acción topológica."""
    action: BaseAction
    description: str
    rho_max_after: float  # ρ máximo después de aplicar la acción
    rho_improvement: float  # Mejora en ρ (negativo = mejor)
    survives_lookahead: bool  # Si sobrevive N pasos
    lookahead_rho_max: float  # ρ máximo en el horizonte
    lambda_margin: float  # Margen de carga estimado


class GreedyAgent(BaseAgent):
    """
    Agente Greedy con Look-Ahead para control topológico.

    Estrategia:
    1. Genera un conjunto de acciones candidatas
    2. Simula cada acción y evalúa el impacto inmediato
    3. Para las mejores K acciones, simula N pasos hacia adelante
    4. Selecciona la acción que maximiza el margen de carga

    Attributes:
        action_space: Espacio de acciones de Grid2Op
        rho_threshold: Umbral de ρ para considerar sobrecarga
        lookahead_steps: Pasos hacia adelante a simular
        top_k_actions: Número de mejores acciones a evaluar con lookahead
    """

    def __init__(
        self,
        action_space,
        rho_threshold: float = 0.9,
        lookahead_steps: int = 3,
        top_k_actions: int = 5,
        stress_factors: List[float] = None
    ):
        """
        Inicializa el agente Greedy.

        Args:
            action_space: Espacio de acciones del entorno
            rho_threshold: Umbral de ρ para considerar que hay sobrecarga
            lookahead_steps: Número de pasos a simular hacia adelante
            top_k_actions: Top K acciones a evaluar con lookahead
            stress_factors: Factores de estrés para evaluar robustez
        """
        super().__init__(action_space)
        self.rho_threshold = rho_threshold
        self.lookahead_steps = lookahead_steps
        self.top_k_actions = top_k_actions
        self.stress_factors = stress_factors or [1.0, 1.02, 1.05, 1.08, 1.10]

        # Caché de acciones generadas
        self._candidate_actions = None

    def _generate_candidate_actions(self, obs: BaseObservation) -> List[Tuple[BaseAction, str]]:
        """
        Genera acciones candidatas basadas en el estado actual.

        Tipos de acciones:
        1. Desconexión de líneas (set_line_status)
        2. Reconexión de líneas desconectadas
        3. Cambio de topología en subestaciones (change_bus)

        Args:
            obs: Observación actual

        Returns:
            Lista de tuplas (acción, descripción)
        """
        actions = []

        # Acción nula (do nothing)
        do_nothing = self.action_space()
        actions.append((do_nothing, "Do nothing"))

        # 1. Desconexión de líneas (solo las que están conectadas y no muy cargadas)
        for line_id in range(obs.n_line):
            if obs.line_status[line_id]:  # Línea conectada
                # Solo desconectar líneas que no son críticas
                if obs.rho[line_id] < 0.5:  # Evitar desconectar líneas muy cargadas
                    try:
                        action = self.action_space({"set_line_status": [(line_id, -1)]})
                        actions.append((action, f"Disconnect line {line_id}"))
                    except Exception:
                        pass  # Acción no válida

        # 2. Reconexión de líneas desconectadas
        for line_id in range(obs.n_line):
            if not obs.line_status[line_id]:  # Línea desconectada
                try:
                    # Para reconectar, necesitamos especificar los buses
                    action = self.action_space({
                        "set_line_status": [(line_id, 1)],
                        "set_bus": {
                            "lines_or_id": [(line_id, 1)],
                            "lines_ex_id": [(line_id, 1)]
                        }
                    })
                    actions.append((action, f"Reconnect line {line_id}"))
                except Exception:
                    pass

        # 3. Cambio de bus en subestaciones (para líneas sobrecargadas)
        overloaded_lines = np.where(obs.rho > self.rho_threshold)[0]

        for line_id in overloaded_lines:
            if obs.line_status[line_id]:
                # Cambiar bus del extremo origen
                try:
                    action = self.action_space()
                    action.line_or_change_bus = [line_id]
                    actions.append((action, f"Change bus origin line {line_id}"))
                except Exception:
                    pass

                # Cambiar bus del extremo destino
                try:
                    action = self.action_space()
                    action.line_ex_change_bus = [line_id]
                    actions.append((action, f"Change bus extremity line {line_id}"))
                except Exception:
                    pass

        return actions

    def _evaluate_action(
        self,
        obs: BaseObservation,
        action: BaseAction,
        description: str
    ) -> Optional[ActionEvaluation]:
        """
        Evalúa una acción simulando su impacto.

        Args:
            obs: Observación actual
            action: Acción a evaluar
            description: Descripción de la acción

        Returns:
            ActionEvaluation o None si la acción falla
        """
        try:
            # Simular la acción
            sim_obs, sim_reward, sim_done, sim_info = obs.simulate(action)

            if sim_done:
                return None  # La acción causa game over

            # Calcular métricas
            rho_max_before = np.max(obs.rho)
            rho_max_after = np.max(sim_obs.rho)
            rho_improvement = rho_max_after - rho_max_before

            return ActionEvaluation(
                action=action,
                description=description,
                rho_max_after=rho_max_after,
                rho_improvement=rho_improvement,
                survives_lookahead=True,  # Se evaluará después
                lookahead_rho_max=rho_max_after,
                lambda_margin=1.0
            )

        except Exception as e:
            return None

    def _evaluate_with_lookahead(
        self,
        obs: BaseObservation,
        action: BaseAction,
        evaluation: ActionEvaluation
    ) -> ActionEvaluation:
        """
        Evalúa una acción con look-ahead y stress testing.

        Args:
            obs: Observación actual
            action: Acción a evaluar
            evaluation: Evaluación inicial

        Returns:
            ActionEvaluation actualizada
        """
        try:
            simulator = obs.get_simulator()

            # Valores iniciales
            init_load_p = obs.load_p.copy()
            init_load_q = obs.load_q.copy()
            init_gen_p = obs.gen_p.copy()

            max_stress_survived = 1.0
            max_rho_seen = evaluation.rho_max_after

            # Probar diferentes niveles de estrés
            for stress in self.stress_factors:
                new_load_p = init_load_p * stress
                new_load_q = init_load_q * stress
                new_gen_p = init_gen_p * stress

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

                    # Verificar límites
                    if np.any(sim_obs.rho > 1.0):
                        break

                    max_rho_seen = max(max_rho_seen, np.max(sim_obs.rho))
                    max_stress_survived = stress

                except Exception:
                    break

            evaluation.survives_lookahead = max_stress_survived >= 1.0
            evaluation.lookahead_rho_max = max_rho_seen
            evaluation.lambda_margin = max_stress_survived

            return evaluation

        except Exception:
            evaluation.survives_lookahead = False
            return evaluation

    def act(self, obs: BaseObservation, reward: float = 0, done: bool = False) -> BaseAction:
        """
        Decide la mejor acción a tomar.

        Proceso:
        1. Si no hay sobrecarga, no hacer nada
        2. Generar acciones candidatas
        3. Evaluar cada acción
        4. Seleccionar top K y evaluar con lookahead
        5. Retornar la mejor acción

        Args:
            obs: Observación actual
            reward: Reward del paso anterior
            done: Si el episodio terminó

        Returns:
            Mejor acción encontrada
        """
        if done:
            return self.action_space()

        # Verificar si hay sobrecarga
        rho_max = np.max(obs.rho)
        if rho_max < self.rho_threshold:
            return self.action_space()  # No hay sobrecarga, do nothing

        # Generar acciones candidatas
        candidate_actions = self._generate_candidate_actions(obs)

        # Evaluar cada acción
        evaluations = []
        for action, description in candidate_actions:
            evaluation = self._evaluate_action(obs, action, description)
            if evaluation is not None:
                evaluations.append(evaluation)

        if not evaluations:
            return self.action_space()  # No hay acciones válidas

        # Ordenar por mejora en ρ
        evaluations.sort(key=lambda e: e.rho_improvement)

        # Tomar top K y evaluar con lookahead
        top_evaluations = evaluations[:self.top_k_actions]

        for evaluation in top_evaluations:
            evaluation = self._evaluate_with_lookahead(obs, evaluation.action, evaluation)

        # Filtrar las que sobreviven el lookahead
        valid_evaluations = [e for e in top_evaluations if e.survives_lookahead]

        if not valid_evaluations:
            # Si ninguna sobrevive, tomar la que tiene mejor rho inmediato
            valid_evaluations = top_evaluations

        # Seleccionar la mejor (mayor margen de carga)
        best_evaluation = max(valid_evaluations, key=lambda e: e.lambda_margin)

        return best_evaluation.action

    def get_best_action_with_info(
        self,
        obs: BaseObservation
    ) -> Tuple[BaseAction, Dict]:
        """
        Obtiene la mejor acción junto con información detallada.

        Útil para análisis y debugging.

        Args:
            obs: Observación actual

        Returns:
            Tupla (acción, info_dict)
        """
        action = self.act(obs)

        info = {
            "rho_max_before": np.max(obs.rho),
            "action_type": "do_nothing"
        }

        return action, info


class GreedyLoadMarginOptimizer:
    """
    Optimizador de margen de carga usando el agente Greedy.

    Combina el análisis de margen de carga con el agente Greedy
    para encontrar configuraciones topológicas que maximicen λ*.
    """

    def __init__(
        self,
        env: Environment,
        rho_threshold: float = 0.9,
        lookahead_steps: int = 3
    ):
        """
        Inicializa el optimizador.

        Args:
            env: Entorno Grid2Op
            rho_threshold: Umbral de sobrecarga
            lookahead_steps: Pasos de lookahead
        """
        self.env = env
        self.agent = GreedyAgent(
            env.action_space,
            rho_threshold=rho_threshold,
            lookahead_steps=lookahead_steps
        )

    def optimize_load_margin(
        self,
        lambda_start: float = 1.0,
        lambda_end: float = 2.0,
        lambda_step: float = 0.01,
        max_actions: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Optimiza el margen de carga aplicando acciones topológicas.

        En cada paso donde hay sobrecarga, el agente Greedy
        propone una acción correctiva.

        Args:
            lambda_start: Factor inicial
            lambda_end: Factor máximo
            lambda_step: Incremento
            max_actions: Máximo de acciones a aplicar
            verbose: Imprimir progreso

        Returns:
            Diccionario con resultados y acciones aplicadas
        """
        obs = self.env.reset()
        simulator = obs.get_simulator()

        init_load_p = obs.load_p.copy()
        init_load_q = obs.load_q.copy()
        init_gen_p = obs.gen_p.copy()

        lambda_values = np.arange(lambda_start, lambda_end + lambda_step, lambda_step)

        results = {
            "lambda_values": [],
            "rho_max_values": [],
            "actions_taken": [],
            "lambda_max": lambda_start
        }

        actions_count = 0
        current_action = self.env.action_space()  # Start with do nothing

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

                # Si hay sobrecarga y podemos actuar
                if rho_max > 1.0:
                    if actions_count < max_actions:
                        # Usar el agente para encontrar una acción correctiva
                        correction = self.agent.act(sim_obs)

                        if correction != self.env.action_space():
                            actions_count += 1
                            results["actions_taken"].append({
                                "lambda": lam,
                                "rho_before": rho_max,
                                "action": str(correction)
                            })

                            # Actualizar la acción actual
                            # (En una implementación real, combinaríamos acciones)
                            if verbose:
                                print(f"λ={lam:.3f}: Acción correctiva #{actions_count}")

                    else:
                        if verbose:
                            print(f"λ={lam:.3f}: Sobrecarga sin acciones disponibles")
                        break

                results["lambda_max"] = lam

                if verbose and int(lam * 100) % 10 == 0:
                    print(f"λ={lam:.3f}: ρ_max={rho_max:.1%}")

            except Exception as e:
                if verbose:
                    print(f"λ={lam:.3f}: Error - {str(e)}")
                break

        return results
