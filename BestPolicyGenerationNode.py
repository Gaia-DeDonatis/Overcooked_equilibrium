"""
Best Policy Generation Node for Ax optimization in t-SNE embedding space.

This node always samples the best policy found so far (greedy exploitation).
When mapped through TSNEBayesOptimizer, this will select the nearest unevaluated
policy to the current best.
"""

from typing import Any
from collections.abc import Sequence
import numpy as np
from scipy.spatial.distance import cdist

from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import TransitionCriterion
from ax.core.experiment import Experiment
from ax.core.data import Data
from ax.core.types import TParameterization
from ax.core.trial_status import TrialStatus
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.decoder import object_from_json
from ax.storage.json_store.registry import (
    CORE_ENCODER_REGISTRY,
    CORE_CLASS_ENCODER_REGISTRY,
    CORE_DECODER_REGISTRY,
    CORE_CLASS_DECODER_REGISTRY,
)

from logging_utils import opt_logger as logger


def best_policy_generation_node_to_dict(node: "BestPolicyGenerationNode") -> dict[str, Any]:
    """
    Encoder function for BestPolicyGenerationNode serialization.
    """
    visualization_spec_json = None
    if node._visualization_generator_spec is not None:
        visualization_spec_json = object_to_json(
            node._visualization_generator_spec,
            encoder_registry=CORE_ENCODER_REGISTRY,
            class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
        )

    return {
        "__type": "BestPolicyGenerationNode",
        "name": node.name,
        "coords": node.coords.tolist(),
        "policy_names": node.policy_names.tolist(),
        "evaluated_policies": list(node.evaluated_policies),
        "best_coords": node.best_coords.tolist() if node.best_coords is not None else None,
        "best_policy": node.best_policy,
        "best_value": node.best_value,
        "minimize": node.minimize,
        "max_trials_to_consider": node.max_trials_to_consider,
        "visualization_generator_spec": visualization_spec_json,
    }


def best_policy_generation_node_from_dict(**kwargs) -> "BestPolicyGenerationNode":
    """
    Decoder function for BestPolicyGenerationNode deserialization.
    """
    visualization_spec = None
    viz_spec_data = kwargs.get("visualization_generator_spec")
    if viz_spec_data is not None:
        if isinstance(viz_spec_data, GeneratorSpec):
            visualization_spec = viz_spec_data
        elif isinstance(viz_spec_data, dict):
            visualization_spec = object_from_json(
                viz_spec_data,
                decoder_registry=CORE_DECODER_REGISTRY,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )
        else:
            logger.warning(f"[OPT - BEST] Unexpected visualization_generator_spec type: {type(viz_spec_data)}")

    generator_specs = [visualization_spec] if visualization_spec else None

    node = BestPolicyGenerationNode(
        coords=np.array(kwargs["coords"]),
        policy_names=np.array(kwargs["policy_names"]),
        evaluated_policies=set(kwargs.get("evaluated_policies", [])),
        generator_specs=generator_specs,
        max_trials_to_consider=kwargs.get("max_trials_to_consider"),
        name=kwargs.get("name", "BestPolicy"),
    )

    # Restore runtime state
    if kwargs.get("best_coords") is not None:
        node.best_coords = np.array(kwargs["best_coords"])
    node.best_policy = kwargs.get("best_policy")
    node.best_value = kwargs.get("best_value")
    node.minimize = kwargs.get("minimize", False)

    return node


class BestPolicyGenerationNode(ExternalGenerationNode):
    """
    A generation node that always samples the best policy found so far.

    This is a greedy exploitation node - it returns the coordinates of the
    current best policy. When used with TSNEBayesOptimizer, this will be
    mapped to the nearest unevaluated policy, effectively exploring the
    neighborhood of the best point.

    Maintains a surrogate model (via generator_specs) for visualization purposes.
    """

    def __init__(
        self,
        coords: np.ndarray,
        policy_names: np.ndarray,
        evaluated_policies: set = None,
        generator_specs: list[GeneratorSpec] | None = None,
        transition_criteria: Sequence[TransitionCriterion] | None = None,
        max_trials_to_consider: int | None = None,
        name: str = "BestPolicy",
    ) -> None:
        """
        Initialize the best policy generation node.

        Args:
            coords: Array of shape (n_policies, 2) with normalized embedding coordinates.
            policy_names: Array of policy names corresponding to each coordinate.
            evaluated_policies: Set of already evaluated policy names (shared reference).
            generator_specs: Optional list of GeneratorSpecs. The first spec will be
                fitted as a surrogate model for visualization purposes.
            transition_criteria: Optional transition criteria for the generation strategy.
            max_trials_to_consider: If set, only consider trials with index < this value
                when finding the best. Useful to only consider Sobol+BO trials.
            name: Name of this node (default "BestPolicy").
        """
        self._visualization_generator_spec: GeneratorSpec | None = (
            generator_specs[0] if generator_specs and len(generator_specs) > 0 else None
        )

        super().__init__(name=name, transition_criteria=transition_criteria)

        self.coords = coords
        self.policy_names = policy_names
        self.evaluated_policies = evaluated_policies if evaluated_policies is not None else set()
        self.max_trials_to_consider = max_trials_to_consider

        # State updated during generation
        self.best_coords: np.ndarray | None = None
        self.best_policy: str | None = None
        self.best_value: float | None = None
        self.minimize: bool = False

    @property
    def _fitted_adapter(self):
        """
        Return the fitted adapter from the visualization generator spec.

        This enables Ax visualizations (SlicePlot, ContourPlot) to work.
        """
        if self._visualization_generator_spec is not None:
            return self._visualization_generator_spec._fitted_adapter
        return None

    def _fit(
        self,
        experiment: Experiment,
        data: Data | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Fit the node state and optionally the visualization surrogate model.
        """
        super()._fit(experiment=experiment, data=data)

        # Fit the visualization surrogate model for plotting
        if self._visualization_generator_spec is not None:
            try:
                actual_data = data if data is not None else experiment.lookup_data()
                self._visualization_generator_spec.fit(
                    experiment=experiment,
                    data=actual_data,
                )
                logger.info("[OPT - BEST] Fitted visualization surrogate model for plots")
            except Exception as e:
                logger.warning(f"[OPT - BEST] Failed to fit visualization surrogate: {e}")

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """
        Update the generator state with the latest experiment data.

        Finds the best trial so far.
        """
        metric_names = list(experiment.optimization_config.metrics.keys())
        if len(metric_names) != 1:
            raise NotImplementedError("BestPolicy node only supports single-objective optimization.")

        metric_name = metric_names[0]
        self.minimize = experiment.optimization_config.objective.minimize

        # Find the best completed trial
        best_value = None
        best_coords = None
        best_trial_idx = None

        # Get completed trials sorted by index (preserves order)
        completed_trials = sorted(
            [(idx, t) for idx, t in experiment.trials.items() if t.status == TrialStatus.COMPLETED],
            key=lambda x: x[0]
        )

        # If max_trials_to_consider is set, only look at first N completed trials
        if self.max_trials_to_consider is not None:
            completed_trials = completed_trials[:self.max_trials_to_consider]

        for trial_idx, trial in completed_trials:
            trial_df = data.df[data.df["trial_index"] == trial_idx]
            metric_df = trial_df[trial_df["metric_name"] == metric_name]

            if metric_df.empty:
                continue

            trial_value = metric_df["mean"].item()
            trial_params = trial.arm.parameters

            is_better = (
                best_value is None or
                (self.minimize and trial_value < best_value) or
                (not self.minimize and trial_value > best_value)
            )

            if is_better:
                best_value = trial_value
                best_coords = np.array([trial_params["emb_x"], trial_params["emb_y"]])
                best_trial_idx = trial_idx

        if best_coords is None or best_value is None:
            raise ValueError("No completed trials found to determine best point.")

        self.best_coords = best_coords
        self.best_value = float(best_value)

        # Find the best policy name by mapping best_coords back
        best_distances = cdist(best_coords.reshape(1, -1), self.coords, metric='euclidean')[0]
        best_idx = np.argmin(best_distances)
        self.best_policy = str(self.policy_names[best_idx])

        logger.info(f"[OPT - BEST] SELECTED trial {best_trial_idx}: best_policy='{self.best_policy}', coords=({self.best_coords[0]:.3f}, {self.best_coords[1]:.3f}), value={self.best_value:.4f}")

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        """
        Get the next candidate - always returns the best policy's coordinates.

        Adds tiny noise to avoid Ax's MAX_GEN_ATTEMPTS fallback (which triggers
        after 5 identical parameterizations). The noise is small enough that
        nearest-neighbor mapping still finds the correct policy.
        """
        if self.best_coords is None:
            raise RuntimeError("No best coordinates available. Call update_generator_state first.")

        # Add tiny noise to avoid Ax thinking we're stuck (noise << policy spacing)
        noise = np.random.uniform(-1e-6, 1e-6, size=2)
        coords = self.best_coords + noise

        logger.info(f"[OPT - BEST] sampling best_policy='{self.best_policy}', coords=({self.best_coords[0]:.3f}, {self.best_coords[1]:.3f}), value={self.best_value:.4f}")

        return {
            "emb_x": float(coords[0]),
            "emb_y": float(coords[1]),
        }
