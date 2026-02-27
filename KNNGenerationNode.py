"""
KNN Generation Node for Ax optimization in t-SNE embedding space.

This node samples the N nearest neighbors from the best point found so far.
Designed to be used after Bayesian Optimization for local exploitation.
"""

from typing import Any
import numpy as np
from scipy.spatial.distance import cdist

from ax.generation_strategy.external_generation_node import ExternalGenerationNode
from ax.core.experiment import Experiment
from ax.core.data import Data
from ax.core.types import TParameterization
from ax.core.trial_status import TrialStatus

from logging_utils import opt_logger as logger


def knn_generation_node_to_dict(node: "KNNGenerationNode") -> dict[str, Any]:
    """
    Encoder function for KNNGenerationNode serialization.

    Converts numpy arrays to lists and sets to lists for JSON compatibility.
    """
    return {
        "__type": "KNNGenerationNode",
        "name": node.name,
        "coords": node.coords.tolist(),
        "policy_names": node.policy_names.tolist(),
        "n_neighbors": node.n_neighbors,
        "evaluated_policies": list(node.evaluated_policies),
        # Also save runtime state for proper restoration
        "best_coords": node.best_coords.tolist() if node.best_coords is not None else None,
        "best_policy": node.best_policy,
        "best_value": node.best_value,
        "neighbor_queue": [
            (name, coords.tolist(), dist)
            for name, coords, dist in node.neighbor_queue
        ],
        "minimize": node.minimize,
        # ExternalGenerationNode fields
        "generator_specs": node.generator_specs,
        "transition_criteria": node.transition_criteria,
    }


def knn_generation_node_from_dict(**kwargs) -> "KNNGenerationNode":
    """
    Decoder function for KNNGenerationNode deserialization.

    Reconstructs numpy arrays from lists and sets from lists.
    Ax calls this with keyword arguments extracted from the JSON dict.
    """
    node = KNNGenerationNode(
        coords=np.array(kwargs["coords"]),
        policy_names=np.array(kwargs["policy_names"]),
        n_neighbors=kwargs["n_neighbors"],
        evaluated_policies=set(kwargs.get("evaluated_policies", [])),
    )

    # Restore runtime state
    if kwargs.get("best_coords") is not None:
        node.best_coords = np.array(kwargs["best_coords"])
    node.best_policy = kwargs.get("best_policy")
    node.best_value = kwargs.get("best_value")
    node.neighbor_queue = [
        (name, np.array(coords), dist)
        for name, coords, dist in kwargs.get("neighbor_queue", [])
    ]
    node.minimize = kwargs.get("minimize", False)

    return node


class KNNGenerationNode(ExternalGenerationNode):
    """
    A generation node that samples the N nearest neighbors in embedding space
    from the best sample found so far.

    This node is designed to be used after Bayesian Optimization to exploit
    the region around the best point by systematically sampling nearby points.
    """

    def __init__(
        self,
        coords: np.ndarray,
        policy_names: np.ndarray,
        n_neighbors: int = 10,
        evaluated_policies: set = None,
    ) -> None:
        """
        Initialize the kNN generation node.

        Args:
            coords: Array of shape (n_policies, 2) with normalized embedding coordinates.
            policy_names: Array of policy names corresponding to each coordinate.
            n_neighbors: Number of nearest neighbors to sample from.
            evaluated_policies: Set of already evaluated policy names (shared reference).
        """
        super().__init__(name="kNN")
        self.coords = coords
        self.policy_names = policy_names
        self.n_neighbors = n_neighbors
        self.evaluated_policies = evaluated_policies if evaluated_policies is not None else set()

        # State updated during generation
        self.best_coords: np.ndarray | None = None
        self.best_policy: str | None = None
        self.best_value: float | None = None
        self.neighbor_queue: list[tuple[str, np.ndarray, float]] = []  # (policy_name, coords, distance)
        self.minimize: bool = False

    def update_generator_state(self, experiment: Experiment, data: Data) -> None:
        """
        Update the generator state with the latest experiment data.

        Finds the best trial so far and computes the N nearest neighbors
        that haven't been evaluated yet.
        """
        # Get metric name and optimization direction
        metric_names = list(experiment.optimization_config.metrics.keys())
        if len(metric_names) != 1:
            raise NotImplementedError("kNN node only supports single-objective optimization.")

        metric_name = metric_names[0]
        self.minimize = experiment.optimization_config.objective.minimize

        # Find the best completed trial
        best_value = None
        best_coords = None

        for trial_idx, trial in experiment.trials.items():
            if trial.status != TrialStatus.COMPLETED:
                continue

            trial_df = data.df[data.df["trial_index"] == trial_idx]
            metric_df = trial_df[trial_df["metric_name"] == metric_name]

            if metric_df.empty:
                continue

            trial_value = metric_df["mean"].item()
            trial_params = trial.arm.parameters

            # Check if this is the best so far
            is_better = (
                best_value is None or
                (self.minimize and trial_value < best_value) or
                (not self.minimize and trial_value > best_value)
            )

            if is_better:
                best_value = trial_value
                best_coords = np.array([trial_params["emb_x"], trial_params["emb_y"]])

        if best_coords is None or best_value is None:
            raise ValueError("No completed trials found to determine best point.")

        self.best_coords = best_coords
        self.best_value = float(best_value)

        # Compute distances from best point to all policies
        distances = cdist(best_coords.reshape(1, -1), self.coords, metric='euclidean')[0]

        # Sort by distance and build the queue of unevaluated neighbors
        sorted_indices = np.argsort(distances)

        # Find the best policy name by mapping best_coords back
        best_distances = cdist(best_coords.reshape(1, -1), self.coords, metric='euclidean')[0]
        best_idx = np.argmin(best_distances)
        self.best_policy = str(self.policy_names[best_idx])

        self.neighbor_queue = []
        for idx in sorted_indices:
            policy_name = str(self.policy_names[idx])
            if policy_name not in self.evaluated_policies:
                dist = float(distances[idx])
                self.neighbor_queue.append((policy_name, self.coords[idx], dist))
                if len(self.neighbor_queue) >= self.n_neighbors:
                    break

        logger.info(f"[OPT - KNN] best_policy='{self.best_policy}', coords=({self.best_coords[0]:.3f}, {self.best_coords[1]:.3f}), value={self.best_value:.4f}")
        logger.info(f"[OPT - KNN] found {len(self.neighbor_queue)} unevaluated neighbors within queue")

    def get_next_candidate(
        self, pending_parameters: list[TParameterization]
    ) -> TParameterization:
        """
        Get the next candidate from the nearest neighbor queue.

        Returns the closest unevaluated neighbor to the best point.
        """
        if not self.neighbor_queue:
            raise RuntimeError("No more neighbors available in the queue. All nearby policies have been evaluated.")

        # Pop the next neighbor (closest one)
        policy_name, coords, distance = self.neighbor_queue.pop(0)

        if self.best_coords is not None and self.best_value is not None:
            logger.info(f"[OPT - KNN] best_policy='{self.best_policy}', best_coords=({self.best_coords[0]:.3f}, {self.best_coords[1]:.3f}), best_value={self.best_value:.4f}")
        logger.info(f"[OPT - KNN] sampled_neighbor='{policy_name}', coords=({coords[0]:.3f}, {coords[1]:.3f}), distance={distance:.4f}")

        return {
            "emb_x": float(coords[0]),
            "emb_y": float(coords[1]),
        }
