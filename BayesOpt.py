from BayesOptUtils import *
import pandas as pd
import numpy as np
import logging
import json
from scipy.spatial.distance import cdist
from ax.api.client import Client
from ax.storage.json_store.encoder import object_to_json
from ax.storage.json_store.decoder import object_from_json, generation_strategy_from_json
from ax.storage.json_store.registry import (
    CORE_ENCODER_REGISTRY,
    CORE_DECODER_REGISTRY,
    CORE_CLASS_ENCODER_REGISTRY,
    CORE_CLASS_DECODER_REGISTRY,
)
from KNNGenerationNode import (
    KNNGenerationNode,
    knn_generation_node_to_dict,
    knn_generation_node_from_dict,
)
from logging_utils import opt_logger as logger


def _get_knn_encoder_registry():
    """Get encoder registry with KNNGenerationNode registered."""
    registry = dict(CORE_ENCODER_REGISTRY)
    registry[KNNGenerationNode] = knn_generation_node_to_dict
    return registry


def _get_knn_decoder_registry():
    """Get decoder registry with KNNGenerationNode registered."""
    registry = dict(CORE_DECODER_REGISTRY)
    registry["KNNGenerationNode"] = knn_generation_node_from_dict
    return registry

# Suppress Ax library info logs
logging.getLogger("ax").setLevel(logging.WARNING)
logging.getLogger("ax.api").setLevel(logging.WARNING)
logging.getLogger("ax.api.client").setLevel(logging.WARNING)
logging.getLogger("ax.core").setLevel(logging.WARNING)
logging.getLogger("ax.modelbridge").setLevel(logging.WARNING)
logging.getLogger("ax.service").setLevel(logging.WARNING)

class BayesOptimizer:
    def __init__(self, parameters: list, preexisting_trials = [], n_init =3, verbose=True) -> None:

        self.client = Client()
        self.parameters = parameters
        self.n_init = n_init
        self.client.configure_experiment(
            parameters=self.parameters,
        )

        self.generation_strategy = self._construct_generation_strategy()
        self.client.set_generation_strategy(generation_strategy=self.generation_strategy)

        self.client.configure_optimization(objective="performance") #maximizes, "-performance" to minimize

        self.verbose = verbose
        self._last_node_name = None  # Track for detecting strategy switches

        for parameters, data in preexisting_trials:
            trial_index = self.client.attach_trial(parameters=parameters)
            self.client.complete_trial(trial_index=trial_index, raw_data=data)

    
    def tell(self, trials):
        for trial_index, utility in trials.items():
            if self.verbose:
                logger.info(f"[OPT - TELL] trial_idx={trial_index}, utility={utility}")
            self.client.complete_trial(trial_index=trial_index, raw_data=utility)

        
    def ask(self, n_trials=1, print_model=False):
        # Check for strategy switch before getting trials
        current_node = self.generation_strategy.current_node_name

        r = self.client.get_next_trials(max_trials=n_trials)

        # Check if strategy switched after getting trials
        new_node = self.generation_strategy.current_node_name
        if self._last_node_name is not None and new_node != self._last_node_name:
            logger.info(f"[OPT - SWITCH] {self._last_node_name} -> {new_node}")
        elif self._last_node_name is None:
            logger.info(f"[OPT - INIT] starting with {current_node}")
        self._last_node_name = new_node

        # Optionally print model info right after generation
        if self.verbose:
            self.print_generator_config()

        node_tag = new_node.upper().replace(" ", "_").replace("MODULAR_BOTORCH", "BO")
        if self.verbose:
            logger.info(f"[OPT - {node_tag}] next_trial={r}")

        return r

    def get_best(self):
        best_parameters, prediction, index, name = self.client.get_best_parameterization()

        if self.verbose:
            logger.info(f"[OPT - BEST] params={best_parameters}, prediction={prediction}, idx={index}")
        return best_parameters, prediction, index, name
    
    def compute_analyses(self, display=True):
        # display=True instructs Ax to sort then render the resulting analyses
        cards = self.client.compute_analyses(analyses=[OverviewAnalysis()], display=display)
        return cards


    def save(self, filepath):
        self.client.save_to_json_file(filepath)
    
    def load(self, filepath):
        self.client = Client.load_from_json_file(filepath=filepath)

        
    def plot_all_slices(self, display=True):
        from ax.analysis.plotly import SlicePlot
        param_names = [p.name for p in self.parameters]
        slice_analyses = [SlicePlot(parameter_name=name) for name in param_names]
        return self.client.compute_analyses(analyses=slice_analyses, display=display)

    def _construct_generation_strategy(self):
        def construct_generation_strategy(
            generator_spec: GeneratorSpec, node_name: str,
        ) -> GenerationStrategy:
            """Constructs a Center + Sobol + Modular BoTorch `GenerationStrategy`
            using the provided `generator_spec` for the Modular BoTorch node.
            """
            botorch_node = GenerationNode(
                name=node_name,
                generator_specs=[generator_spec],
            )
            sobol_node = GenerationNode(
                name="Sobol",
                generator_specs=[
                    GeneratorSpec(
                        generator_enum=Generators.SOBOL,
                    ),
                ],
                transition_criteria=[
                    # Transition to BoTorch node once there are n_init trials on the experiment.
                    MinTrials(
                        threshold=self.n_init,
                        transition_to=botorch_node.name,
                        use_all_trials_in_exp=True,
                    )
                ]
            )
            # Center node is a customized node that uses a simplified logic and has a
            # built-in transition criteria that transitions after generating once.
            center_node = CenterGenerationNode(next_node_name=sobol_node.name)
            return GenerationStrategy(
                name=f"Center+Sobol+{node_name}",
                nodes=[center_node, sobol_node, botorch_node]
            )

        # Let's construct the simplest version with all defaults.
        generation_strategy = construct_generation_strategy(
            generator_spec=GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR),
            node_name="Modular BoTorch",
        )
        return generation_strategy

    def close(self):
        """Clean up optimizer resources."""
        # Save state if needed, clear references
        self.client = None
        
class TSNEBayesOptimizer:
    """
    Bayesian Optimizer that operates in t-SNE embedding space.

    Maps continuous BO suggestions to nearest unevaluated discrete policies.
    """

    def __init__(self, embedding_csv: str, n_init=3, n_bo=10, n_knn=10, verbose=True):
        """
        Args:
            embedding_csv: Path to CSV with columns ['policy', 'x', 'y']
            n_init: Number of initial random trials before BO kicks in
            n_bo_trials: Number of BO trials before switching to kNN
            n_knn_neighbors: Number of nearest neighbors to explore in kNN phase
            verbose: Print debug information
        """
        self.verbose = verbose
        self.n_init = n_init
        self.n_bo_trials = n_bo
        self.n_knn_neighbors = n_knn

        # Load and normalize embeddings
        self.policies_df = pd.read_csv(embedding_csv)
        self._normalize_embeddings()

        # Track evaluated policies
        self.evaluated_policies = set()
        self.trial_to_policy = {}  # Maps trial_idx -> policy_name
        self._suggested_trial_to_actual = {}  # Maps suggested trial_idx -> actual trial_idx
        self._actual_trial_idx = None

        # Create continuous parameter space over normalized embeddings
        parameters = [
            RangeParameterConfig(name="emb_x", parameter_type="float", bounds=(-1.0, 1.0)),
            RangeParameterConfig(name="emb_y", parameter_type="float", bounds=(-1.0, 1.0)),
        ]

        # Initialize underlying optimizer with custom generation strategy
        self._optimizer = BayesOptimizer(parameters, n_init=n_init, verbose=False)

        # Replace the generation strategy with one that includes kNN
        self._optimizer.generation_strategy = self._construct_generation_strategy_with_knn()
        self._optimizer.client.set_generation_strategy(generation_strategy=self._optimizer.generation_strategy)

        # Track strategy switches
        self._last_node_name = None

    def _normalize_embeddings(self):
        """Normalize x, y coordinates to [-1, 1] range."""
        x_min, x_max = self.policies_df['x'].min(), self.policies_df['x'].max()
        y_min, y_max = self.policies_df['y'].min(), self.policies_df['y'].max()

        # Add small epsilon to avoid division issues if all points are identical
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        self.policies_df['x_norm'] = 2 * (self.policies_df['x'] - x_min) / x_range - 1
        self.policies_df['y_norm'] = 2 * (self.policies_df['y'] - y_min) / y_range - 1

        # Store coordinate arrays for fast lookup
        self._coords = self.policies_df[['x_norm', 'y_norm']].values
        self._policy_names = self.policies_df['policy'].values

        if self.verbose:
            logger.info(f"[OPT - INIT] loaded {len(self._policy_names)} policies from embedding space")
            logger.info(f"[OPT - INIT] x_range=[{x_min:.3f}, {x_max:.3f}], y_range=[{y_min:.3f}, {y_max:.3f}] -> normalized to [-1, 1]")

    def _map_to_policy(self, emb_x: float, emb_y: float) -> tuple:
        """Map continuous (x, y) to nearest unevaluated policy."""
        query_point = np.array([[emb_x, emb_y]])
        distances = cdist(query_point, self._coords, metric='euclidean')[0]

        # Sort by distance and find nearest unevaluated
        for idx in np.argsort(distances):
            policy_name = self._policy_names[idx]
            if policy_name not in self.evaluated_policies:
                coords = self._coords[idx]
                return policy_name, coords, distances[idx]

        return None, None, None

    def ask(self, n_trials=1):
        """
        Get next trial(s) to evaluate.

        Returns:
            dict: {trial_idx: {'policy': policy_name, 'emb_x': x, 'emb_y': y}}
        """
        if self.remaining_policies() == 0:
            if self.verbose:
                logger.info("[OPT - ASK] all policies have been evaluated!")
            return {}

        # Check for strategy switch
        current_node = self._optimizer.generation_strategy.current_node_name

        # Log completed trial count for debugging transitions
        n_completed = len(self.evaluated_policies)
        logger.info(f"[OPT - STATUS] completed_trials={n_completed}, threshold={self.n_init} (Sobol->BO), {self.n_init + self.n_bo_trials} (BO->kNN)")

        # Get suggestions from underlying BO (this guides where to look)
        raw_trials = self._optimizer.ask(n_trials=n_trials)

        # Detect and log strategy switches
        new_node = self._optimizer.generation_strategy.current_node_name
        node_tag = new_node.upper().replace(" ", "_").replace("MODULAR_BOTORCH", "BO")
        if self._last_node_name is not None and new_node != self._last_node_name:
            logger.info(f"[OPT - SWITCH] {self._last_node_name} -> {new_node}")
        elif self._last_node_name is None:
            logger.info(f"[OPT - {node_tag}] starting with {current_node}")
        self._last_node_name = new_node

        mapped_trials = {}
        for suggested_trial_idx, params in raw_trials.items():
            policy_name, coords, distance = self._map_to_policy(params['emb_x'], params['emb_y'])

            if policy_name is None:
                if self.verbose:
                    logger.info(f"[OPT - {node_tag}] trial {suggested_trial_idx}: no unevaluated policies remaining")
                continue

            # IMPORTANT: Abandon the suggested trial and attach a new one at the ACTUAL
            # policy coordinates. This ensures the GP learns from the correct locations.
            self._optimizer.client.mark_trial_abandoned(trial_index=suggested_trial_idx)

            actual_params = {'emb_x': float(coords[0]), 'emb_y': float(coords[1])}
            actual_trial_idx = self._optimizer.client.attach_trial(parameters=actual_params)

            # Store mapping for tell()
            self.trial_to_policy[actual_trial_idx] = policy_name
            self._suggested_trial_to_actual[suggested_trial_idx] = actual_trial_idx

            mapped_trials[actual_trial_idx] = {
                'policy': policy_name,
                'emb_x': coords[0],
                'emb_y': coords[1],
                '_suggested_x': params['emb_x'],  # Original BO suggestion (for debugging)
                '_suggested_y': params['emb_y'],
                '_distance': distance,
            }

        self._actual_trial_idx = actual_trial_idx
        logger.info(f"[OPT - ASK - {node_tag}] sampled_trial={mapped_trials}")
        return mapped_trials

    def tell(self, results: dict):
        """
        Report evaluation results.

        Args:
            results: {trial_idx: utility_value} or {trial_idx: {'performance': value}}
        """
        for trial_idx, value in results.items():
            # Mark policy as evaluated
            if trial_idx in self.trial_to_policy:
                policy_name = self.trial_to_policy[trial_idx]
                self.evaluated_policies.add(policy_name)

            # Format the result
            if isinstance(value, dict):
                raw_data = value
            else:
                raw_data = {'performance': value}

            # Complete the trial at the ACTUAL coordinates
            self._optimizer.client.complete_trial(trial_index=trial_idx, raw_data=raw_data)

            if self.verbose:
                logger.info(f"[OPT - TELL] completed trial {trial_idx} with result={raw_data} and  marked policy '{policy_name}' as evaluated")

    def remaining_policies(self) -> int:
        """Return count of unevaluated policies."""
        return len(self._policy_names) - len(self.evaluated_policies)

    def get_best(self):
        """Get the best policy found so far."""
        best_params, prediction, index, name = self._optimizer.get_best()

        # Find which policy this corresponds to
        if index in self.trial_to_policy:
            best_policy = self.trial_to_policy[index]
        else:
            # Map the best embedding coords to policy
            best_policy, _, _ = self._map_to_policy(best_params['emb_x'], best_params['emb_y'])

        if self.verbose:
            logger.info(f"[OPT - BEST] policy={best_policy}, coords=({best_params['emb_x']:.3f}, {best_params['emb_y']:.3f}), prediction={prediction}")

        return best_policy, best_params, prediction, index, name

    def get_evaluated_policies(self) -> list:
        """Return list of all evaluated policies with their trial indices."""
        return [(idx, policy) for idx, policy in self.trial_to_policy.items()]

    def print_model_info(self):
        """Print summary of the t-SNE Bayesian Optimization configuration."""
        logger.info("[OPT - INFO] " + "=" * 60)
        logger.info("[OPT - INFO] T-SNE BAYESIAN OPTIMIZATION SUMMARY")
        logger.info("[OPT - INFO] " + "=" * 60)
        logger.info(f"[OPT - INFO] total_policies={len(self._policy_names)}, evaluated={len(self.evaluated_policies)}, remaining={self.remaining_policies()}")
        logger.info("[OPT - INFO] embedding_space=2D t-SNE (normalized to [-1, 1])")

    def compute_analyses(self, display=True):
        """Generate analysis plots."""
        return self._optimizer.compute_analyses(display=display)

    def save(self, filepath):
        """
        Save optimizer state to JSON using Ax's serialization with custom KNN encoder.

        Saves both the Ax client state (including KNNGenerationNode) and
        TSNEBayesOptimizer-specific state (evaluated_policies, trial_to_policy).
        """
        encoder_registry = _get_knn_encoder_registry()

        # Serialize experiment and generation strategy with custom encoder
        state = {
            "_type": "Client",
            "experiment": object_to_json(
                self._optimizer.client._experiment,
                encoder_registry=encoder_registry,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            ),
            "generation_strategy": object_to_json(
                self._optimizer.client._generation_strategy,
                encoder_registry=encoder_registry,
                class_encoder_registry=CORE_CLASS_ENCODER_REGISTRY,
            ) if self._optimizer.client._generation_strategy is not None else None,
            # TSNEBayesOptimizer-specific state
            "tsne_optimizer_state": {
                'evaluated_policies': list(self.evaluated_policies),
                'trial_to_policy': {str(k): v for k, v in self.trial_to_policy.items()},
                'config': {
                    'n_init': self.n_init,
                    'n_bo_trials': self.n_bo_trials,
                    'n_knn_neighbors': self.n_knn_neighbors,
                },
            },
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"[OPT - SAVE] saved state to {filepath} ({len(self.evaluated_policies)} evaluated policies)")

    def load(self, filepath):
        """
        Load optimizer state from JSON using Ax's deserialization with custom KNN decoder.

        Restores the full Ax client state (including generation strategy with KNNGenerationNode)
        and TSNEBayesOptimizer-specific state.
        """
        with open(filepath, 'r') as f:
            state = json.load(f)

        decoder_registry = _get_knn_decoder_registry()

        # Extract TSNEBayesOptimizer-specific state
        tsne_state = state.get('tsne_optimizer_state', {})
        loaded_evaluated_policies = set(tsne_state.get('evaluated_policies', []))
        self.trial_to_policy = {int(k): v for k, v in tsne_state.get('trial_to_policy', {}).items()}

        # Deserialize experiment
        experiment = object_from_json(
            state.get("experiment"),
            decoder_registry=decoder_registry,
            class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
        )

        # Deserialize generation strategy using the specialized function
        gen_strategy = None
        if state.get("generation_strategy") is not None:
            gen_strategy = generation_strategy_from_json(
                generation_strategy_json=state.get("generation_strategy"),
                experiment=experiment,
                decoder_registry=decoder_registry,
                class_decoder_registry=CORE_CLASS_DECODER_REGISTRY,
            )

        # Create a new client and set the experiment and generation strategy
        new_client = Client()
        new_client.set_experiment(experiment=experiment)
        if gen_strategy is not None:
            new_client.set_generation_strategy(generation_strategy=gen_strategy)
            self._optimizer.generation_strategy = gen_strategy

            # Find the KNNGenerationNode and sync its evaluated_policies reference
            for node in gen_strategy._nodes:
                if isinstance(node, KNNGenerationNode):
                    node.evaluated_policies = self.evaluated_policies
                    break

        # Replace the optimizer's client
        self._optimizer.client = new_client

        # Update our evaluated_policies set with loaded data
        self.evaluated_policies.clear()
        self.evaluated_policies.update(loaded_evaluated_policies)

        logger.info(f"[OPT - LOAD] loaded state from {filepath} ({len(self.evaluated_policies)} evaluated policies)")

    def _construct_generation_strategy_with_knn(self):
        """
        Construct generation strategy: Center -> Sobol -> BO -> kNN
        """
        # kNN node - samples nearest neighbors from best point
        knn_node = KNNGenerationNode(
            coords=np.asarray(self._coords),
            policy_names=np.asarray(self._policy_names),
            n_neighbors=self.n_knn_neighbors,
            evaluated_policies=self.evaluated_policies,  # Shared reference
            generator_specs=[GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)],
            
        )

        # BoTorch node for Bayesian Optimization
        botorch_node = GenerationNode(
            name="Modular BoTorch",
            generator_specs=[GeneratorSpec(generator_enum=Generators.BOTORCH_MODULAR)],
            transition_criteria=[
                MinTrials(
                    threshold=self.n_init + self.n_bo_trials,
                    transition_to="kNN",
                    only_in_statuses=[TrialStatus.COMPLETED],
                    use_all_trials_in_exp=True,  # Count all completed trials, not just this node's
                )
            ],
        )

        # Sobol node for space-filling initialization
        sobol_node = GenerationNode(
            name="Sobol",
            generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
            transition_criteria=[
                MinTrials(
                    threshold=self.n_init,
                    transition_to="Modular BoTorch",
                    only_in_statuses=[TrialStatus.COMPLETED],
                    use_all_trials_in_exp=True,  # Count all completed trials, not just this node's
                )
            ],
        )

        # Center node generates the center point first
        center_node = CenterGenerationNode(next_node_name="Sobol")

        return GenerationStrategy(
            name="Center+Sobol+BO+kNN",
            nodes=[center_node, sobol_node, botorch_node, knn_node],
        )

    def close(self):
        self._optimizer.close()
