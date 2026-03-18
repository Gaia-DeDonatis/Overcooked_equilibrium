from BayesOptUtils import *
import pandas as pd
import numpy as np
import random
import logging
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects
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
from BestPolicyGenerationNode import (
    BestPolicyGenerationNode,
    best_policy_generation_node_to_dict,
    best_policy_generation_node_from_dict,
)
from logging_utils import opt_logger as logger
from botorch.models import SingleTaskGP
from ax.generators.torch.botorch_modular.surrogate import SurrogateSpec
from ax.generators.torch.botorch_modular.utils import ModelConfig
from gpytorch.constraints import Interval
from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel
from botorch.acquisition import qNoisyExpectedImprovement, qExpectedImprovement, qUpperConfidenceBound, qLogNoisyExpectedImprovement


def create_surrogate_spec(
    noise_variance: float = 0.4,
) -> SurrogateSpec:
    """
    Create a SurrogateSpec with noise configuration (serialization-safe).

    Args:
        noise_variance: Observation noise variance. Higher = more uncertainty = more exploration.
                       Default 0.4. Use 1.0+ for more exploration.

    Returns:
        SurrogateSpec configured for SingleTaskGP with noise constraint.
    """
    return SurrogateSpec(
        model_configs=[
            ModelConfig(
                botorch_model_class=SingleTaskGP,
                likelihood_options={
                    "noise_constraint": Interval(
                        lower_bound=noise_variance * 0.5,
                        upper_bound=noise_variance * 2.0,
                        initial_value=noise_variance,
                    ),
                },
            )
        ]
    )


# Alias for backwards compatibility
def create_low_noise_surrogate_spec(noise_variance: float = 0.01) -> SurrogateSpec:
    """Create a low-noise surrogate spec. See create_surrogate_spec for more options."""
    return create_surrogate_spec(noise_variance=noise_variance)

def _get_custom_encoder_registry():
    """Get encoder registry with custom generation nodes registered."""
    registry = dict(CORE_ENCODER_REGISTRY)
    registry[KNNGenerationNode] = knn_generation_node_to_dict
    registry[BestPolicyGenerationNode] = best_policy_generation_node_to_dict
    return registry


def _get_custom_decoder_registry():
    """Get decoder registry with custom generation nodes registered."""
    registry = dict(CORE_DECODER_REGISTRY)
    registry["KNNGenerationNode"] = knn_generation_node_from_dict
    registry["BestPolicyGenerationNode"] = best_policy_generation_node_from_dict
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
                        generator_kwargs={"seed": random.randint(0, 2**31 - 1)},
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

    def __init__(self, embedding_csv: str, n_init=3, n_bo=10, n_knn=10, n_best=5,
                 surrogate_spec: SurrogateSpec | None = None,
                 acqf_class=qLogNoisyExpectedImprovement,
                 acqf_options: dict | None = None,
                 verbose=True):
        """
        Args:
            embedding_csv: Path to CSV with columns ['policy', 'x', 'y']
            n_init: Number of initial random trials before BO kicks in
            n_bo_trials: Number of BO trials before switching to kNN
            n_knn_neighbors: Number of nearest neighbors to explore in kNN phase
            n_best: Number of greedy best-policy trials after kNN phase
            surrogate_spec: Custom SurrogateSpec for the GP model. Use
                           create_surrogate_spec() to configure kernel and noise.
            acqf_class: Acquisition function class. Options:
                       - qNoisyExpectedImprovement (default): handles noisy observations
                       - qExpectedImprovement: assumes noiseless observations
                       - qUpperConfidenceBound: tunable exploration via beta parameter
            acqf_options: Dict of options for the acquisition function.
                         e.g., {"beta": 2.0} for qUpperConfidenceBound
            verbose: Print debug information
        """
        self.verbose = verbose
        self.n_init = n_init
        self.n_bo_trials = n_bo
        self.n_knn_neighbors = n_knn
        self.n_best_trials = n_best
        self.surrogate_spec = surrogate_spec
        self.acqf_class = acqf_class
        self.acqf_options = acqf_options or {}

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
        # Convert to Python list to ensure consistent string types (not numpy.str_)
        self._policy_names = self.policies_df['policy'].tolist()

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

    def _map_to_nearest_policy(self, emb_x: float, emb_y: float) -> tuple:
        """Map continuous (x, y) to nearest policy (regardless of evaluation status)."""
        query_point = np.array([[emb_x, emb_y]])
        distances = cdist(query_point, self._coords, metric='euclidean')[0]
        idx = np.argmin(distances)
        return self._policy_names[idx], self._coords[idx], distances[idx]

    def ask(self, n_trials=1):
        """
        Get next trial(s) to evaluate.

        Args:
            n_trials: Number of trials to generate.

        Returns:
            dict: {trial_idx: {'policy': policy_name, 'emb_x': x, 'emb_y': y}}
        """
        if self.remaining_policies() == 0:
            if self.verbose:
                logger.info("[OPT - ASK] all policies have been evaluated!")
            return {}

        # Log status
        n_completed = len(self.evaluated_policies)
        logger.info(f"[OPT - STATUS] completed_trials={n_completed}, threshold={self.n_init} (Sobol->BO), {self.n_init + self.n_bo_trials} (BO->kNN)")

        # Get suggestions from Ax
        raw_trials = self._optimizer.ask(n_trials=n_trials)

        # Detect and log strategy switches
        new_node = self._optimizer.generation_strategy.current_node_name
        node_tag = new_node.upper().replace(" ", "_").replace("MODULAR_BOTORCH", "BO")
        if self._last_node_name is not None and new_node != self._last_node_name:
            logger.info(f"[OPT - SWITCH] {self._last_node_name} -> {new_node}")
        elif self._last_node_name is None:
            logger.info(f"[OPT - {node_tag}] starting with {new_node}")
        self._last_node_name = new_node

        is_best_policy = new_node.startswith("BestPolicy")

        mapped_trials = {}
        for suggested_trial_idx, params in raw_trials.items():
            # BestPolicy: map to nearest policy (already evaluated, for re-evaluation)
            # Other phases: map to nearest UNEVALUATED policy
            if is_best_policy:
                policy_name, coords, distance = self._map_to_nearest_policy(params['emb_x'], params['emb_y'])
            else:
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

    def get_seed_bo_trial_count(self) -> int:
        """Return the number of trials in seed+BO phase (for GP fitting)."""
        return self.n_init + self.n_bo_trials

    def get_trial_data(self, max_trials: int | None = None) -> list:
        """
        Get trial data as a list of dicts, optionally limited to first N trials.

        Args:
            max_trials: If provided, only return data for first N trials.
                       Use get_seed_bo_trial_count() to get seed+BO trials only.

        Returns:
            List of dicts with keys: trial_idx, policy, x, y, value
        """
        experiment = self._optimizer.client._experiment  # type: ignore
        data = experiment.lookup_data()

        sorted_trials = list(sorted(self.trial_to_policy.items()))
        if max_trials is not None:
            sorted_trials = sorted_trials[:max_trials]

        trial_data = []
        for trial_idx, policy in sorted_trials:
            trial = experiment.trials[trial_idx]
            params = trial.arms[0].parameters if hasattr(trial, 'arms') else trial.arm.parameters  # type: ignore
            trial_df = data.df[(data.df["trial_index"] == trial_idx) & (data.df["metric_name"] == "performance")]
            if not trial_df.empty:
                value = trial_df["mean"].item()
                trial_data.append({
                    'trial_idx': trial_idx,
                    'policy': policy,
                    'x': params['emb_x'],
                    'y': params['emb_y'],
                    'value': value
                })
        return trial_data

    def save(self, filepath):
        """
        Save optimizer state to JSON using Ax's serialization with custom KNN encoder.

        Saves both the Ax client state (including KNNGenerationNode) and
        TSNEBayesOptimizer-specific state (evaluated_policies, trial_to_policy).
        """
        encoder_registry = _get_custom_encoder_registry()

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
                    'n_best_trials': self.n_best_trials,
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

        decoder_registry = _get_custom_decoder_registry()

        # Extract TSNEBayesOptimizer-specific state
        tsne_state = state.get('tsne_optimizer_state', {})
        loaded_evaluated_policies = set(tsne_state.get('evaluated_policies', []))
        self.trial_to_policy = {int(k): v for k, v in tsne_state.get('trial_to_policy', {}).items()}

        # Restore config values (n_init, n_bo_trials, n_knn_neighbors, n_best_trials)
        config = tsne_state.get('config', {})
        if 'n_init' in config:
            self.n_init = config['n_init']
        if 'n_bo_trials' in config:
            self.n_bo_trials = config['n_bo_trials']
        if 'n_knn_neighbors' in config:
            self.n_knn_neighbors = config['n_knn_neighbors']
        if 'n_best_trials' in config:
            self.n_best_trials = config['n_best_trials']

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

            # Find custom nodes and sync their evaluated_policies reference
            for node in gen_strategy._nodes:
                if isinstance(node, KNNGenerationNode):
                    node.evaluated_policies = self.evaluated_policies
                elif isinstance(node, BestPolicyGenerationNode):
                    node.evaluated_policies = self.evaluated_policies

        # Replace the optimizer's client
        self._optimizer.client = new_client

        # Update our evaluated_policies set with loaded data
        self.evaluated_policies.clear()
        self.evaluated_policies.update(loaded_evaluated_policies)

        logger.info(f"[OPT - LOAD] loaded state from {filepath} ({len(self.evaluated_policies)} evaluated policies, n_init={self.n_init}, n_bo={self.n_bo_trials}, n_knn={self.n_knn_neighbors}, n_best={self.n_best_trials})")

    def _construct_generation_strategy_with_knn(self):
        """
        Construct generation strategy: Center -> Sobol -> BO -> kNN -> BestPolicy

        Uses configured surrogate_spec (kernel, noise) and acqf_class (acquisition function).
        """
        # Build model_kwargs with surrogate_spec and acquisition function
        model_kwargs = {
            "botorch_acqf_class": self.acqf_class,
        }
        if self.surrogate_spec is not None:
            model_kwargs["surrogate_spec"] = self.surrogate_spec
        if self.acqf_options:
            model_kwargs["acqf_options"] = self.acqf_options

        # Log configuration
        kernel_info = "default (Matern 2.5)"
        if self.surrogate_spec is not None and self.surrogate_spec.model_configs:
            mc = self.surrogate_spec.model_configs[0]
            if mc.covar_module_options and "base_kernel" in mc.covar_module_options:
                kernel_info = str(type(mc.covar_module_options["base_kernel"]).__name__)
        logger.info(f"[OPT - INIT] kernel={kernel_info}, acqf={self.acqf_class.__name__}, acqf_options={self.acqf_options}")

        # Calculate the trial cutoff for BestPolicy nodes (only consider Sobol+BO)
        # We need to account for abandoned trials too - each phase creates 2 trials (suggested + actual)
        # But we filter by COMPLETED status, so we use the completed trial count
        best_policy_max_trials = self.n_init + self.n_bo_trials

        # BestPolicy2 node - re-evaluate the same best policy after kNN (final phase)
        best_policy_node_2 = BestPolicyGenerationNode(
            coords=np.asarray(self._coords),
            policy_names=np.asarray(self._policy_names),
            evaluated_policies=self.evaluated_policies,
            generator_specs=[GeneratorSpec(
                generator_enum=Generators.BOTORCH_MODULAR,
                model_kwargs=model_kwargs,
            )],
            max_trials_to_consider=best_policy_max_trials,  # Same cutoff as BestPolicy1
            name="BestPolicy2",
        )

        # kNN node - samples nearest neighbors from best point
        knn_node = KNNGenerationNode(
            coords=np.asarray(self._coords),
            policy_names=np.asarray(self._policy_names),
            n_neighbors=self.n_knn_neighbors,
            evaluated_policies=self.evaluated_policies,
            generator_specs=[GeneratorSpec(
                generator_enum=Generators.BOTORCH_MODULAR,
                model_kwargs=model_kwargs,
            )],
            transition_criteria=[
                MinTrials(
                    threshold=self.n_init + self.n_bo_trials+self.n_knn_neighbors+1,  # Count trials generated by this node
                    transition_to="BestPolicy2",
                    only_in_statuses=[TrialStatus.COMPLETED],
                    use_all_trials_in_exp=True,  # Only count this node's trials (includes abandoned)
                )
            ],
        )
        # BestPolicy1 node - evaluate best from Sobol+BO before kNN
        best_policy_node_1 = BestPolicyGenerationNode(
            coords=np.asarray(self._coords),
            policy_names=np.asarray(self._policy_names),
            evaluated_policies=self.evaluated_policies,
            generator_specs=[GeneratorSpec(
                generator_enum=Generators.BOTORCH_MODULAR,
                model_kwargs=model_kwargs,
            )],
            max_trials_to_consider=best_policy_max_trials,  # Only consider Sobol+BO trials
            name="BestPolicy1",
            transition_criteria=[
                MinTrials(
                    threshold=self.n_init + self.n_bo_trials+1, 
                    transition_to="kNN",
                    only_in_statuses=[TrialStatus.COMPLETED],
                    use_all_trials_in_exp=True,  # Only count trials from this node
                )
            ],
        )
        # BoTorch node for Bayesian Optimization
        botorch_node = GenerationNode(
            name="Modular BoTorch",
            generator_specs=[GeneratorSpec(
                generator_enum=Generators.BOTORCH_MODULAR,
                model_kwargs=model_kwargs,
            )],
            transition_criteria=[
                MinTrials(
                    threshold=self.n_init + self.n_bo_trials,
                    transition_to="BestPolicy1",
                    only_in_statuses=[TrialStatus.COMPLETED],
                    use_all_trials_in_exp=True,
                )
            ],
        )

        # Sobol node for space-filling initialization
        sobol_node = GenerationNode(
            name="Sobol",
            generator_specs=[GeneratorSpec(
                generator_enum=Generators.SOBOL,
                generator_kwargs={"seed": random.randint(0, 2**31 - 1)},
            )],
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
            name="Center+Sobol+BO+BestPolicy1+kNN+BestPolicy2",
            nodes=[center_node, sobol_node, botorch_node, best_policy_node_1, knn_node, best_policy_node_2],
        )

    def generate_final_visualizations(self, prolific_id: str) -> None:
        """
        Generate final PDF visualizations at experiment completion.

        Call this at the end of the experiment to save:
        - Embedding trajectory with GP heatmap background and sample order
        - Ax slice plots for each parameter
        - Ax contour plot with sample order overlay

        Args:
            prolific_id: Participant ID for the save directory.
        """
        save_dir = os.path.join("submissions", prolific_id)
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"[OPT - VIZ] Generating final visualizations in {save_dir}")

        # Ensure the model is fitted for visualization
        self._ensure_model_fitted()

        # 1. Generate embedding trajectory with GP heatmap
        self._plot_embedding_with_gp_heatmap(save_dir)

        # 2. Generate Ax slice/contour plots
        self._plot_ax_surfaces(save_dir)

        logger.info(f"[OPT - VIZ] Completed generating final visualizations")

    def _ensure_model_fitted(self) -> None:
        """Ensure the current node has a fitted model for visualization."""
        try:
            gs = self._optimizer.generation_strategy
            experiment = self._optimizer.client._experiment
            if gs.adapter is None:
                gs._curr._fit(experiment=experiment)
                logger.info("[OPT - VIZ] Fitted model for visualization")
        except Exception as e:
            logger.warning(f"[OPT - VIZ] Could not fit model: {e}")

    def _plot_embedding_with_gp_heatmap(self, save_dir: str) -> None:
        """Plot t-SNE embedding with GP heatmap background and sample order annotations."""
        try:
            from matplotlib.patches import Patch

            fig, ax = plt.subplots(figsize=(14, 12))

            # Try to get GP predictions for heatmap
            heatmap_added = False
            try:
                gs = self._optimizer.generation_strategy
                adapter = gs.adapter
                if adapter is not None and hasattr(adapter, 'model'):
                    # Create grid for heatmap
                    grid_size = 50
                    x_grid = np.linspace(-1, 1, grid_size)
                    y_grid = np.linspace(-1, 1, grid_size)
                    xx, yy = np.meshgrid(x_grid, y_grid)
                    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

                    # Get predictions from the model
                    from ax.core.observation import ObservationFeatures
                    obs_features = [
                        ObservationFeatures(parameters={"emb_x": float(p[0]), "emb_y": float(p[1])})
                        for p in grid_points
                    ]
                    predictions = adapter.predict(obs_features)
                    # predictions is (means_dict, covariances_dict)
                    metric_name = list(predictions[0].keys())[0]
                    means = np.array(predictions[0][metric_name]).reshape(grid_size, grid_size)

                    # Plot heatmap
                    im = ax.contourf(xx, yy, means, levels=20, cmap='RdYlGn', alpha=0.6)
                    cbar = plt.colorbar(im, ax=ax, label='Predicted Performance')
                    heatmap_added = True
                    logger.info("[OPT - VIZ] Added GP heatmap to embedding plot")
            except Exception as e:
                logger.warning(f"[OPT - VIZ] Could not add GP heatmap: {e}")

            # Plot all policies (gray background if no heatmap, white otherwise)
            bg_color = 'white' if heatmap_added else 'lightgray'
            ax.scatter(self._coords[:, 0], self._coords[:, 1], c=bg_color, alpha=0.3, s=30,
                      edgecolors='gray', linewidths=0.3)

            # Plot evaluated policies with order numbers and color by phase
            for i, (trial_idx, policy) in enumerate(sorted(self.trial_to_policy.items())):
                idx_arr = np.where(self._policy_names == policy)[0]
                if len(idx_arr) > 0:
                    coords = self._coords[idx_arr[0]]
                    color = self._get_phase_color(i)
                    ax.scatter(coords[0], coords[1], c=color, s=120, zorder=5,
                              edgecolors='black', linewidths=1)
                    ax.annotate(str(i+1), (coords[0], coords[1]), fontsize=9, ha='center', va='center',
                               fontweight='bold', color='white',
                               path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])

            ax.set_xlabel('t-SNE Dimension 1 (emb_x)', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2 (emb_y)', fontsize=12)
            title = 'Optimization Trajectory in Embedding Space'
            if heatmap_added:
                title += '\n(Background: GP Predicted Performance)'
            ax.set_title(title, fontsize=14)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

            # Add legend
            bp1_idx = self.n_init + self.n_bo_trials + 1
            knn_end = bp1_idx + self.n_knn_neighbors
            legend_elements = [
                Patch(facecolor='green', edgecolor='black', label=f'Sobol (1-{self.n_init})'),
                Patch(facecolor='blue', edgecolor='black', label=f'BO ({self.n_init+1}-{self.n_init + self.n_bo_trials})'),
                Patch(facecolor='purple', edgecolor='black', label=f'BestPolicy1 ({bp1_idx})'),
                Patch(facecolor='red', edgecolor='black', label=f'kNN ({bp1_idx + 1}-{knn_end})'),
                Patch(facecolor='purple', edgecolor='black', label=f'BestPolicy2 ({knn_end + 1}+)'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "embedding_trajectory_with_gp.pdf"), bbox_inches='tight', dpi=150)
            plt.close()
            logger.info(f"[OPT - VIZ] Saved embedding_trajectory_with_gp.pdf")
        except Exception as e:
            logger.warning(f"[OPT - VIZ] Failed to generate embedding trajectory plot: {e}")
            import traceback
            traceback.print_exc()

    def _plot_ax_surfaces(self, save_dir: str) -> None:
        """Generate Ax slice and contour plots using the fitted surrogate model."""
        try:
            from ax.analysis.plotly.surface.slice import SlicePlot
            from ax.analysis.plotly.surface.contour import ContourPlot

            experiment = self._optimizer.client._experiment
            generation_strategy = self._optimizer.generation_strategy

            # Check if we have a fitted adapter
            adapter = generation_strategy.adapter
            if adapter is None:
                logger.warning("[OPT - VIZ] No fitted adapter available for Ax surface plots")
                return

            # Slice plots for each parameter
            for param in ["emb_x", "emb_y"]:
                try:
                    card = SlicePlot(parameter_name=param).compute(
                        experiment=experiment,
                        generation_strategy=generation_strategy,
                    )
                    # Add sample order annotations to the plot
                    fig = card.get_figure()
                    self._add_sample_annotations_to_plotly(fig, param)
                    fig.write_image(os.path.join(save_dir, f"slice_{param}.pdf"))
                    logger.info(f"[OPT - VIZ] Saved slice_{param}.pdf")
                except Exception as e:
                    logger.warning(f"[OPT - VIZ] SlicePlot for {param} failed: {e}")

            # Contour plot with sample annotations
            try:
                card = ContourPlot(
                    x_parameter_name="emb_x",
                    y_parameter_name="emb_y",
                ).compute(
                    experiment=experiment,
                    generation_strategy=generation_strategy,
                )
                # Add sample order annotations to the contour plot
                fig = card.get_figure()
                self._add_sample_annotations_to_contour(fig)
                fig.write_image(os.path.join(save_dir, "contour_with_samples.pdf"))
                logger.info(f"[OPT - VIZ] Saved contour_with_samples.pdf")
            except Exception as e:
                logger.warning(f"[OPT - VIZ] ContourPlot failed: {e}")

        except ImportError as e:
            logger.warning(f"[OPT - VIZ] Could not import Ax plotting modules: {e}")
        except Exception as e:
            logger.warning(f"[OPT - VIZ] Failed to generate Ax surface plots: {e}")

    def _get_phase_color(self, trial_order: int) -> str:
        """Get color for a trial based on its order (0-indexed)."""
        if trial_order < self.n_init:
            return 'green'  # Sobol
        elif trial_order < self.n_init + self.n_bo_trials:
            return 'blue'  # BO
        elif trial_order == self.n_init + self.n_bo_trials:
            return 'purple'  # BestPolicy1
        elif trial_order < self.n_init + self.n_bo_trials + 1 + self.n_knn_neighbors:
            return 'red'  # kNN
        else:
            return 'purple'  # BestPolicy2

    def _add_sample_annotations_to_plotly(self, fig, param: str) -> None:
        """Add sample order annotations to a plotly figure."""
        try:
            import plotly.graph_objects as go
            for i, (trial_idx, policy) in enumerate(sorted(self.trial_to_policy.items())):
                idx_arr = np.where(self._policy_names == policy)[0]
                if len(idx_arr) > 0:
                    coords = self._coords[idx_arr[0]]
                    param_val = coords[0] if param == "emb_x" else coords[1]
                    color = self._get_phase_color(i)
                    fig.add_annotation(
                        x=param_val, y=0,
                        text=str(i+1),
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=1,
                        arrowcolor=color,
                        font=dict(size=10, color=color),
                        yshift=20
                    )
        except Exception as e:
            logger.warning(f"[OPT - VIZ] Could not add annotations to slice plot: {e}")

    def _add_sample_annotations_to_contour(self, fig) -> None:
        """Add sample order annotations to a contour plot."""
        try:
            import plotly.graph_objects as go
            # Add scatter points with sample numbers
            x_vals, y_vals, texts, colors = [], [], [], []
            for i, (trial_idx, policy) in enumerate(sorted(self.trial_to_policy.items())):
                idx_arr = np.where(self._policy_names == policy)[0]
                if len(idx_arr) > 0:
                    coords = self._coords[idx_arr[0]]
                    x_vals.append(coords[0])
                    y_vals.append(coords[1])
                    texts.append(str(i+1))
                    colors.append(self._get_phase_color(i))

            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='markers+text',
                marker=dict(size=12, color=colors, line=dict(width=1, color='black')),
                text=texts,
                textposition='top center',
                textfont=dict(size=10, color='black'),
                name='Samples',
                showlegend=True
            ))
        except Exception as e:
            logger.warning(f"[OPT - VIZ] Could not add annotations to contour plot: {e}")

    def close(self):
        self._optimizer.close()
