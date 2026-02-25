from BayesOptUtils import *
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from ax.api.client import Client

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

        for parameters, data in preexisting_trials:
            trial_index = self.client.attach_trial(parameters=parameters)
            self.client.complete_trial(trial_index=trial_index, raw_data=utlity)

    
    def tell(self, trials):
        for trial_index, utility in trials.items():
            if self.verbose:
                print("Trial Idx:", trial_index)
                print("Utility:", utility)
            self.client.complete_trial(trial_index=trial_index, raw_data=utility)

        
    def ask(self, n_trials=1, print_model=False):
        r = self.client.get_next_trials(max_trials=n_trials)



        # Optionally print model info right after generation
        if self.verbose:
            self.print_generator_config()
            
        if self.verbose:
            print("Next Trial:", r)

        return r


    def _print_kernel_structure(self, kernel, indent=""):
        """Helper to print kernel structure clearly."""
        kernel_type = type(kernel).__name__
        print(f"{indent}{kernel_type}")

        # For composite/product kernels
        if hasattr(kernel, 'kernels'):
            print(f"{indent}  Components:")
            for i, k in enumerate(kernel.kernels):
                print(f"{indent}    [{i}] {type(k).__name__}")
                # If it's a scaled kernel, show the base
                if hasattr(k, 'base_kernel'):
                    print(f"{indent}        base: {type(k.base_kernel).__name__}")

        # For scaled kernels
        elif hasattr(kernel, 'base_kernel'):
            print(f"{indent}  Base: {type(kernel.base_kernel).__name__}")
    
    def get_best(self):
        best_parameters, prediction, index, name = self.client.get_best_parameterization()
        
        if self.verbose:
            print("Best Parameters:", best_parameters)
            print("Prediction (mean, variance):", prediction)
            print("Index:", index)
            print("Name:", name)
        return best_parameters, prediction, index, name
    
    def compute_analyses(self, display=True):
        # display=True instructs Ax to sort then render the resulting analyses
        cards = self.client.compute_analyses(analyses=[OverviewAnalysis()], display=display)
        return cards

    def print_model_info(self):
        """Print summary of the Bayesian Optimization configuration."""
        print("\n" + "=" * 80)
        print("BAYESIAN OPTIMIZATION SUMMARY")
        print("=" * 80)

        # Show generation strategy status
        print(f"\nGeneration Strategy: {self.generation_strategy.name}")
        print(f"Current Node: {self.generation_strategy.current_node_name}")
        print(f"Total Trials: {len(self.client._experiment.trials)}")

        # Show parameter configuration
        print("\n" + "-" * 80)
        print("SEARCH SPACE")
        print("-" * 80)
        for param in self.parameters:
            ordered_str = f" (ordered)" if hasattr(param, 'is_ordered') and param.is_ordered else ""
            print(f"  {param.name}: {param.parameter_type}{ordered_str}")

            # Handle different parameter types
            if hasattr(param, 'values'):
                print(f"    values: {param.values}")
            elif hasattr(param, 'bounds'):
                print(f"    bounds: {param.bounds}")

        # Show model configuration
        self.print_generator_config()


    def save(self, filepath):
        self.client.save_to_json_file(filepath)
    
    def load(self, filepath):
        self.cleint = Client.load_from_json_file(filepath = filepath)

    def print_generator_config(self):
        """Print the Bayesian Optimization model configuration."""
        print("\n" + "=" * 80)
        print("BAYESIAN OPTIMIZATION MODEL CONFIGURATION")
        print("=" * 80)

        gs = self.generation_strategy
        print(f"\nGeneration Node: {gs.current_node_name}")

        if not hasattr(gs, 'adapter') or not gs.adapter:
            print("No adapter found (still in initialization phase)")
            print("=" * 80 + "\n")
            return

        adapter = gs.adapter
        generator = adapter.generator

        # === SHOW TRANSFORMATIONS ===
        print("\n" + "-" * 80)
        print("PARAMETER TRANSFORMATIONS")
        print("-" * 80)

        # Try to find transforms in various locations
        transforms_found = False
        for attr in ['_transforms', 'transforms', '_model_bridge']:
            if hasattr(adapter, attr):
                obj = getattr(adapter, attr, None)
                if obj is not None:
                    if isinstance(obj, list) and len(obj) > 0:
                        print(f"\nTransforms (from adapter.{attr}): {len(obj)}")
                        for i, transform in enumerate(obj):
                            print(f"  [{i}] {type(transform).__name__}")
                        transforms_found = True
                        break

        if not transforms_found:
            print("\nNo explicit transform list found")

        # Show search space digest which tells us how parameters are handled
        print(f"\nSearch Space Digest:")
        if hasattr(generator, 'search_space_digest'):
            ssd = generator.search_space_digest
            print(f"  Type: {type(ssd).__name__}")

            # Print all relevant attributes
            for attr in ['feature_names', 'bounds', 'categorical_features', 'discrete_features',
                        'ordinal_features', 'task_features', 'fidelity_features']:
                if hasattr(ssd, attr):
                    val = getattr(ssd, attr, None)
                    if val is not None and (not isinstance(val, (list, dict)) or len(val) > 0):
                        print(f"  {attr}: {val}")
        else:
            print("  Not available")

        # === SURROGATE (GP) MODEL ===
        print("\n" + "-" * 80)
        print("GAUSSIAN PROCESS MODEL")
        print("-" * 80)

        if hasattr(generator, '_surrogate'):
            surrogate = generator._surrogate

            # Check for the actual fitted model
            if hasattr(surrogate, 'model') and surrogate.model is not None:
                botorch_model = surrogate.model
                print(f"\nGP Class: {type(botorch_model).__name__}")

                # Extract kernel information
                if hasattr(botorch_model, 'covar_module'):
                    print(f"\nKernel Configuration:")
                    self._print_kernel_structure(botorch_model.covar_module, indent="  ")

                # Extract mean module
                if hasattr(botorch_model, 'mean_module'):
                    print(f"\nMean Function: {type(botorch_model.mean_module).__name__}")

                # Extract likelihood
                if hasattr(botorch_model, 'likelihood'):
                    print(f"Likelihood: {type(botorch_model.likelihood).__name__}")

            else:
                print("\nGP model not yet fitted (will be created during generation)")

        # === ACQUISITION FUNCTION ===
        print("\n" + "-" * 80)
        print("ACQUISITION FUNCTION")
        print("-" * 80)

        if hasattr(generator, '_acquisition'):
            acquisition = generator._acquisition

            # Try to find the acquisition function class
            acqf_class = None
            for attr in ['_acqf_class', 'acqf_class', '_default_botorch_acqf_class']:
                if hasattr(acquisition, attr):
                    acqf_class = getattr(acquisition, attr, None)
                    if acqf_class is not None and hasattr(acqf_class, '__name__'):
                        print(f"\nAcquisition Class: {acqf_class.__name__}")
                        break

            if acqf_class is None:
                # Check acquisition options for clues
                if hasattr(acquisition, 'options') and acquisition.options:
                    print(f"\nAcquisition Options: {acquisition.options}")
                else:
                    print("\nAcquisition Class: qLogNoisyExpectedImprovement (default)")

        print("\n" + "=" * 80 + "\n")
        
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

    def __init__(self, embedding_csv: str, n_init=3, verbose=True):
        """
        Args:
            embedding_csv: Path to CSV with columns ['policy', 'x', 'y']
            n_init: Number of initial random trials before BO kicks in
            verbose: Print debug information
        """
        self.verbose = verbose

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

        # Initialize underlying optimizer
        self._optimizer = BayesOptimizer(parameters, n_init=n_init, verbose=False)

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
            print(f"Loaded {len(self._policy_names)} policies from embedding space")
            print(f"Original x range: [{x_min:.3f}, {x_max:.3f}] -> normalized to [-1, 1]")
            print(f"Original y range: [{y_min:.3f}, {y_max:.3f}] -> normalized to [-1, 1]")

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
                print("All policies have been evaluated!")
            return {}

        # Get suggestions from underlying BO (this guides where to look)
        raw_trials = self._optimizer.ask(n_trials=n_trials)

        mapped_trials = {}
        for suggested_trial_idx, params in raw_trials.items():
            policy_name, coords, distance = self._map_to_policy(params['emb_x'], params['emb_y'])

            if policy_name is None:
                if self.verbose:
                    print(f"Trial {suggested_trial_idx}: No unevaluated policies remaining")
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

            if self.verbose:
                print(f"Trial {actual_trial_idx}: BO suggested ({params['emb_x']:.3f}, {params['emb_y']:.3f})")
                print(f"  -> Mapped to policy '{policy_name}' at ({coords[0]:.3f}, {coords[1]:.3f})")
                print(f"  -> Distance: {distance:.4f}")
                print(f"  -> Attached trial at ACTUAL coordinates (not suggested)")
        self._actual_trial_idx = actual_trial_idx
        print("MAPPED_TRIAL", mapped_trials)
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
                if self.verbose:
                    print(f"Marked policy '{policy_name}' as evaluated")

            # Format the result
            if isinstance(value, dict):
                raw_data = value
            else:
                raw_data = {'performance': value}

            # Complete the trial at the ACTUAL coordinates
            self._optimizer.client.complete_trial(trial_index=trial_idx, raw_data=raw_data)

            if self.verbose:
                print(f"Completed trial {trial_idx} with result: {raw_data}")

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
            print(f"Best Policy: {best_policy}")
            print(f"Embedding coords: ({best_params['emb_x']:.3f}, {best_params['emb_y']:.3f})")
            print(f"Prediction (mean, variance): {prediction}")

        return best_policy, best_params, prediction, index, name

    def get_evaluated_policies(self) -> list:
        """Return list of all evaluated policies with their trial indices."""
        return [(idx, policy) for idx, policy in self.trial_to_policy.items()]

    def print_model_info(self):
        """Print summary of the t-SNE Bayesian Optimization configuration."""
        print("\n" + "=" * 80)
        print("T-SNE BAYESIAN OPTIMIZATION SUMMARY")
        print("=" * 80)
        print(f"\nTotal policies: {len(self._policy_names)}")
        print(f"Evaluated: {len(self.evaluated_policies)}")
        print(f"Remaining: {self.remaining_policies()}")
        print("\nEmbedding space: 2D t-SNE (normalized to [-1, 1])")
        self._optimizer.print_model_info()

    def compute_analyses(self, display=True):
        """Generate analysis plots."""
        return self._optimizer.compute_analyses(display=display)

    def save(self, filepath):
        """Save optimizer state."""
        self._optimizer.save(filepath)
        
    def load(self, filepath):
        self._optimizer.load(filepath)
    
    def close(self):
        self._optimizer.close()
