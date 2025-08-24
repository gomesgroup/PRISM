#
# MIT License
#
# Copyright (c) 2024 HTE RT team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""HTE Regression Transformer algorithm.

HTE Regression Transformer is a specialized model for predicting and generating
chemical reactions conditioned on High-Throughput Experimentation (HTE) rates
and molecular descriptors.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterable, Optional, TypeVar, Union

from typing_extensions import Protocol, runtime_checkable

# Note: These imports would need to be adapted for actual GT4SD integration
# from ....domains.materials import Molecule, Sequence
# from ....exceptions import InvalidItem
# from ....properties.core import PropertyValue
# from ...core import AlgorithmConfiguration, GeneratorAlgorithm
# from ...registry import ApplicationsRegistry

from .implementation import HTEConditionalGenerator

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Type variables for GT4SD compatibility
T = TypeVar("T", bound=str)  # Target type (property conditions)
S = TypeVar("S", str, dict)  # Sample type (generated molecules)
Targeted = Callable[[T], Iterable[Any]]


# Mock classes for standalone implementation
class AlgorithmConfiguration:
    """Mock AlgorithmConfiguration for standalone implementation."""
    
    algorithm_type: ClassVar[str]
    domain: ClassVar[str]
    
    def ensure_artifacts(self) -> str:
        """Ensure model artifacts are available."""
        # In real GT4SD, this would download/cache models
        return "/path/to/model/artifacts"
    
    def get_conditional_generator(self, resources_path: str, context: Optional[str] = None):
        """Get the conditional generator instance."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def validate_item(self, item: Any):
        """Validate generated item."""
        return item


class GeneratorAlgorithm:
    """Mock GeneratorAlgorithm for standalone implementation."""
    
    max_samples: int = 50
    
    def __init__(self, configuration, target):
        self.configuration = configuration
        self.target = target
        self.local_artifacts = None
    
    def sample(self, number_samples: int = 10):
        """Generate samples."""
        generator = self.get_generator(self.configuration, self.target)
        return generator(number_samples)
    
    def get_generator(self, configuration, target):
        """Get generator function."""
        raise NotImplementedError("Subclasses must implement this method")


class ApplicationsRegistry:
    """Mock registry for standalone implementation."""
    
    _registry = {}
    
    @classmethod
    def register_algorithm_application(cls, algorithm_class):
        """Register algorithm application."""
        def decorator(config_class):
            cls._registry[config_class.__name__] = {
                'algorithm': algorithm_class,
                'configuration': config_class
            }
            return config_class
        return decorator
    
    @classmethod
    def get_application_instance(cls, algorithm_name: str, **kwargs):
        """Get algorithm instance."""
        if algorithm_name in cls._registry:
            entry = cls._registry[algorithm_name]
            config = entry['configuration'](**kwargs)
            return entry['algorithm'](configuration=config, target=kwargs.get('target'))
        raise ValueError(f"Algorithm {algorithm_name} not found in registry")


class HTERegressionTransformer(GeneratorAlgorithm):
    """HTE Regression Transformer Algorithm.
    
    This algorithm generates chemical reactions conditioned on High-Throughput
    Experimentation (HTE) rates and molecular descriptors. It can both predict
    HTE rates for given reactions and generate reactions with target HTE rates.
    """

    #: The maximum number of samples a user can try to run in one go
    max_samples: int = 50

    def __init__(
        self,
        configuration: AlgorithmConfiguration,
        target: Optional[T],
    ) -> None:
        """Instantiate HTE Regression Transformer ready to generate items.

        Args:
            configuration: domain and application specification defining 
                parameters, types and validations.
            target: a target for which to generate items. Can be:
                - Dict with HTE rate target: {"hte_rate": 0.5}
                - String with property conditioning: "<hte>0.5|<d0>1.2<d1>-0.3"
                - Dict with descriptors: {"d0": 1.2, "d1": -0.3, "hte_rate": 0.5}

        Example:
            Generate molecules with high HTE rates::

                config = HTERegressionTransformerMolecules(
                    search='sample', temperature=0.8, tolerance=10
                )
                target = {"hte_rate": 1.5, "d0": 0.5, "d1": -0.2}
                hte_generator = HTERegressionTransformer(
                    configuration=config, target=target
                )
                items = list(hte_generator.sample(10))
                print(items)
        """

        configuration = self.validate_configuration(configuration)

        # No validation/check on the target input here, since model is not yet loaded.
        super().__init__(
            configuration=configuration,
            target=target,
        )

    def get_generator(
        self,
        configuration: AlgorithmConfiguration,
        target: Optional[T],
    ) -> Targeted[T]:
        """Get the function to sample with the given configuration.

        Args:
            configuration: helps to set up specific application of HTE RT.
            target: context or condition for the generation.

        Returns:
            callable with target generating a batch of items.
        """
        logger.info("ensure artifacts for the application are present.")
        self.local_artifacts = configuration.ensure_artifacts()
        implementation: HTEConditionalGenerator = configuration.get_conditional_generator(
            resources_path=self.local_artifacts, context=target
        )
        
        # Adjust max_samples based on task type
        if implementation.task == "regression" and configuration.search == "greedy":
            self.max_samples = 1
            logger.warning(
                "max_samples was set to 1 due to regression task and greedy search"
            )

        return implementation.generate_batch

    def validate_configuration(
        self, configuration: AlgorithmConfiguration
    ) -> AlgorithmConfiguration:
        """Validate the configuration."""
        
        @runtime_checkable
        class AnyHTERegressionTransformerConfiguration(Protocol):
            """Protocol for HTE Regression Transformer configurations."""

            def get_conditional_generator(
                self, resources_path: str, context: Optional[str] = None
            ) -> HTEConditionalGenerator:
                ...

            def validate_item(self, item: Any) -> S:
                ...

        # TODO: Implement proper validation and raise InvalidAlgorithmConfiguration
        assert isinstance(configuration, AnyHTERegressionTransformerConfiguration)
        assert isinstance(configuration, AlgorithmConfiguration)
        return configuration


@ApplicationsRegistry.register_algorithm_application(HTERegressionTransformer)
@dataclass
class HTERegressionTransformerMolecules(AlgorithmConfiguration):
    """
    Configuration to generate molecules given HTE rate targets and molecular descriptors.

    Implementation extends the Regression Transformer for High-Throughput Experimentation
    rate prediction and conditional generation.

    Examples:
        Generate molecules with target HTE rate::

            config = HTERegressionTransformerMolecules(
                algorithm_version='hte_v1', search='sample', temperature=0.8, tolerance=5
            )
            target = {"hte_rate": 0.5, "d0": 1.2, "d1": -0.3}
            hte_generator = HTERegressionTransformer(
                configuration=config, target=target
            )
            molecules = list(hte_generator.sample(5))

        Predict HTE rate for a given reaction::

            config = HTERegressionTransformerMolecules(
                algorithm_version='hte_v1', search='greedy'
            )
            target = "<hte>[MASK]|CC(C)C(=O)Nc1ccc(Cl)cc1>>CC(C)C(=O)Nc1ccc(O)cc1"
            hte_generator = HTERegressionTransformer(
                configuration=config, target=target
            )
            prediction = list(hte_generator.sample(1))[0]
    """

    algorithm_type: ClassVar[str] = "conditional_generation"
    domain: ClassVar[str] = "materials"
    algorithm_version: str = field(
        default="hte_v1",
        metadata=dict(
            description="The version of the HTE algorithm to use.",
            options=["hte_v1", "hte_multi_property"],
        ),
    )

    search: str = field(
        default="sample",
        metadata=dict(
            description="Search algorithm to use for the generation: sample or greedy",
            options=["sample", "greedy"],
        ),
    )

    temperature: float = field(
        default=0.8,
        metadata=dict(
            description="Temperature parameter for the softmax sampling in decoding."
        ),
    )
    
    batch_size: int = field(
        default=8,
        metadata=dict(description="Batch size for the conditional generation"),
    )
    
    tolerance: Union[float, Dict[str, float]] = field(
        default=10.0,
        metadata=dict(
            description="""Precision tolerance for the conditional generation task. This is the
            tolerated deviation between desired/primed property and predicted property of the
            generated molecule. Given in percentage with respect to the property range encountered
            during training. Either a single float or a dict of floats with properties as keys.
            NOTE: The tolerance is *only* used for post-hoc filtering of the generated molecules.
            """
        ),
    )
    
    use_descriptors: bool = field(
        default=True,
        metadata=dict(
            description="Whether to use molecular descriptors as additional conditioning."
        ),
    )
    
    n_descriptors: int = field(
        default=16,
        metadata=dict(
            description="Number of PCA molecular descriptors to use for conditioning."
        ),
    )
    
    property_ranges: Dict[str, tuple] = field(
        default_factory=lambda: {"hte_rate": (-3.0, 3.0)},
        metadata=dict(
            description="Valid ranges for each property (z-scored values)."
        ),
    )

    sampling_wrapper: Dict = field(
        default_factory=dict,
        metadata=dict(
            description="""High-level entry point for reaction-level access. Provide a
            dictionary that is used to build a custom sampling wrapper.
            Example: {
                'fraction_to_mask': 0.3,
                'property_goal': {'hte_rate': 0.5},
                'descriptor_values': {'d0': 1.2, 'd1': -0.3}
            }
            - 'fraction_to_mask' specifies the ratio of tokens that can be changed.
            - 'property_goal' specifies the target HTE rate and other properties.
            - 'descriptor_values' specifies molecular descriptor conditioning.
            """
        ),
    )

    def get_target_description(self) -> Dict[str, str]:
        """Get description of the target for generation.

        Returns:
            target description.
        """
        return {
            "title": "HTE Rate and Descriptor Conditioning",
            "description": (
                "Target can be: 1) Dict with 'hte_rate' and descriptor keys (d0, d1, ...), "
                "2) String with property tokens '<hte>value|reaction_smiles', "
                "3) Dict with 'property_goal' and 'descriptor_values' keys."
            ),
            "type": "string or dict",
        }

    def get_conditional_generator(
        self, resources_path: str, context: Optional[str] = None
    ) -> HTEConditionalGenerator:
        """Instantiate the actual generator implementation.

        Args:
            resources_path: local path to model files.
            context: input sequence/target to be used for the generation.

        Returns:
            instance with generate_batch method for targeted generation.
        """
        self.generator = HTEConditionalGenerator(
            resources_path=resources_path,
            context=context,
            search=self.search,
            temperature=self.temperature,
            batch_size=self.batch_size,
            tolerance=self.tolerance,
            use_descriptors=self.use_descriptors,
            n_descriptors=self.n_descriptors,
            property_ranges=self.property_ranges,
            sampling_wrapper=self.sampling_wrapper,
        )
        return self.generator

    def validate_item(self, item: str) -> str:
        """Check that item is a valid reaction SMILES.

        Args:
            item: a generated item that is possibly not valid.

        Raises:
            InvalidItem: in case the item can not be validated.

        Returns:
            the validated item.
        """
        if item is None:
            raise ValueError("Generated item is None")
        
        # Delegate validation to the generator
        items, _ = self.generator.validate_output([item])
        if items[0] is None:
            if self.generator.task == "generation":
                detail = f'Invalid reaction SMILES: "{item}"'
            else:
                detail = f'Invalid property prediction: "{item}"'
            raise ValueError(detail)
        return item

    @classmethod
    def get_filepath_mappings_for_training_pipeline_arguments(
        cls, training_pipeline_arguments
    ) -> Dict[str, str]:
        """Get filepath mappings for the given training pipeline arguments.
        
        Args:
            training_pipeline_arguments: training pipeline arguments.
            
        Returns:
            a mapping between artifacts' files and training pipeline's output files.
        """
        # This would be implemented for full GT4SD integration
        # For now, return basic mapping
        return {
            "pytorch_model.bin": "model/pytorch_model.bin",
            "config.json": "model/config.json",
            "vocab.txt": "tokenizer/vocab.txt",
            "tokenizer_config.json": "tokenizer/tokenizer_config.json",
            "special_tokens_map.json": "tokenizer/special_tokens_map.json",
        }


# For multi-property support (future extension)
@ApplicationsRegistry.register_algorithm_application(HTERegressionTransformer)
@dataclass
class HTERegressionTransformerMultiProperty(HTERegressionTransformerMolecules):
    """
    Configuration for multi-property HTE Regression Transformer.
    
    Supports simultaneous prediction/generation conditioned on multiple properties:
    HTE rate, yield, selectivity, conversion, etc.
    """
    
    algorithm_version: str = field(
        default="hte_multi_property",
        metadata=dict(
            description="Multi-property version of HTE RT",
        ),
    )
    
    property_ranges: Dict[str, tuple] = field(
        default_factory=lambda: {
            "hte_rate": (-3.0, 3.0),
            "yield": (0.0, 100.0),
            "selectivity": (0.0, 100.0),
            "conversion": (0.0, 100.0),
        },
        metadata=dict(
            description="Valid ranges for each property."
        ),
    )
    
    property_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "hte_rate": 1.0,
            "yield": 0.8,
            "selectivity": 0.6,
            "conversion": 0.4,
        },
        metadata=dict(
            description="Relative importance weights for each property during generation."
        ),
    )
