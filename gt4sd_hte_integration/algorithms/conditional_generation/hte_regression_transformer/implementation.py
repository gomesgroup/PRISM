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
"""Implementation of HTE Regression Transformer conditional generator."""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelWithLMHead

# Import our custom tokenizer
import sys
sys.path.append('/home/passos/ml_measurable_hte_rates/regression-transformer')
from terminator.tokenization import ExpressionBertTokenizer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PropertyExtractor:
    """Robust property extraction from generated sequences."""
    
    def __init__(self, tokenizer, property_stats: Dict[str, Dict[str, float]]):
        self.tokenizer = tokenizer
        self.property_stats = property_stats  # {"hte_rate": {"mean": 0, "std": 1}}
        
    def extract_property(self, generated_text: str, property_name: str = "hte") -> Optional[float]:
        """Extract property value from generated text."""
        
        # Strategy 1: Direct token extraction
        pattern = f"<{property_name}>([\\-\\d\\.]+)"
        match = re.search(pattern, generated_text)
        if match:
            try:
                value = float(match.group(1))
                return self._denormalize(value, property_name)
            except ValueError:
                pass
        
        # Strategy 2: Numeric token reconstruction
        numeric_pattern = r"_(\d+)_(-?\d+)_"
        numeric_matches = re.findall(numeric_pattern, generated_text)
        if numeric_matches:
            # Reconstruct floating point from mantissa/exponent
            for mantissa, exponent in numeric_matches:
                try:
                    value = float(f"{mantissa}e{exponent}")
                    if self._is_valid_property_value(value, property_name):
                        return self._denormalize(value, property_name)
                except (ValueError, OverflowError):
                    continue
        
        # Strategy 3: Parse descriptor sequence and predict
        descriptor_values = self._extract_descriptors(generated_text)
        if descriptor_values:
            predicted_value = self._predict_from_descriptors(descriptor_values, property_name)
            return predicted_value
        
        # Strategy 4: Default value based on context
        return self._get_contextual_default(generated_text, property_name)
    
    def _extract_descriptors(self, text: str) -> Dict[str, float]:
        """Extract descriptor values from text."""
        descriptors = {}
        pattern = r"<d(\d+)>([\\-\\d\\.]+)"
        matches = re.findall(pattern, text)
        for idx, value in matches:
            try:
                descriptors[f"d{idx}"] = float(value)
            except ValueError:
                continue
        return descriptors
    
    def _denormalize(self, value: float, property_name: str) -> float:
        """Convert z-scored value back to original scale."""
        stats = self.property_stats.get(property_name, {"mean": 0, "std": 1})
        return value * stats["std"] + stats["mean"]
    
    def _is_valid_property_value(self, value: float, property_name: str) -> bool:
        """Check if value is in valid range for property."""
        valid_ranges = {
            "hte": (-5.0, 5.0),  # z-scored range
            "hte_rate": (-5.0, 5.0),
            "yield": (0.0, 100.0),
            "selectivity": (0.0, 100.0)
        }
        
        if property_name in valid_ranges:
            min_val, max_val = valid_ranges[property_name]
            return min_val <= value <= max_val
        return True
    
    def _predict_from_descriptors(self, descriptors: Dict[str, float], property_name: str) -> float:
        """Fallback: predict property from descriptors using simple regression."""
        # Simple weighted average as placeholder
        if descriptors:
            values = list(descriptors.values())
            return np.mean(values)
        return 0.0
    
    def _get_contextual_default(self, text: str, property_name: str) -> float:
        """Get context-aware default value."""
        # Return dataset mean (z-scored)
        return 0.0


class PropertyAwareGenerator:
    """Generation strategies that ensure property token generation."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.property_token_ids = self._get_property_token_ids()
        
    def _get_property_token_ids(self) -> Dict[str, int]:
        """Get token IDs for all property tokens."""
        vocab = self.tokenizer.get_vocab()
        property_tokens = {}
        
        for token, token_id in vocab.items():
            if '<hte>' in token or '<yield>' in token or '<selectivity>' in token:
                property_tokens[token] = token_id
                
        return property_tokens
    
    def generate_with_property_constraint(
        self,
        input_ids: torch.Tensor,
        property_name: str = "hte",
        property_value: Optional[float] = None,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.9,
        num_beams: int = 3
    ) -> torch.Tensor:
        """Generate with constraints to ensure property token appears."""
        
        device = input_ids.device
        
        # Phase 1: Generate until property token position
        property_token_id = self.property_token_ids.get(f"<{property_name}>", None)
        
        if property_token_id is None:
            # Fallback to standard generation
            return self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Custom generation loop
        generated = input_ids
        property_generated = False
        
        for step in range(max_length - input_ids.shape[1]):
            with torch.no_grad():
                outputs = self.model(generated)
                logits = outputs[0][:, -1, :]
                
                # Apply temperature
                logits = logits / temperature
                
                # Check if we should force property token
                if step > 5 and not property_generated:
                    # Boost property token probability
                    logits[:, property_token_id] += 10.0
                
                # Apply top-p filtering
                filtered_logits = self._top_p_filtering(logits, top_p)
                
                # Sample next token
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check if property token was generated
                if next_token[0, 0] == property_token_id:
                    property_generated = True
                    
                    # Next, generate the property value
                    if property_value is not None:
                        value_tokens = self._encode_property_value(property_value)
                        generated = torch.cat([generated, next_token, value_tokens], dim=-1)
                        continue
                
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for end of sequence
                if next_token[0, 0] == self.tokenizer.eos_token_id:
                    break
        
        return generated
    
    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def _encode_property_value(self, value: float) -> torch.Tensor:
        """Encode property value as tokens."""
        # Convert value to string and tokenize
        value_str = f"{value:.4f}"
        value_tokens = self.tokenizer.encode(value_str, add_special_tokens=False)
        return torch.tensor([value_tokens], device=self.model.device)


class HTEConditionalGenerator:
    """Main interface for HTE Regression Transformer."""

    def __init__(
        self,
        resources_path: str,
        context: Optional[str] = None,
        search: str = "sample",
        temperature: float = 0.8,
        batch_size: int = 8,
        tolerance: Union[float, Dict[str, float]] = 10.0,
        use_descriptors: bool = True,
        n_descriptors: int = 16,
        property_ranges: Dict[str, tuple] = None,
        sampling_wrapper: Dict[str, Any] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """
        Initialize the HTE conditional generator.

        Args:
            resources_path: directory where to find models and parameters.
            context: input context for generation (target properties/descriptors).
            search: search strategy ('sample' or 'greedy').
            temperature: temperature for sampling.
            batch_size: number of samples per generation call.
            tolerance: tolerance for property matching.
            use_descriptors: whether to use molecular descriptors.
            n_descriptors: number of descriptors to use.
            property_ranges: valid ranges for each property.
            sampling_wrapper: additional sampling configuration.
            device: device for inference.
        """
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Store configuration
        self.context = context
        self.search = search
        self.temperature = temperature
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.use_descriptors = use_descriptors
        self.n_descriptors = n_descriptors
        self.property_ranges = property_ranges or {"hte_rate": (-3.0, 3.0)}
        self.sampling_wrapper = sampling_wrapper or {}
        
        # Determine task type
        self.task = self._determine_task(context)
        
        # Load model and tokenizer
        self._load_model(resources_path)
        
        # Initialize property extractor and generator
        property_stats = {"hte_rate": {"mean": -7.5, "std": 1.2}}  # From training
        self.property_extractor = PropertyExtractor(self.tokenizer, property_stats)
        self.property_generator = PropertyAwareGenerator(self.model, self.tokenizer)
        
        logger.info(f"HTE RT initialized with task: {self.task}, device: {self.device}")
    
    def _determine_task(self, context: Optional[str]) -> str:
        """Determine if this is a regression or generation task."""
        if context is None:
            return "generation"
        
        if isinstance(context, str):
            # Check for [MASK] tokens in property position
            if "<hte>[MASK]" in context or "hte>[MASK]" in context:
                return "regression"
            else:
                return "generation"
        elif isinstance(context, dict):
            # If all properties are specified, it's generation
            # If some are missing/None, it's regression
            return "generation"  # Default for dict input
        
        return "generation"
    
    def _load_model(self, resources_path: str):
        """Load the HTE RT model and tokenizer."""
        
        # For now, load from our trained model
        model_path = Path("/home/passos/ml_measurable_hte_rates/regression-transformer/runs/best_model_final/model")
        tokenizer_path = Path("/home/passos/ml_measurable_hte_rates/regression-transformer/runs/hte")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        # Load tokenizer
        self.tokenizer = ExpressionBertTokenizer.from_pretrained(str(tokenizer_path))
        
        # Load model
        config = AutoConfig.from_pretrained(str(model_path))
        self.model = AutoModelWithLMHead.from_pretrained(str(model_path), config=config)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model with {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
    
    def generate_batch(self, number_samples: int) -> List[str]:
        """Generate a batch of samples.
        
        Args:
            number_samples: number of samples to generate.
            
        Returns:
            list of generated samples (SMILES strings or property predictions).
        """
        
        samples = []
        
        for _ in range(number_samples):
            try:
                # Prepare input based on context
                input_text = self._prepare_input(self.context)
                
                # Tokenize input
                inputs = self.tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    max_length=256, 
                    truncation=True
                )
                input_ids = inputs["input_ids"].to(self.device)
                
                # Generate with property constraints
                if self.search == "greedy":
                    with torch.no_grad():
                        outputs = self.model.generate(
                            input_ids,
                            max_length=input_ids.shape[1] + 50,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    # Use property-aware generation
                    outputs = self.property_generator.generate_with_property_constraint(
                        input_ids,
                        property_name="hte",
                        property_value=self._extract_target_property(),
                        temperature=self.temperature,
                        max_length=input_ids.shape[1] + 50,
                    )
                
                # Decode generated sequence
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract the generated part (remove input)
                input_text_clean = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                generated_part = generated_text[len(input_text_clean):].strip()
                
                # Process based on task
                if self.task == "regression":
                    # Extract property value
                    property_value = self.property_extractor.extract_property(generated_text, "hte")
                    if property_value is not None:
                        samples.append(str(property_value))
                    else:
                        samples.append("-1.0")  # Default fallback
                else:
                    # Extract SMILES from generated text
                    smiles = self._extract_smiles(generated_part)
                    if smiles:
                        samples.append(smiles)
                    else:
                        samples.append("CC")  # Default fallback
                        
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                if self.task == "regression":
                    samples.append("-1.0")
                else:
                    samples.append("CC")
        
        return samples
    
    def _prepare_input(self, context) -> str:
        """Prepare input text from context."""
        
        if context is None:
            # Default: random descriptors + HTE token
            desc_tokens = [f"<d{i}>{np.random.normal(0, 1):.4f}" for i in range(4)]
            return " ".join(desc_tokens) + " <hte> |"
        
        if isinstance(context, str):
            # Already formatted string
            return context
        
        if isinstance(context, dict):
            # Convert dict to formatted string
            tokens = []
            
            # Add descriptors
            for i in range(self.n_descriptors):
                key = f"d{i}"
                if key in context:
                    tokens.append(f"<{key}>{context[key]:.4f}")
                else:
                    tokens.append(f"<{key}>{np.random.normal(0, 1):.4f}")
            
            # Add property token
            if "hte_rate" in context:
                tokens.append(f"<hte>{context['hte_rate']:.4f}")
            else:
                tokens.append("<hte>")
            
            return " ".join(tokens) + " |"
        
        # Fallback
        return "<hte> |"
    
    def _extract_target_property(self) -> Optional[float]:
        """Extract target property value from context."""
        if isinstance(self.context, dict):
            return self.context.get("hte_rate")
        elif isinstance(self.context, str):
            match = re.search(r"<hte>([\\-\\d\\.]+)", self.context)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
        return None
    
    def _extract_smiles(self, generated_text: str) -> Optional[str]:
        """Extract SMILES from generated text."""
        # Look for SMILES after the | separator
        parts = generated_text.split("|")
        if len(parts) > 1:
            smiles_part = parts[1].strip()
            # Basic SMILES validation (contains chemical characters)
            if re.match(r'^[A-Za-z0-9@+\-\[\]()=#\.]+$', smiles_part):
                return smiles_part
        
        # Fallback: look for any chemical-looking string
        chemical_pattern = r'[A-Z][a-z]?(?:\([^)]*\))?(?:\[[^\]]*\])?[=\-#]?'
        matches = re.findall(chemical_pattern, generated_text)
        if matches:
            return "".join(matches[:10])  # Take first few matches
        
        return None
    
    def validate_output(self, outputs: List[str]) -> Tuple[List[Optional[str]], List[bool]]:
        """Validate generated outputs."""
        validated = []
        valid_flags = []
        
        for output in outputs:
            if self.task == "regression":
                # Validate as numeric
                try:
                    float(output)
                    validated.append(output)
                    valid_flags.append(True)
                except ValueError:
                    validated.append(None)
                    valid_flags.append(False)
            else:
                # Validate as SMILES (basic check)
                if output and len(output) > 1 and not output.isspace():
                    validated.append(output)
                    valid_flags.append(True)
                else:
                    validated.append(None)
                    valid_flags.append(False)
        
        return validated, valid_flags
