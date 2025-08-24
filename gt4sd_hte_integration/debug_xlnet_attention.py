#!/usr/bin/env python
"""
Diagnose and fix XLNet attention mechanism issues.

The error "tensor a (8) must match tensor b (9)" suggests sequence length mismatch
in attention masks during generation.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.append('/home/passos/ml_measurable_hte_rates/regression-transformer')

from terminator.tokenization import ExpressionBertTokenizer
from transformers import AutoConfig, AutoModelWithLMHead


class XLNetAttentionFixer:
    """Diagnose and fix XLNet attention mechanism issues."""
    
    def __init__(self):
        self.model_path = "/home/passos/ml_measurable_hte_rates/regression-transformer/runs/best_model_final/model"
        self.tokenizer_path = "/home/passos/ml_measurable_hte_rates/regression-transformer/runs/hte"
        
        # Load components
        print("🔧 Loading model components...")
        self.tokenizer = ExpressionBertTokenizer.from_pretrained(self.tokenizer_path)
        self.config = AutoConfig.from_pretrained(self.model_path)
        
        # Use CPU to avoid CUDA issues during debugging
        self.device = torch.device("cpu")
        self.model = None  # Load only when needed
        
    def analyze_tokenization_issue(self, test_input: str):
        """Analyze tokenization and sequence length issues."""
        print(f"\n🔍 Analyzing tokenization for: {test_input}")
        
        # Basic tokenization
        inputs = self.tokenizer(test_input, return_tensors="pt")
        
        print(f"Input text: {test_input}")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
        print(f"Input IDs: {inputs['input_ids'][0].tolist()}")
        
        # Check attention mask
        if 'attention_mask' in inputs:
            print(f"Attention mask shape: {inputs['attention_mask'].shape}")
            print(f"Attention mask: {inputs['attention_mask'][0].tolist()}")
        else:
            print("No attention mask generated")
        
        # Check for padding token issues
        pad_token_id = self.tokenizer.pad_token_id
        print(f"Pad token ID: {pad_token_id}")
        
        # Decode tokens to verify
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        print(f"Tokens: {tokens}")
        
        return inputs
    
    def create_proper_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create proper attention mask that matches XLNet requirements."""
        batch_size, seq_len = input_ids.shape
        
        # Create basic attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        # Handle padding tokens
        if self.tokenizer.pad_token_id is not None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return attention_mask
    
    def safe_model_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """Perform safe model forward pass with proper error handling."""
        
        if self.model is None:
            print("Loading model for forward pass...")
            self.model = AutoModelWithLMHead.from_pretrained(self.model_path, config=self.config)
            self.model.to(self.device)
            self.model.eval()
        
        print(f"Forward pass - Input shape: {input_ids.shape}")
        if attention_mask is not None:
            print(f"Forward pass - Attention mask shape: {attention_mask.shape}")
        
        try:
            with torch.no_grad():
                if attention_mask is not None:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=input_ids)
                
                print("✅ Forward pass successful!")
                print(f"Output shape: {outputs[0].shape}")
                return outputs
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return None
    
    def fixed_generate_step(self, input_ids: torch.Tensor, max_new_tokens: int = 5):
        """Fixed generation step that avoids attention dimension issues."""
        
        if self.model is None:
            print("Loading model for generation...")
            self.model = AutoModelWithLMHead.from_pretrained(self.model_path, config=self.config)
            self.model.to(self.device)
            self.model.eval()
        
        print(f"\n🎯 Starting fixed generation...")
        print(f"Input shape: {input_ids.shape}")
        
        generated_ids = input_ids.clone()
        
        for step in range(max_new_tokens):
            print(f"\n--- Generation step {step + 1} ---")
            print(f"Current sequence length: {generated_ids.shape[1]}")
            
            try:
                # Create attention mask for current sequence
                attention_mask = self.create_proper_attention_mask(generated_ids)
                print(f"Attention mask shape: {attention_mask.shape}")
                
                # Forward pass
                with torch.no_grad():
                    outputs = self.model(input_ids=generated_ids, attention_mask=attention_mask)
                    logits = outputs[0]  # [batch_size, seq_len, vocab_size]
                
                # Get logits for next token
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Simple greedy selection
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [batch_size, 1]
                
                print(f"Next token ID: {next_token_id[0].item()}")
                next_token_str = self.tokenizer.convert_ids_to_tokens([next_token_id[0].item()])[0]
                print(f"Next token: {next_token_str}")
                
                # Append to sequence
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                
                # Check for end conditions
                if next_token_id[0].item() in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                    print("End token reached")
                    break
                    
            except Exception as e:
                print(f"❌ Generation step {step + 1} failed: {e}")
                print(f"Current sequence shape: {generated_ids.shape}")
                break
        
        return generated_ids
    
    def create_simple_generator(self):
        """Create a simple, robust generator that avoids XLNet attention issues."""
        
        def simple_generate(input_text: str, max_new_tokens: int = 10) -> str:
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            
            # Use fixed generation
            generated_ids = self.fixed_generate_step(input_ids, max_new_tokens)
            
            # Decode
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            input_text_clean = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
            # Extract generated part
            generated_part = generated_text[len(input_text_clean):].strip()
            
            return generated_part
        
        return simple_generate


def test_xlnet_fixes():
    """Test the XLNet attention fixes."""
    print("🚀 Testing XLNet Attention Fixes")
    print("=" * 50)
    
    fixer = XLNetAttentionFixer()
    
    # Test inputs that previously caused issues
    test_inputs = [
        "<d0>0.5 <d1>-0.3 <hte> |",
        "<hte> |",
        "<d0>1.2 <hte>-1.5 |"
    ]
    
    print("\n" + "="*30 + " TOKENIZATION ANALYSIS " + "="*30)
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n--- Test Input {i} ---")
        inputs = fixer.analyze_tokenization_issue(test_input)
        
        # Test forward pass
        attention_mask = fixer.create_proper_attention_mask(inputs["input_ids"])
        outputs = fixer.safe_model_forward(inputs["input_ids"], attention_mask)
    
    print("\n" + "="*30 + " GENERATION TESTING " + "="*30)
    
    # Create simple generator
    simple_generator = fixer.create_simple_generator()
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n--- Generation Test {i} ---")
        print(f"Input: {test_input}")
        
        try:
            generated = simple_generator(test_input, max_new_tokens=5)
            print(f"Generated: {generated}")
            print("✅ Generation successful!")
        except Exception as e:
            print(f"❌ Generation failed: {e}")


if __name__ == "__main__":
    test_xlnet_fixes()
