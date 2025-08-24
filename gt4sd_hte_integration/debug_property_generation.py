#!/usr/bin/env python
"""
Advanced debugging script for HTE Regression Transformer property token generation.

This script analyzes why the model generates numeric tokens (_1_-3_) instead of 
proper property tokens (<hte>0.5) and implements multiple debugging strategies.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

# Add paths
sys.path.append('/home/passos/ml_measurable_hte_rates/regression-transformer')
from terminator.tokenization import ExpressionBertTokenizer
from transformers import AutoConfig, AutoModelWithLMHead

class PropertyGenerationDebugger:
    """Advanced debugger for property token generation issues."""
    
    def __init__(self):
        self.model_path = Path("/home/passos/ml_measurable_hte_rates/regression-transformer/runs/best_model_final/model")
        self.tokenizer_path = Path("/home/passos/ml_measurable_hte_rates/regression-transformer/runs/hte")
        
        # Load model and tokenizer
        print("🔧 Loading model and tokenizer...")
        self.tokenizer = ExpressionBertTokenizer.from_pretrained(str(self.tokenizer_path))
        config = AutoConfig.from_pretrained(str(self.model_path))
        self.model = AutoModelWithLMHead.from_pretrained(str(self.model_path), config=config)
        
        # Move to CPU for easier debugging
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")
        print(f"✅ Tokenizer vocabulary size: {len(self.tokenizer.get_vocab())}")
    
    def analyze_vocabulary(self):
        """Analyze tokenizer vocabulary for property-related tokens."""
        print("\n" + "="*60)
        print("📚 VOCABULARY ANALYSIS")
        print("="*60)
        
        vocab = self.tokenizer.get_vocab()
        
        # Find property tokens
        property_tokens = {}
        descriptor_tokens = {}
        numeric_tokens = {}
        special_tokens = {}
        
        for token, token_id in vocab.items():
            if '<hte>' in token.lower():
                property_tokens[token] = token_id
            elif token.startswith('<d') and '>' in token:
                descriptor_tokens[token] = token_id
            elif '_' in token and any(c.isdigit() for c in token):
                numeric_tokens[token] = token_id
            elif token.startswith('<') and token.endswith('>'):
                special_tokens[token] = token_id
        
        print(f"🎯 Property tokens found: {len(property_tokens)}")
        for token, token_id in sorted(property_tokens.items()):
            print(f"   {token} -> ID {token_id}")
        
        print(f"\n📊 Descriptor tokens found: {len(descriptor_tokens)}")
        if len(descriptor_tokens) <= 20:
            for token, token_id in sorted(descriptor_tokens.items()):
                print(f"   {token} -> ID {token_id}")
        else:
            print(f"   (showing first 10 of {len(descriptor_tokens)})")
            for token, token_id in sorted(list(descriptor_tokens.items())[:10]):
                print(f"   {token} -> ID {token_id}")
        
        print(f"\n🔢 Numeric tokens found: {len(numeric_tokens)}")
        if len(numeric_tokens) <= 20:
            for token, token_id in sorted(numeric_tokens.items()):
                print(f"   {token} -> ID {token_id}")
        else:
            print(f"   (showing first 10 of {len(numeric_tokens)})")
            for token, token_id in sorted(list(numeric_tokens.items())[:10]):
                print(f"   {token} -> ID {token_id}")
        
        print(f"\n⚡ Other special tokens: {len(special_tokens)}")
        for token, token_id in sorted(special_tokens.items()):
            print(f"   {token} -> ID {token_id}")
        
        return property_tokens, descriptor_tokens, numeric_tokens, special_tokens
    
    def trace_generation_step_by_step(self, input_text: str, max_steps: int = 15):
        """Trace token generation step by step with detailed analysis."""
        print(f"\n" + "="*60)
        print("🔍 STEP-BY-STEP GENERATION TRACE")
        print("="*60)
        print(f"Input: {input_text}")
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        print(f"Input tokens: {self.tokenizer.convert_ids_to_tokens(input_ids[0])}")
        print(f"Input IDs: {input_ids[0].tolist()}")
        
        # Generation loop with detailed logging
        generated_ids = input_ids.clone()
        
        for step in range(max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs[0]  # [batch_size, seq_len, vocab_size]
                
                # Get logits for next token
                next_token_logits = logits[0, -1, :]  # [vocab_size]
                
                # Get top-k tokens and their probabilities
                top_k = 10
                top_values, top_indices = torch.topk(next_token_logits, top_k)
                top_probs = F.softmax(top_values, dim=-1)
                
                print("Top 10 most likely next tokens:")
                for i in range(top_k):
                    token_id = top_indices[i].item()
                    token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                    prob = top_probs[i].item()
                    logit = top_values[i].item()
                    print(f"  {i+1:2d}. {token:15s} (ID: {token_id:4d}) - Prob: {prob:.4f}, Logit: {logit:.2f}")
                
                # Check if <hte> token is in top candidates
                vocab = self.tokenizer.get_vocab()
                hte_token_id = vocab.get('<hte>', None)
                if hte_token_id is not None:
                    hte_logit = next_token_logits[hte_token_id].item()
                    hte_prob = F.softmax(next_token_logits, dim=-1)[hte_token_id].item()
                    hte_rank = (next_token_logits > next_token_logits[hte_token_id]).sum().item() + 1
                    print(f"  <hte> token: ID {hte_token_id}, Rank {hte_rank}, Prob: {hte_prob:.6f}, Logit: {hte_logit:.2f}")
                
                # Select next token (greedy)
                next_token_id = torch.argmax(next_token_logits).item()
                next_token = self.tokenizer.convert_ids_to_tokens([next_token_id])[0]
                
                print(f"Selected: {next_token} (ID: {next_token_id})")
                
                # Add to sequence
                generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]])], dim=1)
                
                # Check if we hit special tokens
                if next_token in ['<pad>', '<eos>', '[SEP]']:
                    print(f"Stopping at special token: {next_token}")
                    break
                
                # Check if we generated property token
                if '<hte>' in next_token:
                    print(f"🎯 Generated property token: {next_token}")
                    break
        
        # Final generated sequence
        final_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids[0])
        final_text = self.tokenizer.decode(generated_ids[0])
        
        print(f"\nFinal tokens: {final_tokens}")
        print(f"Final text: {final_text}")
        
        return generated_ids
    
    def analyze_training_data_patterns(self):
        """Analyze patterns in training data to understand expected format."""
        print(f"\n" + "="*60)
        print("📋 TRAINING DATA PATTERN ANALYSIS")
        print("="*60)
        
        train_file = Path("/home/passos/ml_measurable_hte_rates/regression-transformer/runs/hte/train.txt")
        
        if not train_file.exists():
            print("❌ Training file not found")
            return
        
        with open(train_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"📊 Analyzing {len(lines)} training examples...")
        
        # Pattern analysis
        patterns = defaultdict(int)
        hte_values = []
        
        for i, line in enumerate(lines[:100]):  # Analyze first 100 lines
            if i < 5:
                print(f"Example {i+1}: {line[:80]}...")
            
            # Extract HTE values
            hte_match = re.search(r'<hte>([\\-\\d\\.]+)', line)
            if hte_match:
                try:
                    value = float(hte_match.group(1))
                    hte_values.append(value)
                except ValueError:
                    pass
            
            # Pattern detection
            if '<hte>' in line:
                patterns['has_hte_token'] += 1
            if '|' in line:
                patterns['has_separator'] += 1
            if line.count('<d') > 0:
                patterns['has_descriptors'] += 1
        
        print(f"\nPattern statistics (first 100 examples):")
        for pattern, count in patterns.items():
            print(f"  {pattern}: {count}/100 ({count}%)")
        
        if hte_values:
            print(f"\nHTE value statistics:")
            print(f"  Count: {len(hte_values)}")
            print(f"  Range: {min(hte_values):.3f} to {max(hte_values):.3f}")
            print(f"  Mean: {np.mean(hte_values):.3f} ± {np.std(hte_values):.3f}")
        
        return patterns
    
    def test_property_token_forcing(self, input_text: str):
        """Test forcing property token generation through logit manipulation."""
        print(f"\n" + "="*60)
        print("🎯 PROPERTY TOKEN FORCING TEST")
        print("="*60)
        
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        vocab = self.tokenizer.get_vocab()
        hte_token_id = vocab.get('<hte>', None)
        
        if hte_token_id is None:
            print("❌ <hte> token not found in vocabulary")
            return
        
        print(f"🎯 Forcing <hte> token (ID: {hte_token_id})")
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs[0][0, -1, :].clone()  # [vocab_size]
            
            # Original top tokens
            print("Original top 5 tokens:")
            top_vals, top_ids = torch.topk(logits, 5)
            for i in range(5):
                token = self.tokenizer.convert_ids_to_tokens([top_ids[i].item()])[0]
                print(f"  {token} (logit: {top_vals[i].item():.2f})")
            
            # Force <hte> token
            original_logit = logits[hte_token_id].item()
            boost_values = [5.0, 10.0, 20.0]
            
            for boost in boost_values:
                modified_logits = logits.clone()
                modified_logits[hte_token_id] += boost
                
                print(f"\nAfter boosting <hte> by {boost}:")
                print(f"  <hte> logit: {original_logit:.2f} -> {modified_logits[hte_token_id].item():.2f}")
                
                top_vals, top_ids = torch.topk(modified_logits, 5)
                for i in range(5):
                    token = self.tokenizer.convert_ids_to_tokens([top_ids[i].item()])[0]
                    print(f"  {token} (logit: {top_vals[i].item():.2f})")
                
                if top_ids[0].item() == hte_token_id:
                    print(f"  ✅ <hte> is now top token with boost {boost}")
                    break
    
    def test_numeric_token_decoding(self):
        """Test decoding of numeric tokens that the model actually generates."""
        print(f"\n" + "="*60)
        print("🔢 NUMERIC TOKEN DECODING TEST")
        print("="*60)
        
        # Common numeric patterns observed in generation
        test_patterns = [
            "_1_-3_ _5_-4_",
            "_0_0_ _2_-1_",
            "_8_-3_ _1_0_",
            "_7_-3_ _0_0_"
        ]
        
        for pattern in test_patterns:
            print(f"\nPattern: {pattern}")
            
            # Try to reconstruct floating point value
            numeric_matches = re.findall(r"_(\d+)_(-?\d+)_", pattern)
            
            if numeric_matches:
                print("  Potential interpretations:")
                for mantissa, exponent in numeric_matches:
                    try:
                        # Interpretation 1: mantissa * 10^exponent
                        value1 = float(mantissa) * (10 ** int(exponent))
                        print(f"    {mantissa} * 10^{exponent} = {value1}")
                        
                        # Interpretation 2: 0.mantissa * 10^exponent
                        value2 = (float(mantissa) / 10) * (10 ** int(exponent))
                        print(f"    0.{mantissa} * 10^{exponent} = {value2}")
                        
                        # Interpretation 3: scientific notation
                        sci_notation = f"{mantissa}e{exponent}"
                        value3 = float(sci_notation)
                        print(f"    {sci_notation} = {value3}")
                        
                    except (ValueError, OverflowError) as e:
                        print(f"    Error: {e}")
    
    def comprehensive_diagnosis(self):
        """Run comprehensive diagnosis of property generation issues."""
        print("🏥 COMPREHENSIVE PROPERTY GENERATION DIAGNOSIS")
        print("=" * 80)
        
        # Test inputs
        test_inputs = [
            "<d0>0.5 <d1>-0.3 <hte> |",
            "<d0>1.2 <d1>0.8 <hte>",
            "<hte> | CC>>CCO",
            "descriptors <hte>1.5 | reaction",
        ]
        
        for i, input_text in enumerate(test_inputs, 1):
            print(f"\n{'='*20} TEST CASE {i} {'='*20}")
            print(f"Input: {input_text}")
            
            try:
                # Quick generation test
                inputs = self.tokenizer(input_text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_length=input_ids.shape[1] + 10,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                
                generated_text = self.tokenizer.decode(outputs[0])
                input_text_decoded = self.tokenizer.decode(input_ids[0])
                generated_part = generated_text[len(input_text_decoded):].strip()
                
                print(f"Generated: {generated_part}")
                
                # Check for property tokens
                if '<hte>' in generated_part:
                    print("✅ Contains <hte> token")
                else:
                    print("❌ No <hte> token generated")
                
                # Check for numeric patterns
                numeric_patterns = re.findall(r"_\d+_-?\d+_", generated_part)
                if numeric_patterns:
                    print(f"🔢 Numeric patterns found: {numeric_patterns}")
                
            except Exception as e:
                print(f"❌ Generation failed: {e}")


def main():
    """Run comprehensive debugging analysis."""
    print("🚀 Starting HTE Regression Transformer Property Generation Debug")
    print("=" * 80)
    
    try:
        debugger = PropertyGenerationDebugger()
        
        # Run all diagnostic tests
        debugger.analyze_vocabulary()
        debugger.analyze_training_data_patterns()
        
        # Test specific generation scenarios
        test_input = "<d0>0.5 <d1>-0.3 <hte> |"
        debugger.trace_generation_step_by_step(test_input, max_steps=10)
        debugger.test_property_token_forcing(test_input)
        debugger.test_numeric_token_decoding()
        debugger.comprehensive_diagnosis()
        
        print(f"\n" + "="*80)
        print("🎯 DIAGNOSIS SUMMARY")
        print("="*80)
        print("Key findings will help identify the root cause of property generation issues.")
        print("This analysis provides the foundation for implementing fixes in Phase 2.")
        
    except Exception as e:
        print(f"❌ Debug session failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
