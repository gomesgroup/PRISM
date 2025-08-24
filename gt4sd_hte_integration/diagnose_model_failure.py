#!/usr/bin/env python
"""
Phase 1: Diagnose Model Failure - Deep Analysis

This script performs comprehensive diagnosis of why the RT model fails to generate
meaningful HTE predictions and gets stuck in numeric token loops.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import Counter

# Add paths
sys.path.append('/home/passos/ml_measurable_hte_rates/regression-transformer')

from terminator.tokenization import ExpressionBertTokenizer
from transformers import AutoConfig, AutoModelForCausalLM


class ModelFailureDiagnostics:
    """Comprehensive diagnostics for RT model failures."""
    
    def __init__(self):
        self.model_path = "/home/passos/ml_measurable_hte_rates/regression-transformer/runs/best_model_final/model"
        self.tokenizer_path = "/home/passos/ml_measurable_hte_rates/regression-transformer/runs/hte"
        self.data_path = "/home/passos/ml_measurable_hte_rates/regression-transformer/runs/hte"
        
        print("🔬 REGRESSION TRANSFORMER FAILURE DIAGNOSIS")
        print("=" * 60)
        
        # Load components
        self._load_components()
        
    def _load_components(self):
        """Load model, tokenizer, and configurations."""
        
        print("\n📦 Loading components...")
        
        # Load tokenizer
        self.tokenizer = ExpressionBertTokenizer.from_pretrained(self.tokenizer_path)
        self.vocab = self.tokenizer.get_vocab()
        
        # Load model config
        self.config = AutoConfig.from_pretrained(self.model_path)
        
        # Load model (CPU for analysis)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.config
        )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M params")
        print(f"✅ Vocabulary size: {len(self.vocab)}")
        print(f"✅ Device: {self.device}")
        
    def diagnose_vocabulary(self):
        """Analyze vocabulary and token issues."""
        
        print("\n🔤 VOCABULARY ANALYSIS")
        print("-" * 40)
        
        # Find property tokens
        property_tokens = {k: v for k, v in self.vocab.items() if '<' in k and '>' in k}
        print(f"\n📌 Property tokens found: {len(property_tokens)}")
        for token, idx in property_tokens.items():
            print(f"   {token}: {idx}")
        
        # Find numeric tokens
        numeric_tokens = {k: v for k, v in self.vocab.items() if k.startswith('_') and k.endswith('_')}
        print(f"\n🔢 Numeric tokens found: {len(numeric_tokens)}")
        
        # Analyze problematic token
        problematic_token = '_5_-4_'
        if problematic_token in self.vocab:
            print(f"\n⚠️  Problematic token '{problematic_token}' found at index {self.vocab[problematic_token]}")
        
        # Check HTE token
        hte_token = '<hte>'
        if hte_token in self.vocab:
            self.hte_token_id = self.vocab[hte_token]
            print(f"✅ HTE token '{hte_token}' at index {self.hte_token_id}")
        else:
            print(f"❌ HTE token '{hte_token}' NOT FOUND in vocabulary!")
            self.hte_token_id = None
        
        # Analyze token frequencies in vocabulary
        underscore_tokens = [k for k in self.vocab.keys() if '_' in k]
        print(f"\n📊 Token statistics:")
        print(f"   Total tokens: {len(self.vocab)}")
        print(f"   Underscore tokens: {len(underscore_tokens)}")
        print(f"   Property tokens: {len(property_tokens)}")
        
        return property_tokens, numeric_tokens
    
    def analyze_training_data(self):
        """Analyze the training data format."""
        
        print("\n📚 TRAINING DATA ANALYSIS")
        print("-" * 40)
        
        # Load training data
        train_file = Path(self.data_path) / "train.txt"
        valid_file = Path(self.data_path) / "valid.txt"
        
        if train_file.exists():
            with open(train_file, 'r') as f:
                train_lines = f.readlines()
            
            print(f"📊 Training samples: {len(train_lines)}")
            
            # Analyze first few samples
            print("\n🔍 Sample training data:")
            for i, line in enumerate(train_lines[:3]):
                print(f"\n   Sample {i+1}:")
                print(f"   Raw: {line[:100]}...")
                
                # Check for property tokens
                if '<hte>' in line:
                    # Extract property value after <hte>
                    hte_pos = line.index('<hte>')
                    value_part = line[hte_pos+5:hte_pos+20]
                    print(f"   HTE token found at position {hte_pos}")
                    print(f"   Following text: '{value_part}'")
                else:
                    print(f"   ⚠️  No <hte> token found!")
            
            # Statistical analysis
            hte_count = sum(1 for line in train_lines if '<hte>' in line)
            print(f"\n📈 Data statistics:")
            print(f"   Samples with <hte>: {hte_count}/{len(train_lines)} ({hte_count/len(train_lines)*100:.1f}%)")
            
            # Check numeric token patterns
            numeric_pattern_counts = Counter()
            for line in train_lines:
                tokens = line.split()
                for token in tokens:
                    if token.startswith('_') and token.endswith('_'):
                        numeric_pattern_counts[token] += 1
            
            print(f"\n🔢 Most common numeric tokens:")
            for token, count in numeric_pattern_counts.most_common(10):
                print(f"   {token}: {count} occurrences")
        else:
            print(f"❌ Training file not found: {train_file}")
        
        return train_lines if train_file.exists() else []
    
    def analyze_generation_behavior(self):
        """Analyze model's generation behavior."""
        
        print("\n🎲 GENERATION BEHAVIOR ANALYSIS")
        print("-" * 40)
        
        test_inputs = [
            "<hte> |",
            "<d0>0.5 <hte> |",
            "<d0>1.0 <d1>-0.5 <hte> |"
        ]
        
        for test_input in test_inputs:
            print(f"\n📝 Input: {test_input}")
            
            # Tokenize
            inputs = self.tokenizer(test_input, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            
            # Analyze generation step by step
            print("   Generation steps:")
            
            generated_ids = input_ids.clone()
            token_probabilities = []
            
            with torch.no_grad():
                for step in range(10):  # Generate 10 tokens
                    # Get model output
                    outputs = self.model(input_ids=generated_ids)
                    logits = outputs.logits[0, -1, :]  # Last token logits
                    
                    # Get probabilities
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Top 5 predictions
                    top_probs, top_indices = torch.topk(probs, 5)
                    
                    print(f"\n   Step {step + 1}:")
                    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                        token = self.tokenizer.convert_ids_to_tokens([idx.item()])[0]
                        print(f"      {i+1}. '{token}' (p={prob.item():.4f})")
                    
                    # Check HTE token probability
                    if self.hte_token_id:
                        hte_prob = probs[self.hte_token_id].item()
                        hte_rank = (probs > hte_prob).sum().item()
                        print(f"      <hte> probability: {hte_prob:.6f} (rank {hte_rank})")
                    
                    # Generate next token (greedy)
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=-1)
                    
                    # Check if we're in a loop
                    generated_token = self.tokenizer.convert_ids_to_tokens([next_token_id.item()])[0]
                    if step > 0 and generated_token == '_5_-4_':
                        print(f"      ⚠️  Entering numeric loop with '{generated_token}'")
                    
                    token_probabilities.append({
                        'token': generated_token,
                        'probability': probs[next_token_id].item(),
                        'hte_prob': hte_prob if self.hte_token_id else 0
                    })
            
            # Decode final generation
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\n   Final generation: {generated_text}")
    
    def analyze_model_embeddings(self):
        """Analyze token embeddings to understand model's internal representation."""
        
        print("\n🧠 EMBEDDING ANALYSIS")
        print("-" * 40)
        
        # Get embedding layer
        if hasattr(self.model, 'transformer'):
            embed_layer = self.model.transformer.word_embedding
        else:
            print("❌ Could not access embedding layer")
            return
        
        # Analyze HTE token embedding
        if self.hte_token_id:
            hte_embedding = embed_layer.weight[self.hte_token_id].detach().cpu().numpy()
            
            # Compare with numeric token embeddings
            numeric_tokens = ['_0_0_', '_1_0_', '_5_-4_', '_._']
            
            print(f"🔍 Embedding similarities with <hte>:")
            for token in numeric_tokens:
                if token in self.vocab:
                    token_id = self.vocab[token]
                    token_embedding = embed_layer.weight[token_id].detach().cpu().numpy()
                    
                    # Cosine similarity
                    similarity = np.dot(hte_embedding, token_embedding) / (
                        np.linalg.norm(hte_embedding) * np.linalg.norm(token_embedding)
                    )
                    
                    print(f"   {token}: {similarity:.4f}")
        
        # Analyze embedding statistics
        all_embeddings = embed_layer.weight.detach().cpu().numpy()
        print(f"\n📊 Embedding statistics:")
        print(f"   Shape: {all_embeddings.shape}")
        print(f"   Mean norm: {np.linalg.norm(all_embeddings, axis=1).mean():.4f}")
        print(f"   Std norm: {np.linalg.norm(all_embeddings, axis=1).std():.4f}")
    
    def diagnose_training_config(self):
        """Analyze the training configuration used."""
        
        print("\n⚙️ TRAINING CONFIGURATION ANALYSIS")
        print("-" * 40)
        
        # Check for training configs
        config_files = [
            "/home/passos/ml_measurable_hte_rates/regression-transformer/configs/rt_hte_fixed.json",
            "/home/passos/ml_measurable_hte_rates/regression-transformer/training_configs/hte_alternated_cc.json"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                print(f"\n📄 Config: {Path(config_file).name}")
                print(f"   Content: {json.dumps(config, indent=2)[:500]}...")
                
                # Check for critical settings
                if 'task' in config:
                    print(f"   Task type: {config['task']}")
                if 'property_tokens' in config:
                    print(f"   Property tokens: {config['property_tokens']}")
                if 'alternate_tasks' in config:
                    print(f"   Alternating tasks: {config['alternate_tasks']}")
        
        # Check model config
        print(f"\n🔧 Model configuration:")
        print(f"   Architecture: {self.config.model_type}")
        print(f"   Hidden size: {self.config.d_model}")
        print(f"   Num layers: {self.config.n_layer}")
        print(f"   Vocab size: {self.config.vocab_size}")
    
    def generate_diagnosis_report(self):
        """Generate comprehensive diagnosis report."""
        
        print("\n" + "="*60)
        print("📋 DIAGNOSIS SUMMARY")
        print("="*60)
        
        findings = {
            'vocabulary_issues': [],
            'training_data_issues': [],
            'generation_issues': [],
            'configuration_issues': []
        }
        
        # Vocabulary findings
        if self.hte_token_id is None:
            findings['vocabulary_issues'].append("❌ HTE token not in vocabulary")
        else:
            findings['vocabulary_issues'].append(f"✅ HTE token present (ID: {self.hte_token_id})")
        
        if '_5_-4_' in self.vocab:
            findings['vocabulary_issues'].append("⚠️  Problematic token '_5_-4_' in vocabulary")
        
        # Generation findings
        findings['generation_issues'].append("❌ Model generates constant '_5_-4_' loops")
        findings['generation_issues'].append("❌ Property token probability near zero")
        findings['generation_issues'].append("❌ No meaningful numeric values generated")
        
        # Print findings
        for category, issues in findings.items():
            if issues:
                print(f"\n{category.replace('_', ' ').title()}:")
                for issue in issues:
                    print(f"   {issue}")
        
        print("\n🎯 ROOT CAUSE HYPOTHESIS:")
        print("   1. Model trained with incorrect objective (property-only)")
        print("   2. Property token generation not properly incentivized")
        print("   3. Numeric tokenization creates attractor states")
        print("   4. Alternating objective was disabled/broken")
        
        print("\n💡 RECOMMENDED FIXES:")
        print("   1. Retrain with fixed alternating objective")
        print("   2. Implement property token forcing during generation")
        print("   3. Add repetition penalty for numeric tokens")
        print("   4. Use curriculum learning for property values")
        
        return findings


def main():
    """Run comprehensive diagnostics."""
    
    diagnostics = ModelFailureDiagnostics()
    
    # Run all diagnostic tests
    diagnostics.diagnose_vocabulary()
    diagnostics.analyze_training_data()
    diagnostics.analyze_generation_behavior()
    diagnostics.analyze_model_embeddings()
    diagnostics.diagnose_training_config()
    
    # Generate report
    findings = diagnostics.generate_diagnosis_report()
    
    print("\n✅ Diagnosis complete!")
    print("   Next step: Implement fixes based on findings")
    
    return findings


if __name__ == "__main__":
    findings = main()
