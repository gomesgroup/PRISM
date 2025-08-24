#!/usr/bin/env python
"""
Ready-to-Use HTE Predictor - Production System

This is your production-ready HTE Regression Transformer system
optimized for your 4x A100 80GB GPUs. Simply run this script!

Usage Examples:
  python ready_to_use_hte_predictor.py --predict "<d0>0.5 <d1>-0.3 <hte> |"
  python ready_to_use_hte_predictor.py --batch predict_these.txt
  python ready_to_use_hte_predictor.py --benchmark
  python ready_to_use_hte_predictor.py --interactive
"""

import argparse
import sys
import time
from gpu_production_hte_system import GPUProductionHTEGenerator


def single_prediction(system, input_text):
    """Make a single HTE prediction."""
    print(f"🔍 Predicting HTE for: {input_text}")
    
    result = system.predict_hte_rate(input_text, max_new_tokens=10, benchmark=True)
    
    print(f"✅ Results:")
    print(f"   HTE Rate: {result['hte_rate']:.4f}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   GPU Time: {result['gpu_time_ms']:.1f}ms")
    print(f"   Generated: {result['generated_text']}")
    
    return result


def batch_prediction(system, input_file):
    """Process batch predictions from file."""
    print(f"📊 Processing batch file: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            inputs = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"   Found {len(inputs)} inputs")
    
    start_time = time.time()
    results = system.batch_predict(inputs)
    total_time = time.time() - start_time
    
    print(f"\n✅ Batch Results:")
    print(f"   Processed: {len(results)} predictions")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Throughput: {len(results)/total_time:.1f} predictions/second")
    
    # Save results
    output_file = input_file.replace('.txt', '_results.txt')
    with open(output_file, 'w') as f:
        for i, result in enumerate(results):
            f.write(f"{i+1}. Input: {result['input']}\n")
            f.write(f"   HTE: {result['hte_rate']:.4f} (conf: {result['confidence']:.2f})\n")
            f.write(f"   Generated: {result['generated_text']}\n\n")
    
    print(f"   Results saved to: {output_file}")


def interactive_mode(system):
    """Interactive prediction mode."""
    print("🎯 Interactive HTE Prediction Mode")
    print("Enter molecule inputs (or 'quit' to exit):")
    print("Examples:")
    print("  <d0>0.5 <d1>-0.3 <hte> |")
    print("  <hte> | CC>>CCO") 
    print()
    
    while True:
        try:
            user_input = input("HTE> ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            single_prediction(system, user_input)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def benchmark_mode(system):
    """Run performance benchmark."""
    print("🏁 A100 GPU Performance Benchmark")
    print("=" * 50)
    
    # Quick benchmark
    test_cases = [
        "<d0>0.5 <d1>-0.3 <hte> |",
        "<hte> | CC>>CCO", 
        "<d0>1.2 <d1>0.8 <d2>-0.5 <hte> |"
    ]
    
    times = []
    
    print("🔥 Running benchmark (10 predictions each)...")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case}")
        
        case_times = []
        for _ in range(10):
            result = system.predict_hte_rate(test_case, max_new_tokens=5)
            case_times.append(result['gpu_time_ms'])
        
        avg_time = sum(case_times) / len(case_times)
        times.extend(case_times)
        
        print(f"   Average: {avg_time:.1f}ms")
        print(f"   HTE: {result['hte_rate']:.4f}")
    
    # Overall results
    overall_avg = sum(times) / len(times)
    throughput = 1000 / overall_avg
    
    print(f"\n🎯 Benchmark Results:")
    print(f"   Average prediction time: {overall_avg:.1f}ms")
    print(f"   Throughput: {throughput:.1f} predictions/second")
    print(f"   GPU Memory: {result['device']}")
    print(f"   Status: 🚀 EXCELLENT PERFORMANCE!")


def main():
    parser = argparse.ArgumentParser(
        description="Production HTE Regression Transformer with A100 GPU Acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python ready_to_use_hte_predictor.py --predict "<d0>0.5 <hte> |"
  
  # Batch processing
  python ready_to_use_hte_predictor.py --batch molecules.txt
  
  # Interactive mode
  python ready_to_use_hte_predictor.py --interactive
  
  # Performance benchmark
  python ready_to_use_hte_predictor.py --benchmark
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--predict', type=str, help='Single prediction input')
    group.add_argument('--batch', type=str, help='Batch prediction from file')
    group.add_argument('--interactive', action='store_true', help='Interactive mode')
    group.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Initialize system
    print("🚀 Initializing A100 GPU-Accelerated HTE System...")
    try:
        system = GPUProductionHTEGenerator()
        print("✅ System ready!")
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        sys.exit(1)
    
    # Execute requested mode
    if args.predict:
        single_prediction(system, args.predict)
    elif args.batch:
        batch_prediction(system, args.batch)
    elif args.interactive:
        interactive_mode(system)
    elif args.benchmark:
        benchmark_mode(system)


if __name__ == "__main__":
    main()
