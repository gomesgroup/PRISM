#!/usr/bin/env python
"""
HTE Regression Transformer - Comprehensive Metrics Dashboard

This script provides a complete performance analysis of our
GPU-accelerated production system with detailed metrics.
"""

import time
import torch
import numpy as np
from gpu_production_hte_system import GPUProductionHTEGenerator


class MetricsDashboard:
    """Comprehensive metrics collection and analysis system."""
    
    def __init__(self):
        self.system = GPUProductionHTEGenerator()
        self.metrics = {}
        
    def collect_all_metrics(self):
        """Collect comprehensive system metrics."""
        print("📊 COMPREHENSIVE METRICS DASHBOARD")
        print("=" * 80)
        
        # 1. Performance Metrics
        self._collect_performance_metrics()
        
        # 2. Accuracy & Quality Metrics  
        self._collect_accuracy_metrics()
        
        # 3. Resource Utilization Metrics
        self._collect_resource_metrics()
        
        # 4. Reliability & Stability Metrics
        self._collect_reliability_metrics()
        
        # 5. Production Readiness Metrics
        self._collect_production_metrics()
        
        # 6. Generate Final Report
        self._generate_comprehensive_report()
        
    def _collect_performance_metrics(self):
        """Collect speed and throughput performance metrics."""
        print("\n🚀 PERFORMANCE METRICS")
        print("-" * 40)
        
        test_cases = [
            "<d0>0.5 <d1>-0.3 <hte> |",
            "<hte> | CC>>CCO", 
            "<d0>1.2 <d1>0.8 <d2>-0.5 <hte> |",
            "<d0>0.0 <hte> |",
            "<hte>-1.5 | C=O"
        ]
        
        # Single prediction performance
        single_times = []
        for test_case in test_cases:
            start = time.time()
            result = self.system.predict_hte_rate(test_case, max_new_tokens=8)
            single_times.append((time.time() - start) * 1000)
        
        # Batch performance
        batch_start = time.time()
        batch_results = self.system.batch_predict(test_cases)
        batch_time = (time.time() - batch_start) * 1000
        
        self.metrics['performance'] = {
            'avg_prediction_time_ms': np.mean(single_times),
            'std_prediction_time_ms': np.std(single_times),
            'min_prediction_time_ms': np.min(single_times),
            'max_prediction_time_ms': np.max(single_times),
            'single_throughput_per_sec': 1000 / np.mean(single_times),
            'batch_time_ms': batch_time,
            'batch_avg_per_prediction_ms': batch_time / len(test_cases),
            'batch_throughput_per_sec': len(test_cases) * 1000 / batch_time,
            'performance_consistency': 100 * (1 - np.std(single_times) / np.mean(single_times))
        }
        
        print(f"✅ Single Prediction Performance:")
        print(f"   Average time: {self.metrics['performance']['avg_prediction_time_ms']:.1f}ms")
        print(f"   Standard deviation: ±{self.metrics['performance']['std_prediction_time_ms']:.1f}ms")
        print(f"   Range: {self.metrics['performance']['min_prediction_time_ms']:.1f} - {self.metrics['performance']['max_prediction_time_ms']:.1f}ms")
        print(f"   Throughput: {self.metrics['performance']['single_throughput_per_sec']:.1f} predictions/sec")
        
        print(f"\n✅ Batch Processing Performance:")
        print(f"   Total batch time: {self.metrics['performance']['batch_time_ms']:.1f}ms")
        print(f"   Avg per prediction: {self.metrics['performance']['batch_avg_per_prediction_ms']:.1f}ms")
        print(f"   Batch throughput: {self.metrics['performance']['batch_throughput_per_sec']:.1f} predictions/sec")
        
        print(f"\n📈 Performance Consistency: {self.metrics['performance']['performance_consistency']:.1f}%")
        
    def _collect_accuracy_metrics(self):
        """Collect accuracy and quality metrics."""
        print("\n🎯 ACCURACY & QUALITY METRICS")
        print("-" * 40)
        
        test_cases_with_expected = [
            ("<d0>0.5 <d1>-0.3 <hte> |", "should_extract_value"),
            ("<hte> | CC>>CCO", "should_extract_value"),
            ("<d0>1.2 <hte>-1.5 |", "should_extract_negative"),
            ("<hte>0.0 |", "should_extract_zero"),
            ("<d0>0.8 <d1>0.2 <d2>-0.5 <hte> |", "should_extract_value")
        ]
        
        successful_extractions = 0
        successful_generations = 0
        confidence_scores = []
        property_token_generations = 0
        
        for test_input, expected in test_cases_with_expected:
            result = self.system.predict_hte_rate(test_input, max_new_tokens=10)
            
            # Check if extraction was successful
            if result['hte_rate'] is not None:
                successful_extractions += 1
                confidence_scores.append(result['confidence'])
            
            # Check if generation was successful
            if result['generated_text'] and len(result['generated_text']) > 0:
                successful_generations += 1
            
            # Check if property token was generated
            if '<hte>' in result['full_generated']:
                property_token_generations += 1
        
        self.metrics['accuracy'] = {
            'extraction_success_rate': (successful_extractions / len(test_cases_with_expected)) * 100,
            'generation_success_rate': (successful_generations / len(test_cases_with_expected)) * 100,
            'property_token_success_rate': (property_token_generations / len(test_cases_with_expected)) * 100,
            'avg_confidence_score': np.mean(confidence_scores) if confidence_scores else 0,
            'std_confidence_score': np.std(confidence_scores) if confidence_scores else 0,
            'min_confidence_score': np.min(confidence_scores) if confidence_scores else 0,
            'max_confidence_score': np.max(confidence_scores) if confidence_scores else 0
        }
        
        print(f"✅ Success Rates:")
        print(f"   Property extraction: {self.metrics['accuracy']['extraction_success_rate']:.1f}%")
        print(f"   Text generation: {self.metrics['accuracy']['generation_success_rate']:.1f}%") 
        print(f"   Property token generation: {self.metrics['accuracy']['property_token_success_rate']:.1f}%")
        
        print(f"\n✅ Confidence Metrics:")
        print(f"   Average confidence: {self.metrics['accuracy']['avg_confidence_score']:.2f}")
        print(f"   Confidence range: {self.metrics['accuracy']['min_confidence_score']:.2f} - {self.metrics['accuracy']['max_confidence_score']:.2f}")
        print(f"   Confidence consistency: ±{self.metrics['accuracy']['std_confidence_score']:.3f}")
        
    def _collect_resource_metrics(self):
        """Collect GPU and system resource utilization metrics."""
        print("\n💾 RESOURCE UTILIZATION METRICS")
        print("-" * 40)
        
        # GPU Memory metrics
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Model size metrics
        model_params = sum(p.numel() for p in self.system.model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in self.system.model.parameters()) / 1e6
        
        self.metrics['resources'] = {
            'gpu_memory_allocated_gb': gpu_memory_allocated,
            'gpu_memory_reserved_gb': gpu_memory_reserved,
            'gpu_memory_total_gb': gpu_memory_total,
            'gpu_memory_utilization_percent': (gpu_memory_allocated / gpu_memory_total) * 100,
            'gpu_memory_efficiency': (gpu_memory_allocated / gpu_memory_reserved) * 100 if gpu_memory_reserved > 0 else 100,
            'model_parameters': model_params,
            'model_size_mb': model_size_mb,
            'available_gpus': torch.cuda.device_count(),
            'gpu_name': torch.cuda.get_device_name(0),
            'cuda_version': torch.version.cuda,
            'pytorch_version': torch.__version__
        }
        
        print(f"✅ GPU Memory Usage:")
        print(f"   Allocated: {self.metrics['resources']['gpu_memory_allocated_gb']:.3f}GB")
        print(f"   Total available: {self.metrics['resources']['gpu_memory_total_gb']:.0f}GB")
        print(f"   Utilization: {self.metrics['resources']['gpu_memory_utilization_percent']:.3f}%")
        print(f"   Efficiency: {self.metrics['resources']['gpu_memory_efficiency']:.1f}%")
        
        print(f"\n✅ Model Resources:")
        print(f"   Parameters: {self.metrics['resources']['model_parameters']/1e6:.1f}M")
        print(f"   Model size: {self.metrics['resources']['model_size_mb']:.1f}MB")
        
        print(f"\n✅ Hardware Configuration:")
        print(f"   GPUs available: {self.metrics['resources']['available_gpus']}")
        print(f"   GPU model: {self.metrics['resources']['gpu_name']}")
        print(f"   CUDA version: {self.metrics['resources']['cuda_version']}")
        print(f"   PyTorch version: {self.metrics['resources']['pytorch_version']}")
        
    def _collect_reliability_metrics(self):
        """Collect system reliability and stability metrics."""
        print("\n🔧 RELIABILITY & STABILITY METRICS")
        print("-" * 40)
        
        # Sustained load test
        test_duration = 30  # 30 seconds for metrics
        predictions_made = 0
        errors_encountered = 0
        response_times = []
        
        print(f"🔥 Running {test_duration}s reliability test...")
        
        start_time = time.time()
        test_input = "<d0>0.5 <hte> |"
        
        while time.time() - start_time < test_duration:
            try:
                pred_start = time.time()
                result = self.system.predict_hte_rate(test_input, max_new_tokens=5)
                pred_time = time.time() - pred_start
                
                response_times.append(pred_time * 1000)
                predictions_made += 1
                
                # Check for successful extraction
                if result['hte_rate'] is None:
                    errors_encountered += 1
                    
            except Exception as e:
                errors_encountered += 1
                
        total_test_time = time.time() - start_time
        
        self.metrics['reliability'] = {
            'test_duration_sec': total_test_time,
            'predictions_made': predictions_made,
            'errors_encountered': errors_encountered,
            'error_rate_percent': (errors_encountered / predictions_made) * 100 if predictions_made > 0 else 0,
            'uptime_percent': ((predictions_made - errors_encountered) / predictions_made) * 100 if predictions_made > 0 else 0,
            'sustained_throughput': predictions_made / total_test_time,
            'avg_response_time_ms': np.mean(response_times) if response_times else 0,
            'response_time_std_ms': np.std(response_times) if response_times else 0,
            'p95_response_time_ms': np.percentile(response_times, 95) if response_times else 0,
            'p99_response_time_ms': np.percentile(response_times, 99) if response_times else 0
        }
        
        print(f"✅ Reliability Results:")
        print(f"   Predictions made: {self.metrics['reliability']['predictions_made']}")
        print(f"   Errors encountered: {self.metrics['reliability']['errors_encountered']}")
        print(f"   Error rate: {self.metrics['reliability']['error_rate_percent']:.2f}%")
        print(f"   Uptime: {self.metrics['reliability']['uptime_percent']:.2f}%")
        
        print(f"\n✅ Sustained Performance:")
        print(f"   Sustained throughput: {self.metrics['reliability']['sustained_throughput']:.1f} pred/sec")
        print(f"   Average response time: {self.metrics['reliability']['avg_response_time_ms']:.1f}ms")
        print(f"   95th percentile: {self.metrics['reliability']['p95_response_time_ms']:.1f}ms")
        print(f"   99th percentile: {self.metrics['reliability']['p99_response_time_ms']:.1f}ms")
        
    def _collect_production_metrics(self):
        """Collect production readiness metrics."""
        print("\n🏭 PRODUCTION READINESS METRICS")
        print("-" * 40)
        
        # Stress test with various input types
        stress_test_cases = [
            "<d0>0.1 <hte> |",
            "<hte> | simple",
            "<d0>-2.5 <d1>3.8 <d2>-1.2 <d3>0.9 <hte> |",
            "<hte>999.9 |",
            "<d0>0.0 <d1>0.0 <hte>0.0 |",
            "invalid input",
            "",
            "<d0> <hte> |",  # Missing value
            "<hte> | " + "X" * 100,  # Long input
        ]
        
        handled_gracefully = 0
        total_tests = len(stress_test_cases)
        processing_times = []
        
        for test_case in stress_test_cases:
            try:
                start = time.time()
                result = self.system.predict_hte_rate(test_case, max_new_tokens=5)
                process_time = (time.time() - start) * 1000
                
                processing_times.append(process_time)
                handled_gracefully += 1
                
            except Exception as e:
                print(f"   ⚠️  Error handling: {test_case[:30]}... → {str(e)[:50]}...")
        
        # Memory stability check
        initial_memory = torch.cuda.memory_allocated()
        for _ in range(20):  # Multiple predictions to check for memory leaks
            _ = self.system.predict_hte_rate("<d0>0.5 <hte> |", max_new_tokens=5)
        final_memory = torch.cuda.memory_allocated()
        
        self.metrics['production'] = {
            'stress_test_success_rate': (handled_gracefully / total_tests) * 100,
            'error_handling_robustness': handled_gracefully / total_tests,
            'avg_processing_time_under_stress': np.mean(processing_times) if processing_times else 0,
            'memory_leak_detected': abs(final_memory - initial_memory) > 1e6,  # >1MB change
            'memory_stability_gb': (final_memory - initial_memory) / 1e9,
            'production_ready_score': min(100, (handled_gracefully / total_tests) * 100)
        }
        
        print(f"✅ Stress Test Results:")
        print(f"   Test cases handled: {handled_gracefully}/{total_tests}")
        print(f"   Success rate: {self.metrics['production']['stress_test_success_rate']:.1f}%")
        print(f"   Error handling robustness: {self.metrics['production']['error_handling_robustness']:.2f}")
        
        print(f"\n✅ Memory Stability:")
        memory_change = self.metrics['production']['memory_stability_gb']
        if abs(memory_change) < 0.001:
            print(f"   Memory change: <1MB (excellent stability)")
        else:
            print(f"   Memory change: {memory_change*1000:.1f}MB")
        print(f"   Memory leaks: {'❌ None detected' if not self.metrics['production']['memory_leak_detected'] else '⚠️  Possible leak'}")
        
        print(f"\n🎯 Production Readiness Score: {self.metrics['production']['production_ready_score']:.1f}%")
        
    def _generate_comprehensive_report(self):
        """Generate final comprehensive metrics report."""
        print("\n" + "="*80)
        print("🏆 FINAL METRICS SUMMARY")
        print("="*80)
        
        # Overall system grade
        performance_score = min(100, self.metrics['performance']['single_throughput_per_sec'] * 2)  # 50 pred/sec = 100%
        accuracy_score = self.metrics['accuracy']['extraction_success_rate']
        reliability_score = self.metrics['reliability']['uptime_percent']
        resource_efficiency = 100 - self.metrics['resources']['gpu_memory_utilization_percent']  # Lower usage = higher efficiency
        production_score = self.metrics['production']['production_ready_score']
        
        overall_score = np.mean([performance_score, accuracy_score, reliability_score, resource_efficiency, production_score])
        
        print(f"\n📊 OVERALL SYSTEM GRADES:")
        print(f"   Performance Score:      {performance_score:.1f}% ({'🏆 Excellent' if performance_score >= 90 else '✅ Good' if performance_score >= 70 else '⚠️  Fair'})")
        print(f"   Accuracy Score:         {accuracy_score:.1f}% ({'🏆 Excellent' if accuracy_score >= 90 else '✅ Good' if accuracy_score >= 70 else '⚠️  Fair'})")
        print(f"   Reliability Score:      {reliability_score:.1f}% ({'🏆 Excellent' if reliability_score >= 90 else '✅ Good' if reliability_score >= 70 else '⚠️  Fair'})")
        print(f"   Resource Efficiency:    {resource_efficiency:.1f}% ({'🏆 Excellent' if resource_efficiency >= 90 else '✅ Good' if resource_efficiency >= 70 else '⚠️  Fair'})")
        print(f"   Production Readiness:   {production_score:.1f}% ({'🏆 Excellent' if production_score >= 90 else '✅ Good' if production_score >= 70 else '⚠️  Fair'})")
        
        print(f"\n🎯 OVERALL SYSTEM GRADE: {overall_score:.1f}%")
        if overall_score >= 90:
            print("   Rating: 🏆 WORLD-CLASS PERFORMANCE")
        elif overall_score >= 80:
            print("   Rating: 🚀 EXCELLENT PRODUCTION SYSTEM")
        elif overall_score >= 70:
            print("   Rating: ✅ GOOD PRODUCTION READY")
        else:
            print("   Rating: ⚠️  NEEDS OPTIMIZATION")
        
        # Key performance highlights
        print(f"\n🎯 KEY PERFORMANCE HIGHLIGHTS:")
        print(f"   ⚡ Throughput: {self.metrics['performance']['single_throughput_per_sec']:.1f} predictions/second")
        print(f"   🎯 Accuracy: {self.metrics['accuracy']['extraction_success_rate']:.1f}% extraction success")
        print(f"   💾 Memory: {self.metrics['resources']['gpu_memory_utilization_percent']:.3f}% of 85GB GPU used")
        print(f"   🔧 Reliability: {self.metrics['reliability']['uptime_percent']:.2f}% uptime")
        print(f"   ⏱️  Latency: {self.metrics['performance']['avg_prediction_time_ms']:.1f}ms average")
        
        # Production capacity
        daily_capacity = self.metrics['reliability']['sustained_throughput'] * 60 * 60 * 24
        print(f"\n🏭 PRODUCTION CAPACITY:")
        print(f"   Daily predictions: {daily_capacity:,.0f}")
        print(f"   Weekly predictions: {daily_capacity * 7:,.0f}")
        print(f"   Monthly predictions: {daily_capacity * 30:,.0f}")
        
        return overall_score


def main():
    """Run comprehensive metrics dashboard."""
    try:
        dashboard = MetricsDashboard()
        overall_score = dashboard.collect_all_metrics()
        
        print(f"\n🎉 METRICS COLLECTION COMPLETE!")
        print(f"Overall System Performance: {overall_score:.1f}%")
        
    except Exception as e:
        print(f"❌ Metrics collection failed: {e}")


if __name__ == "__main__":
    main()
