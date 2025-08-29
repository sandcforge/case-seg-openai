#!/usr/bin/env python3
"""
Statistics validation script for session output.

Reads all case JSON files from a session output directory, calculates statistics,
and compares with the official statistics_.json file to validate accuracy.

Usage:
    python test_session_statistics.py <session_directory>
    
Example:
    python test_session_statistics.py ../out/session_250828_221716
"""

import json
import sys
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class SessionStatisticsValidator:
    """Validates session statistics by recalculating from case data"""
    
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.all_cases: List[Dict[str, Any]] = []
        self.official_stats: Optional[Dict[str, Any]] = None
        
    def load_case_files(self) -> None:
        """Load all cases_*.json files from the session directory"""
        case_files = list(self.session_dir.glob("cases_*.json"))
        
        if not case_files:
            raise FileNotFoundError(f"No cases_*.json files found in {self.session_dir}")
            
        print(f"Found {len(case_files)} case files")
        
        # Store cases with channel information
        self.channel_cases = []  # List of (channel_url, case) tuples
        
        for case_file in case_files:
            with open(case_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                channel_url = data.get("channel_url", "unknown")
                cases = data.get("global_cases", [])
                
                # Add cases to main list
                self.all_cases.extend(cases)
                
                # Store with channel info
                for case in cases:
                    self.channel_cases.append((channel_url, case))
                
        print(f"Loaded {len(self.all_cases)} total cases from {len(case_files)} channels")
        
        # Print corner cases
        self._print_corner_cases()
        
    def load_official_statistics(self) -> None:
        """Load the official statistics_.json file"""
        stats_files = list(self.session_dir.glob("statistics_*.json"))
        
        if not stats_files:
            raise FileNotFoundError(f"No statistics_*.json file found in {self.session_dir}")
            
        if len(stats_files) > 1:
            print(f"Warning: Multiple statistics files found, using {stats_files[0]}")
            
        with open(stats_files[0], 'r', encoding='utf-8') as f:
            self.official_stats = json.load(f)
            
    def calculate_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics from case data"""
        metrics = {}
        
        # Extract metric values from all cases
        handle_times = [case.get("handle_time", 0) for case in self.all_cases if case.get("handle_time") is not None]
        first_res_times = [case.get("first_res_time") for case in self.all_cases if case.get("first_res_time") is not None and case.get("first_res_time") >= 0]
        usr_msg_nums = [case.get("usr_msg_num", 0) for case in self.all_cases if case.get("usr_msg_num") is not None]
        
        # Calculate handle_time metrics
        metrics["handle_time"] = self._calculate_metric_stats(handle_times, len(self.all_cases))
        
        # Calculate first_res_time metrics (only valid cases)
        metrics["first_res_time"] = self._calculate_metric_stats(first_res_times, len(self.all_cases))
        
        # Calculate usr_msg_num metrics  
        metrics["usr_msg_num"] = self._calculate_metric_stats(usr_msg_nums, len(self.all_cases))
        
        return metrics
        
    def _calculate_metric_stats(self, values: List[float], total_cases: int) -> Dict[str, Any]:
        """Calculate statistics for a metric"""
        if not values:
            return {
                "valid_cases": 0,
                "total_cases": total_cases,
                "validity_rate": 0.0,
                "percentiles": {"P5": 0, "P50": 0, "P95": 0},
                "basic_stats": {"min": 0, "max": 0, "mean": 0.0, "std": 0.0}
            }
            
        sorted_values = sorted(values)
        n = len(values)
        
        return {
            "valid_cases": n,
            "total_cases": total_cases,
            "validity_rate": n / total_cases * 100.0,
            "percentiles": {
                "P5": self._percentile(sorted_values, 5),
                "P50": self._percentile(sorted_values, 50),
                "P95": self._percentile(sorted_values, 95)
            },
            "basic_stats": {
                "min": min(values),
                "max": max(values),
                "mean": round(statistics.mean(values), 2),
                "std": round(statistics.pstdev(values), 2)  # Population std dev to match np.std()
            }
        }
    
    def _percentile(self, sorted_values: List[float], percentile: float) -> float:
        """Calculate percentile from sorted values"""
        if not sorted_values:
            return 0.0
            
        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]
            
        # Use the same method as numpy's percentile with linear interpolation
        index = (percentile / 100.0) * (n - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, n - 1)
        
        if lower_index == upper_index:
            return sorted_values[lower_index]
            
        # Linear interpolation
        weight = index - lower_index
        return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight
    
    def _format_channel_for_display(self, channel_url: str) -> str:
        """Format channel URL for display: extract hash part after last underscore"""
        if '_' not in channel_url:
            return channel_url
        return channel_url.split('_')[-1]
    
    def _print_corner_cases(self) -> None:
        """Print corner cases with handle_time=0 or first_res_time=0"""
        print("\n=== CORNER CASES ===")
        
        # Handle time = 0 cases
        handle_time_zero = [(channel_url, case) for channel_url, case in self.channel_cases if case.get("handle_time") == 0]
        print(f"Handle Time = 0: {len(handle_time_zero)} cases")
        for channel_url, case in handle_time_zero[:3]:  # Show first 3
            channel_display = self._format_channel_for_display(channel_url)
            print(f"  - Case {case.get('case_id', 'unknown')} (Channel: {channel_display}): "
                  f"status={case.get('status')}, usr_msg={case.get('usr_msg_num')}, total_msg={case.get('total_msg_num')}")
        if len(handle_time_zero) > 3:
            print(f"  ... and {len(handle_time_zero) - 3} more")
            
        # First response time = 0 cases  
        first_res_zero = [(channel_url, case) for channel_url, case in self.channel_cases if case.get("first_res_time") == 0]
        print(f"\nFirst Response Time = 0: {len(first_res_zero)} cases")
        for channel_url, case in first_res_zero[:3]:  # Show first 3
            channel_display = self._format_channel_for_display(channel_url)
            print(f"  - Case {case.get('case_id', 'unknown')} (Channel: {channel_display}): "
                  f"status={case.get('status')}, handle_time={case.get('handle_time')}, usr_msg={case.get('usr_msg_num')}")
        if len(first_res_zero) > 3:
            print(f"  ... and {len(first_res_zero) - 3} more")
            
        # First response time = -1 cases (invalid)
        first_res_invalid = [(channel_url, case) for channel_url, case in self.channel_cases if case.get("first_res_time") == -1]
        print(f"\nFirst Response Time = -1 (invalid): {len(first_res_invalid)} cases")
        for channel_url, case in first_res_invalid[:3]:  # Show first 3
            channel_display = self._format_channel_for_display(channel_url)
            print(f"  - Case {case.get('case_id', 'unknown')} (Channel: {channel_display}): "
                  f"status={case.get('status')}, usr_msg={case.get('usr_msg_num')}, reason: probably no user messages")
        if len(first_res_invalid) > 3:
            print(f"  ... and {len(first_res_invalid) - 3} more")
        
    def calculate_resolution_metrics(self) -> Dict[str, Any]:
        """Calculate first contact resolution metrics"""
        total_cases = len(self.all_cases)
        resolved_cases = sum(1 for case in self.all_cases if case.get("first_contact_resolution") == 1)
        unresolved_cases = total_cases - resolved_cases
        
        return {
            "valid_cases": total_cases,
            "total_cases": total_cases,
            "validity_rate": 100.0,
            "resolved_cases": resolved_cases,
            "unresolved_cases": unresolved_cases,
            "resolution_rate_percent": resolved_cases / total_cases * 100.0 if total_cases > 0 else 0.0
        }
        
    def compare_metrics(self, calculated: Dict[str, Any], official: Dict[str, Any], path: str = "") -> List[str]:
        """Compare calculated vs official metrics, return list of differences"""
        differences = []
        
        if isinstance(calculated, dict) and isinstance(official, dict):
            for key in calculated:
                if key not in official:
                    differences.append(f"{path}.{key}: Missing in official stats")
                    continue
                    
                if isinstance(calculated[key], (dict, list)):
                    differences.extend(self.compare_metrics(calculated[key], official[key], f"{path}.{key}"))
                else:
                    # Compare numeric values with tolerance
                    if isinstance(calculated[key], (int, float)) and isinstance(official[key], (int, float)):
                        # Use relative tolerance for better comparison of floating point values
                        tolerance = max(abs(official[key]) * 0.01, 0.1)  # 1% relative or 0.1 absolute minimum
                        if abs(calculated[key] - official[key]) > tolerance:
                            differences.append(f"{path}.{key}: Calculated={calculated[key]}, Official={official[key]} (diff: {abs(calculated[key] - official[key]):.3f})")
                    elif calculated[key] != official[key]:
                        differences.append(f"{path}.{key}: Calculated={calculated[key]}, Official={official[key]}")
                        
        return differences
        
    def validate_statistics(self) -> bool:
        """Main validation function"""
        print("Loading case files...")
        self.load_case_files()
        
        print("Loading official statistics...")
        self.load_official_statistics()
        
        print("Calculating performance metrics...")
        calculated_performance = self.calculate_performance_metrics()
        
        print("Calculating resolution metrics...")
        calculated_resolution = self.calculate_resolution_metrics()
        
        # Compare with official statistics
        print("\n=== VALIDATION RESULTS ===")
        
        all_differences = []
        
        # Compare performance metrics
        official_performance = self.official_stats.get("performance_metrics", {})
        perf_diffs = self.compare_metrics(calculated_performance, official_performance, "performance_metrics")
        all_differences.extend(perf_diffs)
        
        # Compare resolution metrics
        official_resolution = self.official_stats.get("first_contact_resolution", {})
        res_diffs = self.compare_metrics(calculated_resolution, official_resolution, "first_contact_resolution")
        all_differences.extend(res_diffs)
        
        # Print results
        if not all_differences:
            print("✅ All statistics match! Validation PASSED")
            return True
        else:
            print(f"❌ Found {len(all_differences)} differences:")
            for diff in all_differences:
                print(f"  - {diff}")
            print("\nValidation FAILED")
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Validate session statistics')
    parser.add_argument('session_dir', help='Session output directory path')
    args = parser.parse_args()
    
    try:
        validator = SessionStatisticsValidator(args.session_dir)
        success = validator.validate_statistics()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()