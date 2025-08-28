#!/usr/bin/env python3
"""
Statistics module for case analysis and reporting.

This module provides functionality to:
- Calculate percentiles (P5, P50, P95) for performance metrics
- Analyze category distributions
- Filter invalid data (-1 values)
- Generate comprehensive statistics reports
"""

import json
import os
from typing import List, Dict, Any
from collections import Counter
import numpy as np
import pandas as pd
from case import Case


class CaseStatistics:
    """
    Comprehensive statistics calculator for case analysis across all channels.
    
    Filters out invalid data (values of -1) and provides detailed analytics
    on case performance metrics and category distributions.
    """
    
    def __init__(self, all_cases: List[Case]):
        """
        Initialize statistics calculator with all cases from all channels.
        
        Args:
            all_cases: List of all Case objects across all processed channels
        """
        self.all_cases = all_cases
        self.stats_result: Dict[str, Any] = {}
    
    def calculate_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for all cases.
        
        Returns:
            Dictionary containing all statistical analyses
        """
        print(f"📊 Calculating comprehensive statistics for {len(self.all_cases)} cases...")
        
        # Basic counts
        total_cases = len(self.all_cases)
        
        # Calculate category statistics
        category_stats = self._calculate_category_stats()
        
        # Calculate performance metrics statistics
        metrics_stats = self._calculate_metrics_stats()
        
        # Calculate first contact resolution statistics
        fcr_stats = self._calculate_fcr_stats()
        
        # Compile comprehensive results
        self.stats_result = {
            "summary": {
                "total_cases": total_cases,
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            },
            "category_distribution": category_stats,
            "performance_metrics": metrics_stats,
            "first_contact_resolution": fcr_stats
        }
        
        return self.stats_result
    
    def _calculate_category_stats(self) -> Dict[str, Any]:
        """Calculate category distribution statistics."""
        print("    📈 Analyzing category distributions...")
        
        # Main category distribution
        main_categories = [case.main_category for case in self.all_cases]
        main_category_counts = Counter(main_categories)
        main_category_percentages = {
            category: (count / len(main_categories)) * 100 
            for category, count in main_category_counts.items()
        }
        
        # Sub category distribution
        sub_categories = [case.sub_category for case in self.all_cases]
        sub_category_counts = Counter(sub_categories)
        sub_category_percentages = {
            category: (count / len(sub_categories)) * 100 
            for category, count in sub_category_counts.items()
        }
        
        return {
            "main_category": {
                "counts": dict(main_category_counts),
                "percentages": main_category_percentages,
                "total_analyzed": len(main_categories)
            },
            "sub_category": {
                "counts": dict(sub_category_counts),
                "percentages": sub_category_percentages,
                "total_analyzed": len(sub_categories)
            }
        }
    
    def _calculate_metrics_stats(self) -> Dict[str, Any]:
        """Calculate performance metrics statistics with invalid data filtering."""
        print("    ⏱️  Analyzing performance metrics...")
        
        metrics_stats = {}
        
        # Define metrics to analyze
        metrics = {
            'handle_time': 'handle_time',
            'first_res_time': 'first_res_time', 
            'usr_msg_num': 'usr_msg_num'
        }
        
        for metric_name, case_attr in metrics.items():
            # Extract valid values (filter out -1)
            values = []
            for case in self.all_cases:
                value = getattr(case, case_attr)
                if value != -1:  # Only include valid data
                    values.append(value)
            
            if values:
                # Calculate percentiles
                p5 = np.percentile(values, 5)
                p50 = np.percentile(values, 50)  # median
                p95 = np.percentile(values, 95)
                
                metrics_stats[metric_name] = {
                    "valid_cases": len(values),
                    "total_cases": len(self.all_cases),
                    "validity_rate": (len(values) / len(self.all_cases)) * 100,
                    "percentiles": {
                        "P5": round(p5, 2),
                        "P50": round(p50, 2),
                        "P95": round(p95, 2)
                    },
                    "basic_stats": {
                        "min": round(min(values), 2),
                        "max": round(max(values), 2),
                        "mean": round(np.mean(values), 2),
                        "std": round(np.std(values), 2)
                    }
                }
            else:
                metrics_stats[metric_name] = {
                    "valid_cases": 0,
                    "total_cases": len(self.all_cases),
                    "validity_rate": 0.0,
                    "percentiles": None,
                    "basic_stats": None
                }
        
        return metrics_stats
    
    def _calculate_fcr_stats(self) -> Dict[str, Any]:
        """Calculate first contact resolution statistics."""
        print("    ✅ Analyzing first contact resolution rates...")
        
        # Extract valid FCR values (filter out -1)
        fcr_values = []
        for case in self.all_cases:
            if case.first_contact_resolution != -1:
                fcr_values.append(case.first_contact_resolution)
        
        if fcr_values:
            resolved_count = sum(1 for value in fcr_values if value == 1)
            resolution_rate = (resolved_count / len(fcr_values)) * 100
            
            return {
                "valid_cases": len(fcr_values),
                "total_cases": len(self.all_cases),
                "validity_rate": (len(fcr_values) / len(self.all_cases)) * 100,
                "resolved_cases": resolved_count,
                "unresolved_cases": len(fcr_values) - resolved_count,
                "resolution_rate_percent": round(resolution_rate, 2)
            }
        else:
            return {
                "valid_cases": 0,
                "total_cases": len(self.all_cases),
                "validity_rate": 0.0,
                "resolved_cases": 0,
                "unresolved_cases": 0,
                "resolution_rate_percent": 0.0
            }
    
    def print_summary_report(self) -> None:
        """Print a formatted summary report to console."""
        if not self.stats_result:
            print("❌ No statistics available. Run calculate_comprehensive_stats() first.")
            return
        
        print(f"\n📊 CASE STATISTICS SUMMARY")
        print(f"=" * 50)
        
        # Basic summary
        summary = self.stats_result["summary"]
        print(f"Total Cases Analyzed: {summary['total_cases']}")
        
        # Category distribution
        print(f"\n📈 CATEGORY DISTRIBUTION")
        print(f"-" * 30)
        main_cat = self.stats_result["category_distribution"]["main_category"]
        print(f"Main Categories:")
        for category, percentage in main_cat["percentages"].items():
            count = main_cat["counts"][category]
            print(f"  {category}: {count} cases ({percentage:.1f}%)")
        
        # Performance metrics
        print(f"\n⏱️  PERFORMANCE METRICS")
        print(f"-" * 30)
        metrics = self.stats_result["performance_metrics"]
        
        for metric_name, data in metrics.items():
            print(f"{metric_name.replace('_', ' ').title()}:")
            if data["percentiles"]:
                print(f"  Valid Cases: {data['valid_cases']}/{data['total_cases']} ({data['validity_rate']:.1f}%)")
                p = data["percentiles"]
                print(f"  P5: {p['P5']}, P50: {p['P50']}, P95: {p['P95']}")
            else:
                print(f"  No valid data available")
        
        # First contact resolution
        print(f"\n✅ FIRST CONTACT RESOLUTION")
        print(f"-" * 30)
        fcr = self.stats_result["first_contact_resolution"]
        if fcr["valid_cases"] > 0:
            print(f"Valid Cases: {fcr['valid_cases']}/{fcr['total_cases']} ({fcr['validity_rate']:.1f}%)")
            print(f"Resolution Rate: {fcr['resolution_rate_percent']:.1f}% ({fcr['resolved_cases']}/{fcr['valid_cases']} cases)")
        else:
            print(f"No valid FCR data available")
        
        print(f"\n" + "=" * 50)
    
    def save_stats_to_file(self, output_dir: str, session: str) -> str:
        """
        Save comprehensive statistics to JSON file.
        
        Args:
            output_dir: Output directory path
            session: Session identifier
            
        Returns:
            Path to saved statistics file
        """
        if not self.stats_result:
            raise ValueError("No statistics available. Run calculate_comprehensive_stats() first.")
        
        session_folder = os.path.join(output_dir, f"session_{session}")
        os.makedirs(session_folder, exist_ok=True)
        
        stats_file = os.path.join(session_folder, f"statistics_{session}.json")
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats_result, f, indent=2, ensure_ascii=False)
            print(f"            Statistics saved to: {stats_file}")
            return stats_file
        except IOError as e:
            print(f"            ❌ Error saving statistics file: {e}")
            raise


def collect_all_cases_from_channels(channels: List) -> List[Case]:
    """
    Collect all Case objects from multiple Channel instances.
    
    Args:
        channels: List of Channel instances
        
    Returns:
        List of all Case objects across all channels
    """
    all_cases = []
    for channel in channels:
        if hasattr(channel, 'cases') and channel.cases:
            all_cases.extend(channel.cases)
    return all_cases