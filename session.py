#!/usr/bin/env python3
"""
Session management module for customer support message segmentation pipeline.

This module contains the Session class that orchestrates the complete processing pipeline:
- File data processing across all channels
- Individual channel segmentation and case generation  
- Cross-channel statistics generation
- Session-level output management
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter
import pandas as pd
import numpy as np

# Local imports
from channel import Channel
from llm_client import LLMClient
from utils import Utils


class Session:
    """
    Session orchestrator for the complete message processing pipeline.
    
    Manages cross-channel operations, file processing, and session-level outputs.
    Encapsulates all business logic that was previously scattered across main.py and FileProcessor.
    """
    
    def __init__(self, 
                 input_file: str,
                 output_dir: str = 'out', 
                 chunk_size: int = 60,
                 overlap: int = 10,
                 model: str = 'gpt-5',
                 session_name: Optional[str] = None,
                 enable_review: bool = False,
                 enable_classification: bool = True):
        """
        Initialize session with explicit parameters.
        
        Args:
            input_file: Path to input CSV file (required)
            output_dir: Base output directory (default: 'out')
            chunk_size: Chunk size for segmentation (default: 80)
            overlap: Overlap size between chunks (default: 20)
            model: LLM model name (default: 'gpt-5')
            session_name: Optional session name (auto-generated if None)
            enable_review: Enable case review flag (default: False)
            enable_classification: Enable case classification flag (default: True)
        """
        # Pipeline configuration
        self.input_file = input_file
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = model
        self.enable_review = enable_review
        self.enable_classification = enable_classification
        
        # Session identification and output management
        self.session_name = session_name or datetime.now().strftime("%y%m%d_%H%M%S")
        self.output_folder = os.path.join(output_dir, f"session_{self.session_name}")
        
        # Data processing state
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        
        # Cross-channel collections
        self.channels: List[Channel] = []
        self.channel_data_list: List[Dict[str, Any]] = []
    
    def run(self) -> None:
        """
        Execute the complete processing pipeline.
        
        Pipeline stages:
        1. File data processing (cross-channel)
        2. Session folder setup
        3. Individual channel processing
        4. Cross-channel statistics generation
        """
        print(f"🚀 Starting pipeline session: {self.session_name}")
        
        try:
            # Stage 1: Process file data across all channels
            if not self.process_file_data():
                print("Error: File processing failed")
                exit(1)
            
            # Stage 2: Create session folder structure
            self.create_session_folder()
            
            # Stage 3: Initialize LLM client and process channels
            self.process_channels()
            
            # Stage 4: Generate cross-channel statistics
            self.generate_statistics()
            
            print(f"\n✅ Pipeline complete!")
            
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
    
    def process_file_data(self) -> bool:
        """
        Process CSV file data and prepare channel data list.
        
        All file processing logic is contained within this method.
        
        Returns:
            True if processing successful, False otherwise
        """
        print("Starting file processing...")
        
        # 1. Load CSV data into DataFrame
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"        Loaded {len(self.df)} messages from {self.input_file}")
        except Exception as e:
            print(f"Error loading file {self.input_file}: {e}")
            return False
        
        # 2. Filter out rows where Deleted = True
        if 'Deleted' in self.df.columns:
            original_count = len(self.df)
            self.df = self.df[self.df['Deleted'] != True].reset_index(drop=True)
            filtered_count = original_count - len(self.df)
            print(f"        Filtered out {filtered_count} deleted rows ({len(self.df)} remaining)")
        else:
            print("No 'Deleted' column found, skipping deletion filter")
        
        # 3. Add role column based on Sender ID pattern
        if 'role' not in self.df.columns:
            self.df['role'] = self.df['Sender ID'].apply(
                lambda x: 'customer_service' if str(x).startswith('psops') else 'user'
            )
            print(f"        Added role column: {self.df['role'].value_counts().to_dict()}")
        else:
            print("Role column already exists, skipping...")
        
        # 4. Parse Created Time to timezone-aware UTC format
        def parse_to_utc(timestamp_str):
            try:
                # Parse with timezone info and convert to UTC
                dt = pd.to_datetime(timestamp_str, utc=True)
                return dt
            except Exception:
                return pd.NaT
        
        self.df['Created Time'] = self.df['Created Time'].apply(parse_to_utc)
        print(f"        Processed time columns, converted {len(self.df)} timestamps to UTC")
        
        # 5. Sort data by Channel URL, Created Time, then Message ID
        self.df = self.df.sort_values([
            'Channel URL',
            'Created Time', 
            'Message ID'
        ]).reset_index(drop=True)
        print(f"        Sorted data by Channel URL, Created Time, and Message ID")
        
        # 6. Add msg_ch_idx column (0..N-1 for each Channel URL group)
        self.df['msg_ch_idx'] = self.df.groupby('Channel URL').cumcount()
        print(f"        Added msg_ch_idx column for {self.df['Channel URL'].nunique()} channels")
        
        # 7. Generate clean DataFrame with essential columns
        essential_columns = [
            'Created Time', 'Sender ID', 'Message', 'Channel URL',
            'role', 'msg_ch_idx', 'Message ID'
        ]
        available_columns = [col for col in essential_columns if col in self.df.columns]
        self.df_clean = self.df[available_columns].copy()
        print(f"        Created clean DataFrame with {len(available_columns)} columns: {available_columns}")
        
        print(f"        Processed {len(self.df_clean)} messages across {self.df_clean['Channel URL'].nunique()} channels")
        
        # 8. Group by channel and create channel data list
        self.channel_data_list = []
        for channel_url in self.df_clean['Channel URL'].unique():
            channel_df = self.df_clean[self.df_clean['Channel URL'] == channel_url].copy()
            # Reset msg_ch_idx to ensure it starts from 0 for each channel
            channel_df['msg_ch_idx'] = range(len(channel_df))
            
            self.channel_data_list.append({
                "channel_url": channel_url,
                "dataframe": channel_df
            })
            
            print(f"                Channel: {Utils.format_channel_for_display(channel_url)} - {len(channel_df)} messages")
        
        return True
    
    def create_session_folder(self) -> None:
        """Create session output folder structure."""
        os.makedirs(self.output_folder, exist_ok=True)
    
    def process_channels(self) -> None:
        """
        Process each channel individually and collect results.
        
        Migrated from main.py channel processing loop.
        """
        # Initialize LLM client
        llm_client = LLMClient(model=self.model)
        print(f"LLM Client initialized with model: {self.model}")
        
        # Process each channel
        for channel_idx, channel_data in enumerate(self.channel_data_list):
            channel_url = channel_data["channel_url"]
            channel_df = channel_data["dataframe"]
            
            print(f"🔄 Channel {channel_idx + 1}/{len(self.channel_data_list)}: {Utils.format_channel_for_display(channel_url)} ({len(channel_df)} messages)")
            
            # Check if channel results already exist
            channel_name = Utils.format_channel_for_display(channel_url)
            channel_cases_file = os.path.join(self.output_folder, f"cases_{channel_name}.json")
            
            # Create Channel instance
            channel = Channel(channel_df, channel_url, self.session_name, self.chunk_size, self.overlap)
            
            if os.path.exists(channel_cases_file):
                print(f"        ⏭️  Loading existing results from file")
                channel.build_cases_via_file(self.output_dir)
            else:
                # Process channel with full pipeline
                channel.build_cases_simple(llm_client)
                
                # Save channel results
                print(f"    💾 Saving results...")
                try:
                    channel.save_results_to_json(self.output_dir)
                    channel.save_results_to_csv(self.output_dir)
                    print(f"    ✅ Results saved successfully")
                except Exception as save_error:
                    print(f"        ❌ Error saving results: {str(save_error)}")
                    print(f"        Processing completed but save failed - continuing...")
            
            # Collect channel for cross-channel operations
            self.channels.append(channel)
        
        # Summary
        print(f"\n✅ Pipeline processing complete!")
        print(f"Processed {len(self.channel_data_list)} channels")
        print(f"Results saved to timestamped session folders in output directory")
        print(f"Each session contains JSON and CSV files for successfully saved channels")
    
    def generate_statistics(self) -> None:
        """
        Generate comprehensive statistics across all channels.
        
        Migrated from main.py statistics generation logic.
        """
        try:
            print(f"\n📊 Generating comprehensive statistics...")
            
            # Collect all cases from all channels
            all_cases = []
            for channel in self.channels:
                if hasattr(channel, 'cases') and channel.cases:
                    all_cases.extend(channel.cases)
            
            if all_cases:
                # Calculate statistics
                stats_result = self._calculate_comprehensive_stats(all_cases)
                
                # Print summary report to console
                self._print_summary_report(stats_result)
                
                # Save statistics to file
                print(f"    💾 Saving statistics...")
                self._save_stats_to_file(stats_result)
                print(f"    ✅ Statistics analysis complete")
            else:
                print(f"    ⚠️  No cases found for statistical analysis")
                
        except Exception as stats_error:
            print(f"    ❌ Error during statistics generation: {str(stats_error)}")
            print(f"    Pipeline completed successfully, but statistics failed")
    
    def _calculate_comprehensive_stats(self, all_cases: List) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for all cases.
        
        Args:
            all_cases: List of all Case objects
            
        Returns:
            Dictionary containing all statistical analyses
        """
        print(f"📊 Calculating comprehensive statistics for {len(all_cases)} cases...")
        
        # Basic counts
        total_cases = len(all_cases)
        
        # Calculate category statistics
        category_stats = self._calculate_category_stats(all_cases)
        
        # Calculate performance metrics statistics
        metrics_stats = self._calculate_metrics_stats(all_cases)
        
        # Calculate first contact resolution statistics
        fcr_stats = self._calculate_fcr_stats(all_cases)
        
        # Compile comprehensive results
        stats_result = {
            "summary": {
                "total_cases": total_cases,
                "analysis_timestamp": pd.Timestamp.now().isoformat()
            },
            "category_distribution": category_stats,
            "performance_metrics": metrics_stats,
            "first_contact_resolution": fcr_stats
        }
        
        return stats_result
    
    def _calculate_category_stats(self, all_cases: List) -> Dict[str, Any]:
        """Calculate category distribution statistics."""
        print("    📈 Analyzing category distributions...")
        
        # Main category distribution
        main_categories = [case.main_category for case in all_cases]
        main_category_counts = Counter(main_categories)
        main_category_percentages = {
            category: (count / len(main_categories)) * 100 
            for category, count in main_category_counts.items()
        }
        
        # Sub category distribution
        sub_categories = [case.sub_category for case in all_cases]
        sub_category_counts = Counter(sub_categories)
        sub_category_percentages = {
            category: (count / len(sub_categories)) * 100 
            for category, count in sub_category_counts.items()
        }
        
        # Build hierarchical mapping
        main_to_sub_mapping = {}
        for case in all_cases:
            main_cat = case.main_category
            sub_cat = case.sub_category
            
            if main_cat not in main_to_sub_mapping:
                main_to_sub_mapping[main_cat] = {}
            
            if sub_cat not in main_to_sub_mapping[main_cat]:
                main_to_sub_mapping[main_cat][sub_cat] = 0
            
            main_to_sub_mapping[main_cat][sub_cat] += 1
        
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
            },
            "hierarchical": main_to_sub_mapping
        }
    
    def _calculate_metrics_stats(self, all_cases: List) -> Dict[str, Any]:
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
            for case in all_cases:
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
                    "total_cases": len(all_cases),
                    "validity_rate": (len(values) / len(all_cases)) * 100,
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
                    "total_cases": len(all_cases),
                    "validity_rate": 0.0,
                    "percentiles": None,
                    "basic_stats": None
                }
        
        return metrics_stats
    
    def _calculate_fcr_stats(self, all_cases: List) -> Dict[str, Any]:
        """Calculate first contact resolution statistics."""
        print("    ✅ Analyzing first contact resolution rates...")
        
        # Extract valid FCR values (filter out -1)
        fcr_values = []
        for case in all_cases:
            if case.first_contact_resolution != -1:
                fcr_values.append(case.first_contact_resolution)
        
        if fcr_values:
            resolved_count = sum(1 for value in fcr_values if value == 1)
            resolution_rate = (resolved_count / len(fcr_values)) * 100
            
            return {
                "valid_cases": len(fcr_values),
                "total_cases": len(all_cases),
                "validity_rate": (len(fcr_values) / len(all_cases)) * 100,
                "resolved_cases": resolved_count,
                "unresolved_cases": len(fcr_values) - resolved_count,
                "resolution_rate_percent": round(resolution_rate, 2)
            }
        else:
            return {
                "valid_cases": 0,
                "total_cases": len(all_cases),
                "validity_rate": 0.0,
                "resolved_cases": 0,
                "unresolved_cases": 0,
                "resolution_rate_percent": 0.0
            }
    
    def _print_summary_report(self, stats_result: Dict[str, Any]) -> None:
        """Print a formatted summary report to console."""
        print(f"\n📊 CASE STATISTICS SUMMARY")
        print(f"=" * 50)
        
        # Basic summary
        summary = stats_result["summary"]
        print(f"Total Cases Analyzed: {summary['total_cases']}")
        
        # Category distribution
        print(f"\n📈 CATEGORY DISTRIBUTION")
        print(f"-" * 30)
        main_cat = stats_result["category_distribution"]["main_category"]
        print(f"Main Categories:")
        for category, percentage in main_cat["percentages"].items():
            count = main_cat["counts"][category]
            print(f"  {category}: {count} cases ({percentage:.1f}%)")
        
        # Display hierarchical sub categories
        print(f"\nHierarchical Category Breakdown:")
        main_cat = stats_result["category_distribution"]["main_category"]
        main_to_sub_mapping = stats_result["category_distribution"]["hierarchical"]
        total_cases = stats_result["summary"]["total_cases"]
        
        # Sort main categories by count (descending)
        sorted_main_cats = sorted(main_cat["counts"].items(), key=lambda x: x[1], reverse=True)
        
        for main_category, main_count in sorted_main_cats:
            main_percentage = main_cat["percentages"][main_category]
            print(f"  {main_category}: {main_count} cases ({main_percentage:.1f}%)")
            
            if main_category in main_to_sub_mapping:
                # Sort sub categories by count (descending)
                sorted_sub_cats = sorted(main_to_sub_mapping[main_category].items(), key=lambda x: x[1], reverse=True)
                
                for sub_category, sub_count in sorted_sub_cats:
                    sub_percentage = (sub_count / total_cases) * 100
                    print(f"    ├─ {sub_category}: {sub_count} cases ({sub_percentage:.1f}%)")
        
        # Performance metrics
        print(f"\n⏱️  PERFORMANCE METRICS")
        print(f"-" * 30)
        print(f"Metric Definitions:")
        print(f"  • handle_time: Time between first and last message in minutes")
        print(f"  • first_res_time: Support response time in minutes")
        print(f"  • usr_msg_num: Count of user messages")
        print(f"  (Value of -1 indicates not processed/invalid data)")
        print()
        metrics = stats_result["performance_metrics"]
        
        for metric_name, data in metrics.items():
            print(f"{metric_name.replace('_', ' ').title()}:")
            if data["percentiles"]:
                print(f"  Valid Cases: {data['valid_cases']}/{data['total_cases']} ({data['validity_rate']:.1f}%)")
                p = data["percentiles"]
                print(f"  P5: {p['P5']}, P50: {p['P50']}, P95: {p['P95']}")
                b = data["basic_stats"]
                print(f"  Min: {b['min']}, Max: {b['max']}, Mean: {b['mean']}, Std: {b['std']}")
            else:
                print(f"  No valid data available")
        
        # First contact resolution
        print(f"\n✅ FIRST CONTACT RESOLUTION")
        print(f"-" * 30)
        fcr = stats_result["first_contact_resolution"]
        if fcr["valid_cases"] > 0:
            print(f"Valid Cases: {fcr['valid_cases']}/{fcr['total_cases']} ({fcr['validity_rate']:.1f}%)")
            print(f"Resolution Rate: {fcr['resolution_rate_percent']:.1f}% ({fcr['resolved_cases']}/{fcr['valid_cases']} cases)")
        else:
            print(f"No valid FCR data available")
        
        print(f"\n" + "=" * 50)
    
    def _save_stats_to_file(self, stats_result: Dict[str, Any]) -> None:
        """Save comprehensive statistics to JSON file."""
        session_folder = os.path.join(self.output_dir, f"session_{self.session_name}")
        os.makedirs(session_folder, exist_ok=True)
        
        stats_file = os.path.join(session_folder, f"statistics_{self.session_name}.json")
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_result, f, indent=2, ensure_ascii=False)
            print(f"            Statistics saved to: {stats_file}")
        except IOError as e:
            print(f"            ❌ Error saving statistics file: {e}")
            raise