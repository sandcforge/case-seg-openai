#!/usr/bin/env python3
"""
Customer support message segmentation system.

This module implements a two-stage pipeline for processing customer support messages:
1. FileProcessor: Loads CSV data and performs preprocessing (role assignment, time parsing, sorting)
2. Channel: Segments processed data into chunks for LLM analysis

Usage:
    python main.py [--input INPUT] [--output-dir OUTPUT_DIR] [--chunk-size SIZE]

Example:
    python main.py --chunk-size 80
"""

import os
import argparse
import pandas as pd # type: ignore
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv # type: ignore

# Local imports
from file_processor import FileProcessor
from channel import Channel
from llm_client import LLMClient
from utils import Utils

# Load environment variables
load_dotenv()


def main() -> None:
    """Main entry point for the message processing pipeline"""
    parser = argparse.ArgumentParser(
        description='Process customer support messages into chunks for LLM analysis'
    )
    parser.add_argument(
        '--input', '-i',
        default='assets/support_messages_andy.csv',
        help='Input CSV file path (default: assets/support_messages_andy.csv)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='out',
        help='Output directory (default: out)'
    )
    parser.add_argument(
        '--chunk-size', '-c',
        type=int,
        default=80,
        help='Chunk size for segmentation (default: 80)'
    )
    parser.add_argument(
        '--overlap', '-l',
        type=int,
        default=20,
        help='Overlap size between chunks, must be < chunk_size/3 (default: 20)'
    )
    parser.add_argument(
        '--model', '-m',
        default='gpt-5',
        help='LLM model to use for tail summary generation (default: claude-sonnet-4-20250514)'
    )
    # API keys are now automatically determined from environment variables based on model prefix
    parser.add_argument(
        '--test-case-segment',
        action='store_true',
        help='Test case segmentation on first chunk only'
    )
    parser.add_argument(
        '--enable-review',
        action='store_true',
        help='Enable LLM-based case review for regions between chunks'
    )
    parser.add_argument(
        '--enable-classification',
        action='store_true',
        default=True,
        help='Enable LLM-based case classification for all cases (default: True)'
    )
    parser.add_argument(
        '--session', '-s',
        help='Session name for output organization (default: auto-generated timestamp)'
    )
    
    args = parser.parse_args()
    
    # Generate session name for this entire pipeline run
    session = args.session or datetime.now().strftime("%y%m%d_%H%M%S")
    print(f"🚀 Starting pipeline session: {session}")
    
    try:
        # Stage 1: File Processing
        processor = FileProcessor(args.input, args.output_dir)
        channel_data_list = processor.process()
        
        if not channel_data_list:
            print("Error: File processing failed")
            exit(1)
        
        print(f"Found {len(channel_data_list)} channels to process")
        
        # Stage 3: Initialize LLM Client for case segmentation
        llm_client = LLMClient(model=args.model)
        print(f"LLM Client initialized with model: {args.model}")
        
        # Process each channel separately
        
        # Create session folder
        session_folder = os.path.join(args.output_dir, f"session_{session}")
        os.makedirs(session_folder, exist_ok=True)
        
        for channel_idx, channel_data in enumerate(channel_data_list):
            channel_url = channel_data["channel_url"]
            channel_df = channel_data["dataframe"]
            
            print(f"🔄 Channel {channel_idx + 1}/{len(channel_data_list)}: {Utils.format_channel_for_display(channel_url)} ({len(channel_df)} messages)")
            
            # Check if channel results already exist
            channel_name = Utils.format_channel_for_display(channel_url)
            channel_cases_file = os.path.join(session_folder, f"cases_{channel_name}.json")
            
            # Stage 2: Channel Segmentation for this channel
            one_ch = Channel(channel_df, channel_url, session, args.chunk_size, args.overlap)

            if os.path.exists(channel_cases_file):
                print(f"        ⏭️  Loading existing results from file")
                # Load existing results using the new method
                one_ch.build_cases_via_file(args.output_dir)

            else:
            
                # Process this channel with full pipeline and save results immediately
                one_ch.build_cases_simple(llm_client)
                # Save this channel's results independently with error protection
                print(f"        💾 Saving results...")
                try:
                    # Protected file save operations
                    one_ch.save_results_to_json(args.output_dir)
                    one_ch.save_results_to_csv(args.output_dir)
                    print(f"        ✅ Results saved successfully")
                    
                except Exception as save_error:
                    print(f"        ❌ Error saving results: {str(save_error)}")
                    print(f"        Processing completed but save failed - continuing...")
                    # Continue processing other channels even if this one fails to save
            
        
        
        # Summary for all channels
        print(f"\n✅ Pipeline processing complete!")
        print(f"Processed {len(channel_data_list)} channels")
        print(f"Results saved to timestamped session folders in output directory")
        print(f"Each session contains JSON and CSV files for successfully saved channels")
        
        print(f"\n✅ Pipeline complete!")
        
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == '__main__':
    main()