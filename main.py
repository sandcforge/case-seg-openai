#!/usr/bin/env python3
"""
Customer support message segmentation system.

This module provides the command-line interface for the Session-based processing pipeline:
1. Session: Orchestrates file processing, channel segmentation, and statistics generation
2. All business logic is encapsulated in the Session class

Usage:
    python main.py [--input INPUT] [--output-dir OUTPUT_DIR] [--chunk-size SIZE]

Example:
    python main.py --chunk-size 80 --session my_analysis
"""

import argparse
from dotenv import load_dotenv # type: ignore

# Local imports
from session import Session
from utils import Utils

# Load environment variables
load_dotenv()


def generate_firebase_token() -> None:
    """Generate Firebase ID token using credentials from environment variables"""
    try:
        id_token = Utils.get_aloy_token()
        print(f"✅ Firebase ID Token\n{id_token}")
    except (ValueError, RuntimeError) as e:
        print(f"❌ Error: {e}")


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
        default=60,
        help='Chunk size for segmentation (default: 80)'
    )
    parser.add_argument(
        '--overlap', '-l',
        type=int,
        default=10,
        help='Overlap size between chunks, must be < chunk_size/3 (default: 20)'
    )
    parser.add_argument(
        '--model', '-m',
        default='gpt-5',
        help='LLM model to use for tail summary generation (default: claude-sonnet-4-20250514)'
    )
    # API keys are now automatically determined from environment variables based on model prefix
    parser.add_argument(
        '--enable-review',
        action='store_true',
        help='Enable LLM-based case review for regions between chunks'
    )
    parser.add_argument(
        '--enable-classification',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable classification when loading cases from existing files (default: True)'
    )
    parser.add_argument(
        '--enable-vision-processing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable vision processing for FILE type messages (default: True)'
    )
    parser.add_argument(
        '--enable-find-sop',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable SOP finding when running QA function (default: True)'
    )
    parser.add_argument(
        '--session', '-s',
        help='Session name for output organization (default: auto-generated timestamp)'
    )
    parser.add_argument(
        '--function', '-f',
        choices=['mbr', 'qa', 'token'],
        default='mbr',
        help='Choose which session function to run (choices: mbr, qa, token)'
    )
    
    args = parser.parse_args()

    # Print configuration parameters
    print("=" * 60)
    print("Configuration Parameters")
    print("=" * 60)
    print(f"  Input File:              {args.input}")
    print(f"  Output Directory:        {args.output_dir}")
    print(f"  Chunk Size:              {args.chunk_size}")
    print(f"  Overlap:                 {args.overlap}")
    print(f"  Model:                   {args.model}")
    print(f"  Session Name:            {args.session or 'auto-generated'}")
    print(f"  Function:                {args.function}")
    print(f"  Enable Review:           {args.enable_review}")
    print(f"  Enable Classification:   {args.enable_classification}")
    print(f"  Enable Vision Processing: {args.enable_vision_processing}")
    print(f"  Enable Find SOP:         {args.enable_find_sop}")
    print("=" * 60)
    print()

    # Create and run session with explicit parameters
    session = Session(
        input_file=args.input,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model=args.model,
        session_name=args.session,
        enable_review=args.enable_review,
        enable_classification=args.enable_classification,
        enable_vision_processing=args.enable_vision_processing,
        enable_find_sop=args.enable_find_sop
    )
    # Execute selected function
    if args.function == 'mbr':
        session.cs_mbr()
    elif args.function == 'qa':
        session.cs_qa()
    elif args.function == 'token':
        generate_firebase_token()


if __name__ == '__main__':
    main()