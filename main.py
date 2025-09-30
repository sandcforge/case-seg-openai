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
        '--force-classification',
        action='store_true',
        help='Force classification re-run when loading cases from existing files'
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
    
    # Create and run session with explicit parameters
    session = Session(
        input_file=args.input,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model=args.model,
        session_name=args.session,
        enable_review=args.enable_review,
        force_classification=args.force_classification
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