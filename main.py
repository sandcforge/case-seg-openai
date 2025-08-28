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
    
    # Create and run session with explicit parameters
    session = Session(
        input_file=args.input,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        model=args.model,
        session_name=args.session,
        enable_review=args.enable_review,
        enable_classification=args.enable_classification
    )
    session.run()


if __name__ == '__main__':
    main()