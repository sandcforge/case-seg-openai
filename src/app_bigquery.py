#!/usr/bin/env python3
"""
BigQuery-based automatic channel processing application.

This application:
1. Scans all channels in BigQuery for unanalyzed messages
2. Checks trigger conditions (configurable chunk_size or 7+ days old)
3. Processes triggered channels: fetch → analyze → segment → save to BigQuery
4. Supports dry-run mode (saves CSV/JSON instead of BigQuery)

Usage:
    # Normal mode (save to BigQuery)
    python app_bigquery.py

    # Dry-run mode (save CSV/JSON for inspection)
    python app_bigquery.py --dry-run

    # Custom chunk size
    python app_bigquery.py --chunk-size 100

    # Custom model
    python app_bigquery.py --model gpt-4
"""

import argparse
from datetime import datetime
from dotenv import load_dotenv

# Local imports - compatible with both direct execution and module execution
try:
    from .utils import Utils
    from .channel import Channel
    from .llm_client import LLMClient
except ImportError:
    from utils import Utils
    from channel import Channel
    from llm_client import LLMClient


def main():
    """Main entry point for BigQuery-based channel processing."""
    # Load environment variables
    load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='BigQuery-based automatic channel processing')

    parser.add_argument('--dry-run', action='store_true',
                       help='Dry-run mode: save CSV/JSON instead of BigQuery')
    parser.add_argument('--channel-urls', type=str, nargs='+',
                       help='Specific channel URLs to process (space-separated). If provided, skips global scanning.')
    parser.add_argument('--chunk-size', type=int, default=80,
                       help='Chunk size for segmentation and trigger threshold (default: 80)')
    parser.add_argument('--model', type=str, default='gpt-5',
                       help='LLM model name (default: gpt-5)')
    parser.add_argument(
        '--enable-vision-processing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable vision processing for FILE type messages (default: True)'
    )
    parser.add_argument(
        '--enable-classification',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable case classification (default: True)'
    )
    parser.add_argument('--output-dir', type=str, default='out',
                       help='Output directory for dry-run mode (default: out)')

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*80)
    print("BigQuery-based Channel Processing")
    print("="*80)
    print(f"Mode:              {'DRY-RUN (CSV/JSON)' if args.dry_run else 'PRODUCTION (BigQuery)'}")
    print(f"Scan Mode:         {'Specific Channels' if args.channel_urls else 'Global Scan'}")
    print(f"Chunk Size:        {args.chunk_size}")
    print(f"Model:             {args.model}")
    print(f"Vision Processing: {'Enabled' if args.enable_vision_processing else 'Disabled'}")
    print(f"Classification:    {'Enabled' if args.enable_classification else 'Disabled'}")
    if args.dry_run:
        print(f"Output Directory:  {args.output_dir}")
    print("="*80)

    # Step 1: Get channels to process
    if args.channel_urls:
        print(f"\n🔍 Checking {len(args.channel_urls)} specified channel(s)...")
    else:
        print(f"\n🔍 Scanning all channels for trigger conditions...")

    messages_df = Utils.get_channels_to_process(
        chunk_size=args.chunk_size,
        idle_days=7,
        channel_urls=args.channel_urls
    )

    if messages_df.empty:
        if args.channel_urls:
            print(f"✅ 0 of {len(args.channel_urls)} channels need processing")
        else:
            print(f"✅ Found 0 channels needing processing")
        print(f"   Total unanalyzed messages: 0")
        print("\n✅ No channels need processing at this time")
        return

    unique_channels = messages_df['channel_url'].nunique()
    if args.channel_urls:
        print(f"✅ {unique_channels} of {len(args.channel_urls)} channels need processing")
    else:
        print(f"✅ Found {unique_channels} channels needing processing")
    print(f"   Total unanalyzed messages: {len(messages_df)}")



    # Step 2: Process each channel
    total_cases = 0
    successful_channels = 0
    session_name = datetime.now().strftime("%y%m%d_%H%M%S")

    # Initialize LLM client once (outside loop)
    llm_client = LLMClient(model=args.model)
    print(f"\n✅ LLM Client initialized with model: {args.model}")

    # Preprocess all messages once (auto-detects and converts column names)
    print(f"\n🔄 Preprocessing all messages...")
    df_clean = Utils.preprocess_dataframe(messages_df, verbose=True)

    # Get unique channel URLs
    channel_urls = df_clean['Channel URL'].unique()

    # Process each channel
    for channel_idx, channel_url in enumerate(channel_urls):
        channel_df = df_clean[df_clean['Channel URL'] == channel_url].copy()

        print(f"\n🔄 Channel {channel_idx + 1}/{len(channel_urls)}: {Utils.format_channel_for_display(channel_url)} ({len(channel_df)} messages)")

        try:
            # Create and process Channel
            channel = Channel(
                df_clean=channel_df,
                channel_url=channel_url,
                session=session_name,
                chunk_size=args.chunk_size,
                overlap=0,  # Fixed at 0 for BigQuery processing
                enable_classification=args.enable_classification,
                enable_vision_processing=args.enable_vision_processing,
                enable_find_sop=False  # Fixed at False - only focus on segmentation and classification
            )

            # Process channel with LLM
            channel.build_cases_via_llm(llm_client)

            if not channel.cases:
                print(f"    ⚠️  No cases generated for this channel")
                continue

            # Save results
            print(f"    💾 Saving results...")
            try:
                if args.dry_run:
                    # Dry-run mode: save CSV and JSON
                    channel.save_results_to_json(args.output_dir)
                    channel.save_results_to_csv(args.output_dir)
                else:
                    # Production mode: save to BigQuery
                    channel.save_results_to_bigquery()

                print(f"    ✅ Results saved successfully")
            except Exception as save_error:
                print(f"        ❌ Error saving results: {str(save_error)}")
                print(f"        Processing completed but save failed - continuing...")

            # Update counters
            total_cases += len(channel.cases)
            successful_channels += 1

        except Exception as e:
            print(f"    ❌ Error processing channel: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Channels processed:    {successful_channels}/{unique_channels}")
    print(f"Total cases generated: {total_cases}")
    if args.dry_run:
        print(f"Results saved to:      {args.output_dir}/session_*")
    else:
        print(f"Results saved to:      BigQuery (plantstory.customer_service.support_message_cases)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
