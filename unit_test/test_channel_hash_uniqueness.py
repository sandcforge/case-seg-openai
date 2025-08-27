#!/usr/bin/env python3
"""
Test script to check if hash parts (after last underscore) in channel URLs are unique

This script analyzes channel URLs like:
sendbird_group_channel_215482988_b374305ff3e440674e786d63916f1d5aacda8249
to extract the hash part (b374305ff3e440674e786d63916f1d5aacda8249) and check for uniqueness.
"""

import pandas as pd
import os
from collections import Counter
from typing import Dict, List


def extract_channel_hash(channel_url: str) -> str:
    """Extract the hash part after the last underscore from a channel URL"""
    if '_' not in channel_url:
        return channel_url
    return channel_url.split('_')[-1]


def test_channel_hash_uniqueness():
    """Test if hash parts in channel URLs are unique"""
    
    # Path to the CSV file
    csv_path = "assets/support_messages_july.csv"
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return False
    
    print(f"📂 Loading {csv_path}...")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check if Channel URL column exists
        if 'Channel URL' not in df.columns:
            print(f"❌ 'Channel URL' column not found in {csv_path}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        # Get unique channel URLs
        unique_channels = df['Channel URL'].dropna().unique()
        total_channels = len(unique_channels)
        
        print(f"📊 Found {total_channels} unique channel URLs")
        
        # Extract hash parts from each channel URL
        channel_hashes = {}
        unknown_format_count = 0
        
        for url in unique_channels:
            url_str = str(url)
            hash_part = extract_channel_hash(url_str)
            
            if hash_part == url_str and '_' not in url_str:
                unknown_format_count += 1
                print(f"⚠️  No underscore found: {url_str}")
                continue
            
            if hash_part not in channel_hashes:
                channel_hashes[hash_part] = []
            channel_hashes[hash_part].append(url_str)
        
        # Find duplicates
        duplicates = {hash_part: urls for hash_part, urls in channel_hashes.items() if len(urls) > 1}
        
        print(f"\n🔍 Analysis Results:")
        print(f"   Successfully parsed: {len(channel_hashes)} channels")
        print(f"   Unknown format: {unknown_format_count} channels")
        print(f"   Unique hash parts: {len(set(channel_hashes.keys()))}")
        print(f"   Duplicate hashes found: {len(duplicates)}")
        
        if unknown_format_count > 0:
            print(f"\n⚠️  WARNING: {unknown_format_count} channels have no underscore in URL")
        
        if duplicates:
            print(f"\n⚠️  DUPLICATE HASH PARTS DETECTED:")
            for hash_part, urls in duplicates.items():
                print(f"\n   Hash '{hash_part}' appears in {len(urls)} channels:")
                for url in urls[:3]:  # Show first 3 examples
                    print(f"      - {url}")
                if len(urls) > 3:
                    print(f"      ... and {len(urls) - 3} more")
            
            print(f"\n❌ Test Result: FAILED - {len(duplicates)} duplicate hash(es) found")
            return False
        else:
            print(f"\n✅ Test Result: PASSED - All hash parts are unique!")
            print(f"   All {len(channel_hashes)} parsed channels have unique hash parts")
            print(f"   Hash parts can be used for channel identification")
            
            # Show some example mappings
            print(f"\n📋 Sample channel hash mappings:")
            sample_count = min(5, len(channel_hashes))
            for i, (hash_part, urls) in enumerate(list(channel_hashes.items())[:sample_count]):
                short_url = urls[0].replace('sendbird_group_channel_', '').replace(f'_{hash_part}', '_...')
                print(f"      {hash_part[:20]}... → {short_url}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return False


def main():
    """Main function to run the test"""
    print("=" * 70)
    print("Channel URL Hash Part Uniqueness Test")
    print("=" * 70)
    
    success = test_channel_hash_uniqueness()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ All tests passed - Hash parts can be used for channel ID!")
    else:
        print("❌ Test failed - check output above for details")
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)