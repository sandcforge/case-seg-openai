#!/usr/bin/env python3
"""
Test script to check if middle numbers in channel URLs can be used for unique identification

This script analyzes channel URLs like:
sendbird_group_channel_215482988_b374305ff3e440674e786d63916f1d5aacda8249
to see if the middle number (215482988) is unique across all channels.
"""

import pandas as pd
import os
import re
from collections import Counter


def extract_channel_number(channel_url: str) -> str:
    """Extract the middle number from a channel URL"""
    # Pattern: sendbird_group_channel_NUMBERS_hash
    match = re.match(r'sendbird_group_channel_(\d+)_[a-f0-9]+', str(channel_url))
    if match:
        return match.group(1)
    return "unknown"


def test_channel_middle_number_uniqueness():
    """Test if middle numbers in channel URLs are unique"""
    
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
        
        # Extract middle numbers from each channel URL
        channel_numbers = {}
        unknown_format_count = 0
        
        for url in unique_channels:
            url_str = str(url)
            number = extract_channel_number(url_str)
            
            if number == "unknown":
                unknown_format_count += 1
                print(f"⚠️  Unknown format: {url_str}")
                continue
            
            if number not in channel_numbers:
                channel_numbers[number] = []
            channel_numbers[number].append(url_str)
        
        # Count number occurrences
        number_counts = Counter([num for num in channel_numbers.keys()])
        
        # Find duplicates
        duplicates = {num: urls for num, urls in channel_numbers.items() if len(urls) > 1}
        
        print(f"\n🔍 Analysis Results:")
        print(f"   Successfully parsed: {len(channel_numbers)} channels")
        print(f"   Unknown format: {unknown_format_count} channels")
        print(f"   Unique middle numbers: {len(set(channel_numbers.keys()))}")
        print(f"   Duplicate numbers found: {len(duplicates)}")
        
        if unknown_format_count > 0:
            print(f"\n⚠️  WARNING: {unknown_format_count} channels have unknown URL format")
        
        if duplicates:
            print(f"\n⚠️  DUPLICATE MIDDLE NUMBERS DETECTED:")
            for number, urls in duplicates.items():
                print(f"\n   Number '{number}' appears in {len(urls)} channels:")
                for url in urls[:3]:  # Show first 3 examples
                    print(f"      - {url}")
                if len(urls) > 3:
                    print(f"      ... and {len(urls) - 3} more")
            
            print(f"\n❌ Test Result: FAILED - {len(duplicates)} duplicate number(s) found")
            return False
        else:
            print(f"\n✅ Test Result: PASSED - All middle numbers are unique!")
            print(f"   All {len(channel_numbers)} parsed channels have unique middle numbers")
            print(f"   Middle numbers can be used for channel identification")
            
            # Show some example mappings
            print(f"\n📋 Sample channel number mappings:")
            sample_count = min(5, len(channel_numbers))
            for i, (number, urls) in enumerate(list(channel_numbers.items())[:sample_count]):
                short_url = urls[0].replace('sendbird_group_channel_', '').replace(f'_{number}_', '_..._')
                print(f"      {number} → {short_url}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return False


def main():
    """Main function to run the test"""
    print("=" * 70)
    print("Channel URL Middle Number Uniqueness Test")
    print("=" * 70)
    
    success = test_channel_middle_number_uniqueness()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ All tests passed - Middle numbers can be used for channel ID!")
    else:
        print("❌ Test failed - check output above for details")
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)