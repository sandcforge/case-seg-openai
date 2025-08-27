#!/usr/bin/env python3
"""
Test script to check for duplicate channel URL suffixes in support_messages_july.csv

This script analyzes the last 5 characters of channel URLs to identify potential
naming conflicts that could affect channel display formatting.
"""

import pandas as pd
import os
from collections import Counter
from typing import Dict, List, Tuple


def test_channel_url_suffix_duplicates():
    """Test for duplicate last 5 characters in channel URLs"""
    
    # Path to the CSV file
    csv_path = "assets/support_message_July.csv"
    
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
        
        # Extract last 5 characters from each channel URL
        url_suffixes: Dict[str, List[str]] = {}
        
        for url in unique_channels:
            url_str = str(url)
            if len(url_str) >= 5:
                suffix = url_str[-5:]
                if suffix not in url_suffixes:
                    url_suffixes[suffix] = []
                url_suffixes[suffix].append(url_str)
        
        # Count suffix occurrences
        suffix_counts = Counter([suffix for suffix in url_suffixes.keys()])
        
        # Find duplicates
        duplicates = {suffix: urls for suffix, urls in url_suffixes.items() if len(urls) > 1}
        
        print(f"\n🔍 Analysis Results:")
        print(f"   Total unique suffixes: {len(url_suffixes)}")
        print(f"   Duplicate suffixes found: {len(duplicates)}")
        
        if duplicates:
            print(f"\n⚠️  DUPLICATE SUFFIXES DETECTED:")
            for suffix, urls in duplicates.items():
                print(f"\n   Suffix '{suffix}' appears in {len(urls)} channels:")
                for url in urls:
                    print(f"      - {url}")
            
            print(f"\n❌ Test Result: FAILED - {len(duplicates)} duplicate suffix(es) found")
            return False
        else:
            print(f"\n✅ Test Result: PASSED - No duplicate suffixes found")
            print(f"   All {total_channels} channels have unique last-5-character suffixes")
            return True
            
    except Exception as e:
        print(f"❌ Error processing file: {str(e)}")
        return False


def main():
    """Main function to run the test"""
    print("=" * 60)
    print("Channel URL Suffix Duplicate Test")
    print("=" * 60)
    
    success = test_channel_url_suffix_duplicates()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Test failed - check output above for details")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)