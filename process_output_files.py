#!/usr/bin/env python3
"""
Script to process JSON and CSV files in the output directory and add additional fields:
1. Add channel_url to each case in JSON (already exists at top level)
2. Add global_msg_id_list: Message IDs corresponding to msg_index_list 
3. Add start_timestamp and end_timestamp from CSV data

Usage: python process_output_files.py [output_directory]
"""

import os
import json
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional


def find_matching_csv_file(json_file: Path) -> Optional[Path]:
    """
    Find the corresponding CSV file for a JSON file.
    
    JSON files follow pattern: cases_{channel_hash}.json
    CSV files follow pattern: segmented_{channel_hash}.csv
    """
    json_name = json_file.stem  # Remove .json extension
    if json_name.startswith('cases_'):
        channel_hash = json_name[6:]  # Remove 'cases_' prefix
        csv_name = f"segmented_{channel_hash}.csv"
        csv_file = json_file.parent / csv_name
        if csv_file.exists():
            return csv_file
    return None


def load_csv_data(csv_file: Path) -> pd.DataFrame:
    """Load CSV file and ensure proper data types."""
    try:
        df = pd.read_csv(csv_file)
        
        # Ensure msg_ch_idx is integer
        df['msg_ch_idx'] = df['msg_ch_idx'].astype(int)
        
        # Ensure Message ID is string (handle potential NaN values)
        df['Message ID'] = df['Message ID'].fillna('').astype(str)
        
        # Parse timestamps with flexible format
        df['Created Time'] = pd.to_datetime(df['Created Time'], format='mixed')
        
        return df
        
    except Exception as e:
        print(f"Error loading CSV file {csv_file}: {e}")
        return None


def get_message_details(df: pd.DataFrame, msg_idx_list: List[int]) -> Dict[str, Any]:
    """
    Get message IDs and timestamp range for given msg_ch_idx list.
    
    Returns:
        dict with keys: global_msg_id_list, start_timestamp, end_timestamp
    """
    # Filter DataFrame for the messages in msg_idx_list
    case_messages = df[df['msg_ch_idx'].isin(msg_idx_list)].copy()
    
    if case_messages.empty:
        print(f"Warning: No messages found for msg_idx_list: {msg_idx_list}")
        return {
            'global_msg_id_list': [],
            'start_timestamp': None,
            'end_timestamp': None
        }
    
    # Sort by msg_ch_idx to maintain order
    case_messages = case_messages.sort_values('msg_ch_idx')
    
    # Get message IDs (convert to list, filter out empty strings)
    global_msg_id_list = [mid for mid in case_messages['Message ID'].tolist() if mid]
    
    # Get timestamp range
    timestamps = case_messages['Created Time']
    start_timestamp = timestamps.min().isoformat() if not timestamps.empty else None
    end_timestamp = timestamps.max().isoformat() if not timestamps.empty else None
    
    return {
        'global_msg_id_list': global_msg_id_list,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp
    }


def process_json_file(json_file: Path, csv_df: pd.DataFrame) -> bool:
    """
    Process a single JSON file and add the required fields.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Load JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract channel_url from top level
        channel_url = data.get('channel_url', '')
        
        # Process each case in global_cases
        global_cases = data.get('global_cases', [])
        
        print(f"    Processing {len(global_cases)} cases...")
        
        for case in global_cases:
            # Add channel_url to each case
            case['channel_url'] = channel_url
            
            # Get msg_index_list
            msg_idx_list = case.get('msg_index_list', [])
            
            if not msg_idx_list:
                print(f"    Warning: Case {case.get('case_id', 'unknown')} has empty msg_index_list")
                case['global_msg_id_list'] = []
                case['start_timestamp'] = None
                case['end_timestamp'] = None
                continue
            
            # Get message details from CSV
            message_details = get_message_details(csv_df, msg_idx_list)
            
            # Add the new fields
            case['global_msg_id_list'] = message_details['global_msg_id_list']
            case['start_timestamp'] = message_details['start_timestamp']
            case['end_timestamp'] = message_details['end_timestamp']
            
            # Debug info
            case_id = case.get('case_id', 'unknown')
            msg_count = len(message_details['global_msg_id_list'])
            print(f"        Case {case_id}: {len(msg_idx_list)} msg_idx -> {msg_count} message IDs")
        
        # Write updated JSON back to file
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"    Error processing JSON file {json_file}: {e}")
        return False


def process_session_directory(session_dir: Path) -> None:
    """Process all JSON files in a session directory."""
    print(f"\n📁 Processing session directory: {session_dir.name}")
    
    # Find all JSON case files (exclude statistics files)
    json_files = [f for f in session_dir.glob("cases_*.json")]
    
    if not json_files:
        print("    No case JSON files found")
        return
    
    print(f"    Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        print(f"  🔄 Processing: {json_file.name}")
        
        # Find corresponding CSV file
        csv_file = find_matching_csv_file(json_file)
        if not csv_file:
            print(f"    ❌ No matching CSV file found for {json_file.name}")
            continue
        
        print(f"    📊 Loading CSV: {csv_file.name}")
        
        # Load CSV data
        csv_df = load_csv_data(csv_file)
        if csv_df is None:
            print(f"    ❌ Failed to load CSV file")
            continue
        
        print(f"    📊 CSV loaded: {len(csv_df)} rows")
        
        # Process the JSON file
        success = process_json_file(json_file, csv_df)
        
        if success:
            print(f"    ✅ Successfully processed {json_file.name}")
        else:
            print(f"    ❌ Failed to process {json_file.name}")


def main():
    """Main function to process all session directories."""
    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("out")
    
    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        return
    
    print(f"🚀 Starting output file processing in: {output_dir}")
    print(f"📅 Processing started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if the provided path is already a session directory
    if output_dir.name.startswith('session_') and output_dir.is_dir():
        print(f"📁 Processing single session directory: {output_dir.name}")
        process_session_directory(output_dir)
    else:
        # Find all session directories within the provided path
        session_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('session_')]
        
        if not session_dirs:
            print("❌ No session directories found")
            return
        
        print(f"📁 Found {len(session_dirs)} session directories")
        
        # Process each session directory
        for session_dir in sorted(session_dirs):
            process_session_directory(session_dir)
    
    print(f"\n✅ Processing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()