#!/usr/bin/env python3
"""
File processing module for customer support message data.

This module handles CSV data loading, preprocessing, and cleaning operations including:
- Role assignment based on sender ID patterns
- Timezone-aware timestamp parsing and UTC conversion
- Data sorting by channel, time, and message ID
- Message indexing within channels
"""

import os
import pandas as pd # type: ignore
import pytz # type: ignore
from pathlib import Path
from typing import List, Dict, Any, Optional

from utils import Utils


class FileProcessor:
    """
    Processes raw CSV files containing customer support messages.
    
    Performs data loading, preprocessing, and cleaning operations including:
    - Role assignment based on sender ID patterns
    - Timezone-aware timestamp parsing and UTC conversion
    - Data sorting by channel, time, and message ID
    - Message indexing within channels
    """
    
    def __init__(self, input_file: str, output_dir: str = "out"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        
    def load_data(self) -> bool:
        """Load CSV data into DataFrame"""
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"        Loaded {len(self.df)} messages from {self.input_file}")
            return True
        except Exception as e:
            print(f"Error loading file {self.input_file}: {e}")
            return False
    
    def add_role_column(self) -> None:
        """Add role column based on Sender ID pattern"""
        if 'role' not in self.df.columns:
            self.df['role'] = self.df['Sender ID'].apply(
                lambda x: 'customer_service' if str(x).startswith('psops') else 'user'
            )
            print(f"        Added role column: {self.df['role'].value_counts().to_dict()}")
        else:
            print("Role column already exists, skipping...")
    
    def process_time_columns(self) -> None:
        """Parse Created Time to timezone-aware UTC format"""        
        def parse_to_utc(time_str):
            try:
                dt = pd.to_datetime(time_str)
                if dt.tz is not None:
                    return dt.astimezone(pytz.UTC)
                else:
                    return pytz.UTC.localize(dt)
            except Exception as e:
                print(f"Error parsing time {time_str}: {e}")
                return pd.NaT
        
        self.df['Created Time'] = self.df['Created Time'].apply(parse_to_utc)
        print(f"        Processed time columns, converted {len(self.df)} timestamps to UTC")
    
    def sort_and_group_data(self) -> None:
        """Sort data by Channel URL, Created Time, then Message ID"""
        self.df = self.df.sort_values([
            'Channel URL', 
            'Created Time', 
            'Message ID'
        ]).reset_index(drop=True)
        
        print(f"        Sorted data by Channel URL, Created Time, and Message ID")
    
    def add_message_index(self) -> None:
        """Add msg_ch_idx column (0..N-1 for each Channel URL group)"""
        self.df['msg_ch_idx'] = self.df.groupby('Channel URL').cumcount()
        print(f"        Added msg_ch_idx column for {self.df['Channel URL'].nunique()} channels")
    
    def filter_deleted_rows(self) -> None:
        """Filter out rows where Deleted = True"""
        if 'Deleted' in self.df.columns:
            original_count = len(self.df)
            self.df = self.df[self.df['Deleted'] != True].reset_index(drop=True)
            filtered_count = original_count - len(self.df)
            print(f"        Filtered out {filtered_count} deleted rows ({len(self.df)} remaining)")
        else:
            print("No 'Deleted' column found, skipping deletion filter")
    
    def create_clean_dataframe(self) -> pd.DataFrame:
        """Generate clean DataFrame with essential columns"""
        essential_columns = [
            'Created Time', 'Sender ID', 'Message', 'Channel URL',
            'role', 'msg_ch_idx', 'Message ID'
        ]
        
        available_columns = [col for col in essential_columns if col in self.df.columns]
        self.df_clean = self.df[available_columns].copy()
        
        print(f"        Created clean DataFrame with {len(available_columns)} columns: {available_columns}")
        return self.df_clean
    
    def save_output(self) -> str:
        """Save processed DataFrame to output file"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        input_path = Path(self.input_file)
        output_filename = f"{input_path.stem}_out.csv"
        output_path = os.path.join(self.output_dir, output_filename)
        
        self.df_clean.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")
        return output_path
    
    def process(self) -> List[Dict[str, Any]]:
        """Execute the complete processing pipeline and return list of channel data"""
        print("Starting file processing...")
        
        if not self.load_data():
            return []
            
        # Execute processing pipeline
        self.filter_deleted_rows()
        self.add_role_column()
        self.process_time_columns()
        self.sort_and_group_data()
        self.add_message_index()
        self.create_clean_dataframe()

        print(f"        Processed {len(self.df_clean)} messages across {self.df_clean['Channel URL'].nunique()} channels")
        
        # Group by channel and create list of channel data
        channel_data_list = []
        for channel_url in self.df_clean['Channel URL'].unique():
            channel_df = self.df_clean[self.df_clean['Channel URL'] == channel_url].copy()
            # Reset msg_ch_idx to ensure it starts from 0 for each channel
            channel_df['msg_ch_idx'] = range(len(channel_df))
            
            channel_data_list.append({
                "channel_url": channel_url,
                "dataframe": channel_df
            })
            
            print(f"                Channel: {Utils.format_channel_for_display(channel_url)} - {len(channel_df)} messages")
        
        return channel_data_list