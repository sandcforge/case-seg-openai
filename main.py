#!/usr/bin/env python3
"""
Customer support message segmentation system.

This module implements a two-stage pipeline for processing customer support messages:
1. FileProcessor: Loads CSV data and performs preprocessing (role assignment, time parsing, sorting)
2. ChannelSegmenter: Segments processed data into overlapping chunks for LLM analysis

Usage:
    python main.py [--input INPUT] [--output-dir OUTPUT_DIR] [--chunk-size SIZE] [--overlap SIZE]

Example:
    python main.py --chunk-size 80 --overlap 20
"""

import os
import argparse
import pandas as pd
import pytz
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    """Data structure for a single chunk of messages"""
    chunk_id: int                    # Sequential chunk ID (0, 1, 2, ...)
    channel_url: str                 # Channel this chunk belongs to
    start_idx: int                   # Start index in the channel (inclusive)
    end_idx: int                     # End index in the channel (exclusive) - half-open interval [start, end)
    messages: pd.DataFrame           # DataFrame slice with messages in this chunk
    total_messages: int              # Number of messages in this chunk
    has_overlap_with_previous: bool  # Whether this chunk overlaps with previous chunk
    overlap_size: int                # Number of overlapping messages with previous chunk

    def get_message_indices(self) -> List[int]:
        """Get list of msg_ch_idx values for messages in this chunk"""
        return self.messages['msg_ch_idx'].tolist()
    
    def format_for_prompt(self) -> str:
        """Format chunk messages as: message_index | sender id | role | timestamp | text"""
        formatted_lines = []
        for _, row in self.messages.iterrows():
            # Handle NaN messages and replace newlines with spaces to keep one line per message
            message_text = str(row['Message']).replace('\n', ' ').replace('\r', ' ')
            if message_text == 'nan':
                message_text = ''
            
            formatted_lines.append(
                f"{row['msg_ch_idx']} | {row['Sender ID']} | {row['role']} | {row['Created Time']} | {message_text}"
            )
        return '\n'.join(formatted_lines)


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
            print(f"Loaded {len(self.df)} messages from {self.input_file}")
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
            print(f"Added role column: {self.df['role'].value_counts().to_dict()}")
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
        print(f"Processed time columns, converted {len(self.df)} timestamps to UTC")
    
    def sort_and_group_data(self) -> None:
        """Sort data by Channel URL, Created Time, then Message ID"""
        self.df = self.df.sort_values([
            'Channel URL', 
            'Created Time', 
            'Message ID'
        ]).reset_index(drop=True)
        
        print(f"Sorted data by Channel URL, Created Time, and Message ID")
    
    def add_message_index(self) -> None:
        """Add msg_ch_idx column (0..N-1 for each Channel URL group)"""
        self.df['msg_ch_idx'] = self.df.groupby('Channel URL').cumcount()
        print(f"Added msg_ch_idx column for {self.df['Channel URL'].nunique()} channels")
    
    def create_clean_dataframe(self) -> pd.DataFrame:
        """Generate clean DataFrame with essential columns"""
        essential_columns = [
            'Created Time', 'Sender ID', 'Message', 'Channel URL',
            'role', 'msg_ch_idx', 'Message ID'
        ]
        
        available_columns = [col for col in essential_columns if col in self.df.columns]
        self.df_clean = self.df[available_columns].copy()
        
        print(f"Created clean DataFrame with {len(available_columns)} columns: {available_columns}")
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
    
    def process(self) -> Optional[pd.DataFrame]:
        """Execute the complete processing pipeline"""
        print("Starting file processing...")
        
        if not self.load_data():
            return None
            
        # Execute processing pipeline
        self.add_role_column()
        self.process_time_columns()
        self.sort_and_group_data()
        self.add_message_index()
        self.create_clean_dataframe()
        
        # Save output
        output_path = self.save_output()
        
        print(f"Processing complete! Output saved to {output_path}")
        print(f"Processed {len(self.df_clean)} messages across {self.df_clean['Channel URL'].nunique()} channels")
        
        return self.df_clean


class ChannelSegmenter:
    """
    Segments processed messages into overlapping chunks for LLM analysis.
    
    Features:
    - Channel separation: Each channel is processed independently
    - Half-open intervals: Uses [start, end) to avoid boundary duplication
    - Overlap validation: Ensures overlap < chunk_size/3 for optimal coverage
    - Chunk tracking: Maintains overlap metadata for context continuity
    """
    
    def __init__(self, df_clean: pd.DataFrame, chunk_size: int = 80, overlap: int = 20):
        self.df_clean = df_clean
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Chunk] = []
        
        self.validate_parameters()
    
    def validate_parameters(self) -> None:
        """Validate chunk_size and overlap parameters"""
        if self.overlap >= self.chunk_size / 3:
            raise ValueError(
                f"Overlap ({self.overlap}) must be less than chunk_size/3 ({self.chunk_size/3:.1f})"
            )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap cannot be negative")
    
    def generate_chunks(self) -> List[Chunk]:
        """Generate chunks for all channels"""
        self.chunks = []
        chunk_id_counter = 0
        
        # Process each channel separately
        for channel_url in self.df_clean['Channel URL'].unique():
            channel_df = self.df_clean[self.df_clean['Channel URL'] == channel_url].copy()
            
            # Generate chunks for this channel
            channel_chunks = self.get_channel_chunks(channel_df, channel_url, chunk_id_counter)
            self.chunks.extend(channel_chunks)
            chunk_id_counter += len(channel_chunks)
        
        print(f"Generated {len(self.chunks)} chunks across {self.df_clean['Channel URL'].nunique()} channels")
        return self.chunks
    
    def get_channel_chunks(self, channel_df: pd.DataFrame, channel_url: str, start_chunk_id: int) -> List[Chunk]:
        """Generate chunks for a single channel using half-open intervals"""
        channel_chunks = []
        total_messages = len(channel_df)
        
        if total_messages == 0:
            return channel_chunks
        
        # Reset index to ensure continuous indexing within channel
        channel_df = channel_df.reset_index(drop=True)
        
        chunk_id = start_chunk_id
        i = 0
        
        while True:
            # Calculate chunk boundaries using half-open intervals
            if i == 0:
                # First chunk: [0, chunk_size)
                start_idx = 0
                end_idx = min(self.chunk_size, total_messages)
                has_overlap_with_previous = False
                overlap_size = 0
            else:
                # Subsequent chunks: [(i)*chunk_size - overlap, (i+1)*chunk_size)
                start_idx = max(0, i * self.chunk_size - self.overlap)
                end_idx = min((i + 1) * self.chunk_size, total_messages)
                has_overlap_with_previous = True
                overlap_size = min(self.overlap, start_idx)
            
            # Break if we've reached the end
            if start_idx >= total_messages:
                break
            
            # Create chunk with DataFrame slice
            chunk_messages = channel_df.iloc[start_idx:end_idx].copy()
            
            chunk = Chunk(
                chunk_id=chunk_id,
                channel_url=channel_url,
                start_idx=start_idx,
                end_idx=end_idx,
                messages=chunk_messages,
                total_messages=len(chunk_messages),
                has_overlap_with_previous=has_overlap_with_previous,
                overlap_size=overlap_size
            )
            
            print(f"Generated chunk {chunk_id}: [{start_idx}, {end_idx}), "
                  f"{len(chunk_messages)} messages, channel: {channel_url[:30]}...")
            channel_chunks.append(chunk)
            chunk_id += 1
            i += 1
            
            # Break if this chunk reaches the end
            if end_idx >= total_messages:
                break
        
        return channel_chunks


def main() -> None:
    """Main entry point for the message processing pipeline"""
    parser = argparse.ArgumentParser(
        description='Process customer support messages into overlapping chunks for LLM analysis'
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
    
    args = parser.parse_args()
    
    try:
        # Stage 1: File Processing
        processor = FileProcessor(args.input, args.output_dir)
        df_clean = processor.process()
        
        if df_clean is None:
            print("Error: File processing failed")
            exit(1)
        
        # Stage 2: Channel Segmentation
        segmenter = ChannelSegmenter(df_clean, args.chunk_size, args.overlap)
        chunks = segmenter.generate_chunks()
        
        print(f"\n✅ Pipeline complete!")
        print(f"Generated {len(chunks)} chunks with chunk_size={args.chunk_size}, overlap={args.overlap}")
        
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == '__main__':
    main()