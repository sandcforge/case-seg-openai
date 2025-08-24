#!/usr/bin/env python3
"""
Customer support message segmentation system.

This module implements a two-stage pipeline for processing customer support messages:
1. FileProcessor: Loads CSV data and performs preprocessing (role assignment, time parsing, sorting)
2. ChannelSegmenter: Segments processed data into chunks for LLM analysis

Usage:
    python main.py [--input INPUT] [--output-dir OUTPUT_DIR] [--chunk-size SIZE]

Example:
    python main.py --chunk-size 80
"""

import os
import argparse
import pandas as pd # type: ignore
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv # type: ignore

# Local imports
from file_processor import FileProcessor
from llm_client import LLMClient
from utils import Utils
from chunk import Chunk

# Load environment variables
load_dotenv()


# ----------------------------
# Merge Overlap Utilities
# ----------------------------

@dataclass(frozen=True)
class CaseRef:
    """Reference to a local case inside a chunk."""
    chunk_idx: int  # 0: chunk k, 1: chunk k+1
    case_id: int

    def uf_key(self) -> str:
        return f"{self.chunk_idx}#{self.case_id}"


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra



class ChannelSegmenter:
    """
    Segments processed messages into chunks for LLM analysis.
    Assumes single channel input.
    
    Features:
    - Half-open intervals: Uses [start, end) to avoid boundary duplication
    - Direct chunking: Simple chunk_size-based segmentation
    - Case merging: Handles pairwise merge and global aggregation
    """
    
    # Case schema and anchor constants
    REQUIRED_FIELDS_DEFAULTS = {
        "summary": "N/A",
        "status": "ongoing",            # 缺省设为 ongoing，便于保守承接
        "pending_party": "N/A",
        "last_update": "N/A",
        "confidence": 0.0,
        "anchors": {}
    }
    
    ANCHOR_KEYS_STRICT = ("tracking", "order", "buyer", "topic")
    ANCHOR_KEYS_LAX = ("tracking", "order", "order_ids", "buyer", "buyers", "topic")
    
    def __init__(self, df_clean: pd.DataFrame, channel_url: str, chunk_size: int = 80, overlap: int = 20, review_gap_threshold: float = 0.05):
        self.df_clean = df_clean
        self.channel_url = channel_url
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.review_gap_threshold = review_gap_threshold
        self.chunks: List[Chunk] = []
        
        self.validate_parameters()
    
    def validate_parameters(self) -> None:
        """Validate chunk_size parameter"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
    
    def generate_chunks(self) -> List[Chunk]:
        """Generate chunks for single channel"""
        total_messages = len(self.df_clean)
        
        # Use the channel URL provided in constructor
        channel_url = self.channel_url
        
        # Reset index to ensure continuous indexing within channel
        channel_df = self.df_clean.reset_index(drop=True)
        
        # Calculate number of chunks needed
        import math
        num_chunks = math.ceil(total_messages / self.chunk_size)
        
        for i in range(num_chunks):
            # Calculate chunk boundaries using half-open intervals
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, total_messages)
            
            # Create chunk with DataFrame slice
            chunk_messages = channel_df.iloc[start_idx:end_idx].copy()
            
            chunk = Chunk(
                chunk_id=i,
                channel_url=channel_url,
                chunk_df=chunk_messages
            )
            
            msg_indices = chunk_messages['msg_ch_idx'].tolist()
            print(f"Generated chunk {i}: msg_indices [{min(msg_indices)}, {max(msg_indices)}], "
                  f"{len(chunk_messages)} messages, channel: {Utils.format_channel_for_display(channel_url)}")
            self.chunks.append(chunk)
        
        print(f"Generated {len(self.chunks)} chunks for single channel")
        return self.chunks
    
    def segment_all_chunks(self, chunks: List[Chunk], llm_client: 'LLMClient') -> List[List[Dict[str, Any]]]:
        """
        对所有chunks进行case segmentation，返回原始分割结果
        
        Args:
            chunks: 要处理的chunk列表
            llm_client: LLM客户端
            
        Returns:
            每个chunk的case分割结果列表
        """
        print(f"Segmenting {len(chunks)} chunks...")
        
        chunk_cases = []
        for i, chunk in enumerate(chunks):
            print(f"\n--- Processing chunk {chunk.chunk_id} ({i+1}/{len(chunks)}) ---")
            
            current_messages = chunk.format_all_messages_for_prompt()
            case_results = chunk.generate_case_segments(
                current_chunk_messages=current_messages,
                llm_client=llm_client
            )
            
            chunk_cases.append(case_results)
        
        print(f"✅ Segmentation complete: {len(chunks)} chunks processed")
        return chunk_cases
    
    def segment_all_chunks_with_review(self, chunks: List[Chunk], llm_client: 'LLMClient') -> Dict[str, Any]:
        """
        处理所有chunks并执行case review，返回全局cases
        
        Args:
            chunks: 要处理的chunk列表
            llm_client: LLM客户端
            
        Returns:
            全局cases处理结果
        """
        print(f"\n=== Processing {len(chunks)} chunks with review pipeline ===")
        
        if not chunks:
            return {"global_cases": [], "total_messages": 0, "chunks_processed": 0}
        
        # Stage 1: 对所有chunks进行case segmentation
        chunk_cases = self.segment_all_chunks(chunks, llm_client)
        
        # Stage 2: 执行case review
        print("🔍 Performing case boundary review")
        return self.execute_case_review(chunk_cases, chunks, llm_client)
    
    def segment_all_chunks_simple(self, chunks: List[Chunk], llm_client: 'LLMClient') -> Dict[str, Any]:
        """
        处理所有chunks并执行简单合并，不进行review
        
        Args:
            chunks: 要处理的chunk列表
            llm_client: LLM客户端
            
        Returns:
            全局cases处理结果
        """
        print(f"\n=== Processing {len(chunks)} chunks with simple merge ===")
        return self.segment_all_chunks(chunks, llm_client)
    

    def execute_case_review(
        self,
        chunks: List[Chunk],
        llm_client: 'LLMClient'
    ) -> Dict[str, Any]:
        """
        执行case review（当前使用简化实现）
        """
        print("⚠️  Advanced case review not fully implemented yet")
        print("   Using simple merge as fallback for now")
        
        # 暂时使用简单合并作为fallback
        return self.execute_merge_pipeline(chunks)
    
    def execute_merge_pipeline(
        self,
        chunks: List[Chunk]
    ) -> Dict[str, Any]:
        """
        执行merge pipeline的数据处理阶段（不包含LLM调用）
        
        Args:
            chunk_cases: 每个chunk的case分割结果
            tail_summaries: 每个chunk的tail summary
            chunks: chunk列表
            
        Returns:
            包含global_cases
        """
        print(f"Executing merge pipeline for {len(chunks)} chunks")
        return []

    def validate_results(self) -> dict:
        """Validate segmentation results and prepare data for saving"""
        # Use channel URL from constructor
        channel_url = self.channel_url
        
        # Initialize counters and data structures
        total_analyzed = len(self.df_clean)
        chunks_processed = 0
        cases_count = 0
        global_cases = []  # Still need this for saving
        
        print(f"\n--- Validating Channel Results ---")
        
        # Generate annotated CSV for this channel
        df_annotated = self.df_clean.copy()
        df_annotated['case_id'] = "unassigned"  # Default: unassigned (string type)
        
        # Map case assignments using channel's local msg_ch_idx
        assignment_stats = {"out_of_range": 0, "conflicts": 0}
        
        # Direct 2-level loop: chunk -> case -> message
        for chunk in self.chunks:
            if chunk.has_segmentation_result and chunk.cases:
                chunks_processed += 1
                
                for case in chunk.cases:
                    cases_count += 1
                    # Convert to dict for saving (only when needed)
                    case_dict = case.to_dict()
                    global_cases.append(case_dict)
                    
                    # Process messages directly from CaseItem object
                    for msg_ch_idx in case.msg_index_list:
                        # Use msg_ch_idx column instead of DataFrame row index
                        mask = df_annotated['msg_ch_idx'] == msg_ch_idx
                        matching_rows = df_annotated[mask]
                        
                        if len(matching_rows) == 0:
                            assignment_stats["out_of_range"] += 1
                            print(f"⚠️  Warning: Message msg_ch_idx {msg_ch_idx} not found in channel data")
                        else:
                            # Check for conflicts
                            if matching_rows['case_id'].iloc[0] != "unassigned":
                                assignment_stats["conflicts"] += 1
                                print(f"⚠️  Warning: Message {msg_ch_idx} already assigned, reassigning to case {case.case_id}")
                            
                            # Assign case_id using msg_ch_idx-based selection
                            df_annotated.loc[mask, 'case_id'] = case.case_id
        
        print(f"Found {cases_count} cases across {total_analyzed} messages")
        
        # Display assignment statistics for this channel
        assigned_count = (df_annotated['case_id'] != "unassigned").sum()
        unassigned_count = (df_annotated['case_id'] == "unassigned").sum()
        coverage_rate = assigned_count / len(df_annotated) * 100 if len(df_annotated) > 0 else 0
        
        print(f"📊 Channel Assignment Statistics:")
        print(f"   Total messages: {len(df_annotated)}")
        print(f"   Assigned to cases: {assigned_count} ({coverage_rate:.1f}%)")
        print(f"   Unassigned: {unassigned_count}")
        print(f"   Cases generated: {cases_count}")
        print(f"   Chunks processed: {chunks_processed}")
        
        # Show warnings if any
        if assignment_stats["out_of_range"] > 0:
            print(f"   ⚠️  Messages not found: {assignment_stats['out_of_range']}")
        if assignment_stats["conflicts"] > 0:
            print(f"   ⚠️  Conflicts resolved: {assignment_stats['conflicts']}")
        
        if coverage_rate == 100.0:
            print(f"✅ Perfect coverage achieved for channel!")
        elif coverage_rate >= 95.0:
            print(f"✅ Excellent coverage achieved for channel!")
        else:
            print(f"⚠️  Coverage could be improved for channel")
        
        # Return validation report and processed data
        return {
            "channel_url": channel_url,
            "global_cases": global_cases,
            "total_messages": total_analyzed,
            "chunks_processed": chunks_processed,
            "df_annotated": df_annotated,
            "assignment_stats": assignment_stats,
            "coverage_rate": coverage_rate
        }

    def save_results_to_json(self, output_dir: str) -> None:
        """Save channel cases to JSON file"""
        import json
        
        # Collect cases directly from chunks 
        global_cases = []
        chunks_processed = 0
        
        for chunk in self.chunks:
            if chunk.has_segmentation_result and chunk.cases:
                chunks_processed += 1
                for case in chunk.cases:
                    case_dict = case.to_dict()
                    global_cases.append(case_dict)
        
        # Save channel cases to JSON
        channel_name = Utils.format_channel_for_display(self.channel_url)
        channel_cases_file = os.path.join(output_dir, f"cases_{channel_name}.json")
        save_result = {
            "channel_url": self.channel_url,
            "global_cases": global_cases,
            "total_messages": len(self.df_clean),
            "chunks_processed": chunks_processed
        }
        
        try:
            with open(channel_cases_file, 'w', encoding='utf-8') as f:
                json.dump(save_result, f, indent=2, ensure_ascii=False)
            print(f"Channel cases saved to: {channel_cases_file}")
        except IOError as e:
            print(f"❌ Error saving JSON file: {e}")
            raise
    
    def save_results_to_csv(self, output_dir: str) -> None:
        """Save annotated messages to CSV file"""
        # Collect cases for message annotation
        global_cases = []
        
        for chunk in self.chunks:
            if chunk.has_segmentation_result and chunk.cases:
                for case in chunk.cases:
                    case_dict = case.to_dict()
                    global_cases.append(case_dict)
        
        # Generate annotated CSV for this channel
        df_annotated = self.df_clean.copy()
        df_annotated['case_id'] = "unassigned"  # Default: unassigned (string type)
        
        # Map case assignments using msg_ch_idx
        for case_dict in global_cases:
            case_id = case_dict.get('case_id', "unknown")
            for msg_ch_idx in case_dict.get('msg_index_list', []):
                mask = df_annotated['msg_ch_idx'] == msg_ch_idx
                df_annotated.loc[mask, 'case_id'] = case_id
        
        # Save annotated CSV for this channel
        channel_name = Utils.format_channel_for_display(self.channel_url)
        channel_segmented_file = os.path.join(output_dir, f"segmented_{channel_name}.csv")
        try:
            df_annotated.to_csv(channel_segmented_file, index=False, encoding='utf-8')
            print(f"Channel annotated CSV saved to: {channel_segmented_file}")
        except IOError as e:
            print(f"❌ Error saving CSV file: {e}")
            raise



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
    
    args = parser.parse_args()
    
    try:
        # Stage 1: File Processing
        processor = FileProcessor(args.input, args.output_dir)
        channel_data_list = processor.process()
        
        if not channel_data_list:
            print("Error: File processing failed")
            exit(1)
        
        print(f"Found {len(channel_data_list)} channels to process")
        
        # Stage 3: Initialize LLM Client for case segmentation
        llm_client = LLMClient(model=args.model)
        print(f"LLM Client initialized with model: {args.model}")
        
        # Process each channel separately
        
        for channel_idx, channel_data in enumerate(channel_data_list):
            channel_url = channel_data["channel_url"]
            channel_df = channel_data["dataframe"]
            
            print(f"\n=== Processing Channel {channel_idx + 1}/{len(channel_data_list)} ===")
            print(f"Channel: {Utils.format_channel_for_display(channel_url)}")
            print(f"Messages: {len(channel_df)}")
            
            # Stage 2: Channel Segmentation for this channel
            one_ch = ChannelSegmenter(channel_df, channel_url, args.chunk_size, args.overlap)
            one_ch.generate_chunks()
            
            print(f"Generated {len(one_ch.chunks)} chunks with chunk_size={args.chunk_size}")
            
            # Process this channel with full pipeline and save results immediately
            if args.enable_review:
                one_ch.segment_all_chunks_with_review(one_ch.chunks, llm_client)
            else:
                one_ch.segment_all_chunks_simple(one_ch.chunks, llm_client)
                
            # Validate results
            one_ch.validate_results()
            
            # Save this channel's results independently
            print(f"\n--- Saving Channel Results ---")
            one_ch.save_results_to_json(args.output_dir)
            one_ch.save_results_to_csv(args.output_dir)
        
        
        # Summary for all channels
        print(f"\n✅ All {len(channel_data_list)} channels processed successfully!")
        print(f"Each channel's results saved to separate files:")
        for i in range(len(channel_data_list)):
            print(f"  Channel {i + 1}: cases_channel_{i + 1}.json, segmented_channel_{i + 1}.csv")
        
        print(f"\n✅ Pipeline complete!")
        
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == '__main__':
    main()