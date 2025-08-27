#!/usr/bin/env python3
"""
Channel processing module for customer support message segmentation.

This module contains the Channel class that handles:
- Message segmentation into chunks for LLM analysis
- Case generation and classification
- Results validation and output
"""

import os
import pandas as pd  # type: ignore
from typing import List, Dict, Any, TYPE_CHECKING
from chunk import Chunk
from utils import Utils

if TYPE_CHECKING:
    from llm_client import LLMClient


class Channel:
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
    
    def __init__(self, df_clean: pd.DataFrame, channel_url: str, session: str, chunk_size: int = 80, overlap: int = 20):
        self.df_clean = df_clean
        self.channel_url = channel_url
        self.session = session
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: List[Chunk] = []
        
        self.validate_parameters()
    
    def validate_parameters(self) -> None:
        """Validate chunk_size and overlap parameters"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size / 4:
            raise ValueError(f"overlap ({self.overlap}) must be < chunk_size/4 ({self.chunk_size/4:.1f})")
    
    @property
    def global_cases(self) -> List[Dict[str, Any]]:
        """Get all cases as list of dictionaries from current chunks"""
        cases = []
        for chunk in self.chunks:
            if chunk.has_segmentation_result and chunk.cases:
                for case in chunk.cases:
                    cases.append(case.to_dict())
        return cases
    
    @property
    def chunks_processed_count(self) -> int:
        """Get count of chunks that have segmentation results"""
        return sum(1 for chunk in self.chunks if chunk.has_segmentation_result and chunk.cases)
    
    def generate_chunks(self) -> List[Chunk]:
        """Generate chunks for single channel"""
        total_messages = len(self.df_clean)
        
        # Use the channel URL provided in constructor
        channel_url = self.channel_url
                
        # Calculate number of chunks needed
        import math
        num_chunks = math.ceil(total_messages / self.chunk_size)
        
        print(f"        Generating {num_chunks} chunks for single channel")
        
        for i in range(num_chunks):
            # Calculate chunk boundaries using half-open intervals
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, total_messages)
            
            # Create chunk with DataFrame slice
            chunk_messages = self.df_clean.iloc[start_idx:end_idx].copy()
            
            chunk = Chunk(
                chunk_id=i,
                channel_url=channel_url,
                chunk_df=chunk_messages
            )
            
            msg_indices = chunk_messages['msg_ch_idx'].tolist()
            print(f"                Generated chunk {i}: msg_indices [{min(msg_indices)}, {max(msg_indices)}], "
                  f"{len(chunk_messages)} messages, channel: {Utils.format_channel_for_display(channel_url)}")
            self.chunks.append(chunk)
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
        print(f"        📦 Segmenting {len(chunks)} chunks")
        
        chunk_cases = []
        for i, chunk in enumerate(chunks):
            print(f"                Chunk {i+1}/{len(chunks)}: Processing chunk {chunk.chunk_id}")
            
            current_messages = chunk.format_all_messages_for_prompt()
            case_results = chunk.generate_case_segments(
                current_chunk_messages=current_messages,
                llm_client=llm_client
            )
            
            chunk_cases.append(case_results)
        
        print(f"        ✅ Segmentation complete ({len(chunks)} chunks processed)")
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
        print(f"        Processing {len(chunks)} chunks with review")
        
        if not chunks:
            return {"global_cases": [], "total_messages": 0, "chunks_processed": 0}
        
        # Stage 1: 对所有chunks进行case segmentation
        chunk_cases = self.segment_all_chunks(chunks, llm_client)
        
        # Stage 2: 执行case review
        print("        🔍 Performing case boundary review")
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
        print(f"        Processing {len(chunks)} chunks with simple merge")
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

    def classify_all_cases(self, llm_client: 'LLMClient') -> None:
        """Classify all cases across all chunks using LLM"""
        total_cases = sum(len(chunk.cases) if chunk.has_segmentation_result and chunk.cases else 0 for chunk in self.chunks)
        
        if total_cases == 0:
            print("No cases found to classify")
            return
        
        print(f"        🏷️  Classifying {total_cases} cases across {len(self.chunks)} chunks")
        
        processed_cases = 0
        chunk_idx = 0
        
        for chunk in self.chunks:
            if not chunk.has_segmentation_result or not chunk.cases:
                continue
            
            chunk_idx += 1    
            print(f"                Chunk {chunk_idx}: Classifying {len(chunk.cases)} cases")
            
            for case in chunk.cases:
                processed_cases += 1
                try:
                    # Classify the case (updates case object in-place and returns result)
                    classification_result = case.classify_case(llm_client)
                    
                    print(f"                        Case {processed_cases}/{total_cases}: {classification_result.main_category}>{classification_result.sub_category} "
                          f"(conf: {classification_result.confidence:.2f})")
                    
                except Exception as e:
                    print(f"                        Case {processed_cases}/{total_cases}: ❌ Classification failed - {str(e)}")
                    # Keep default "unknown" values in case object
        
        print(f"        ✅ Classification complete ({processed_cases} cases processed)")

    def validate_results(self) -> dict:
        """Validate segmentation results without modifying data"""
        # Use channel URL from constructor
        channel_url = self.channel_url
        
        # Initialize counters
        total_analyzed = len(self.df_clean)
        chunks_processed = 0
        cases_count = 0
        
        print(f"        📊 Validating results")
        
        # Get valid message IDs from df_clean
        valid_msg_ids = set(self.df_clean['msg_ch_idx'].tolist())
        assigned_msg_ids = set()
        assignment_stats = {"out_of_range": 0, "conflicts": 0}
        
        # Count chunks and cases
        for chunk in self.chunks:
            if chunk.has_segmentation_result and chunk.cases:
                chunks_processed += 1
                cases_count += len(chunk.cases)
        
        # Collect all assigned message IDs using global_cases property
        for case_dict in self.global_cases:
            for msg_ch_idx in case_dict.get('msg_index_list', []):
                # Check if message exists in channel data
                if msg_ch_idx not in valid_msg_ids:
                    assignment_stats["out_of_range"] += 1
                    print(f"⚠️  Warning: Message msg_ch_idx {msg_ch_idx} not found in channel data")
                    continue
                
                # Check for conflicts (duplicate assignments)
                if msg_ch_idx in assigned_msg_ids:
                    assignment_stats["conflicts"] += 1
                    case_id = case_dict.get('case_id', 'unknown')
                    print(f"⚠️  Warning: Message {msg_ch_idx} already assigned, conflict with case {case_id}")
                
                assigned_msg_ids.add(msg_ch_idx)
        
        print(f"                Found {cases_count} cases across {total_analyzed} messages")
        
        # Calculate coverage statistics
        assigned_count = len(assigned_msg_ids)
        unassigned_count = total_analyzed - assigned_count
        coverage_rate = assigned_count / total_analyzed * 100 if total_analyzed > 0 else 0
        
        print(f"                📊 Channel Assignment Statistics:")
        print(f"                   Total messages: {total_analyzed}")
        print(f"                   Assigned to cases: {assigned_count} ({coverage_rate:.1f}%)")
        print(f"                   Unassigned: {unassigned_count}")
        print(f"                   Cases generated: {cases_count}")
        print(f"                   Chunks processed: {chunks_processed}")
        
        # Show warnings if any
        if assignment_stats["out_of_range"] > 0:
            print(f"                   ⚠️  Messages not found: {assignment_stats['out_of_range']}")
        if assignment_stats["conflicts"] > 0:
            print(f"                   ⚠️  Conflicts detected: {assignment_stats['conflicts']}")
        
        if coverage_rate == 100.0:
            print(f"                ✅ Perfect coverage achieved for channel!")
        elif coverage_rate >= 95.0:
            print(f"                ✅ Excellent coverage achieved for channel!")
        else:
            print(f"                ⚠️  Coverage could be improved for channel")
        
        # Return validation report
        return {
            "channel_url": channel_url,
            "total_messages": total_analyzed,
            "chunks_processed": chunks_processed,
            "assignment_stats": assignment_stats,
            "coverage_rate": coverage_rate
        }

    def save_results_to_json(self, output_dir: str) -> None:
        """Save channel cases to JSON file"""
        import json
        
        # Create session folder for organized output
        session_folder = os.path.join(output_dir, f"session_{self.session}")
        os.makedirs(session_folder, exist_ok=True)
        
        # Save channel cases to JSON in session folder
        channel_name = Utils.format_channel_for_display(self.channel_url)
        channel_cases_file = os.path.join(session_folder, f"cases_{channel_name}.json")
        save_result = {
            "channel_url": self.channel_url,
            "global_cases": self.global_cases,  # Use property
            "total_messages": len(self.df_clean),
            "chunks_processed": self.chunks_processed_count  # Use property
        }
        
        try:
            with open(channel_cases_file, 'w', encoding='utf-8') as f:
                json.dump(save_result, f, indent=2, ensure_ascii=False)
            print(f"                Channel cases saved to: {channel_cases_file}")
        except IOError as e:
            print(f"                ❌ Error saving JSON file: {e}")
            raise
    
    def save_results_to_csv(self, output_dir: str) -> None:
        """Save annotated messages to CSV file"""
        # Generate annotated CSV for this channel
        df_annotated = self.df_clean.copy()
        df_annotated['case_id'] = "unassigned"  # Default: unassigned (string type)
        # Add classification columns (only main_category and sub_category)
        df_annotated['main_category'] = "unknown"
        df_annotated['sub_category'] = "unknown"
        
        # Map case assignments and classification data using msg_ch_idx
        for case_dict in self.global_cases:  # Use property
            case_id = case_dict.get('case_id', "unknown")
            main_category = case_dict.get('main_category', "unknown")
            sub_category = case_dict.get('sub_category', "unknown")
            
            for msg_ch_idx in case_dict.get('msg_index_list', []):
                mask = df_annotated['msg_ch_idx'] == msg_ch_idx
                df_annotated.loc[mask, 'case_id'] = case_id
                df_annotated.loc[mask, 'main_category'] = main_category
                df_annotated.loc[mask, 'sub_category'] = sub_category
        
        # Create session folder for organized output (same folder as JSON)
        session_folder = os.path.join(output_dir, f"session_{self.session}")
        os.makedirs(session_folder, exist_ok=True)
        
        # Save annotated CSV for this channel in session folder
        channel_name = Utils.format_channel_for_display(self.channel_url)
        channel_segmented_file = os.path.join(session_folder, f"segmented_{channel_name}.csv")
        try:
            df_annotated.to_csv(channel_segmented_file, index=False, encoding='utf-8')
            print(f"                Channel annotated CSV saved to: {channel_segmented_file}")
        except IOError as e:
            print(f"                ❌ Error saving CSV file: {e}")
            raise