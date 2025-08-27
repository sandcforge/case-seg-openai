#!/usr/bin/env python3
"""
Chunk processing module for customer support message segmentation.

This module contains:
- Chunk class for processing message chunks and case segmentation
- Message repair and validation logic
"""

import pandas as pd  # type: ignore
from dataclasses import dataclass, field
from typing import List, Dict, Any, TYPE_CHECKING
from datetime import datetime
from utils import Utils

# Import Case-related classes from the new case module
from case import Case, MetaInfo, CasesSegmentationListLLMRes

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from llm_client import LLMClient


# ----------------------------
# Chunk Class
# ----------------------------

@dataclass
class Chunk:
    """Data structure for a single chunk of messages"""
    chunk_id: int                    # Sequential chunk ID (0, 1, 2, ...)
    channel_url: str                 # Channel this chunk belongs to
    chunk_df: pd.DataFrame           # DataFrame slice with messages in this chunk
    has_segmentation_result: bool = False                    # Whether segmentation has been completed
    cases: List[Case] = field(default_factory=list)  # Cached segmentation results
    
    @property
    def total_messages(self) -> int:
        """Number of messages in this chunk (calculated from DataFrame length)"""
        return len(self.chunk_df)

    def get_message_indices(self) -> List[int]:
        """Get list of msg_ch_idx values for messages in this chunk"""
        return self.chunk_df['msg_ch_idx'].tolist()
    
    
    def format_all_messages_for_prompt(self) -> str:
        """Format chunk messages as: message_index | sender id | role | timestamp | text"""
        return self.format_messages_for_prompt(self.chunk_df)
    
    @staticmethod
    def format_messages_for_prompt(chunk_df: pd.DataFrame) -> str:
        """Format chunk messages for LLM prompt: message_index | sender id | role | timestamp | text"""
        formatted_lines = []
        for _, row in chunk_df.iterrows():
            formatted_lines.append(Utils.format_one_msg_for_prompt(row))
        return '\n'.join(formatted_lines)
    
    @staticmethod
    def generate_case_segments(chunk_id: int,
                             channel_url: str, 
                             current_chunk_messages: str, 
                             llm_client: 'LLMClient') -> List[Dict[str, Any]]:
        """Generate case segments using LLM for current chunk messages"""
        # Load the segmentation prompt template
        try:
            prompt_template = llm_client.load_prompt("segmentation_prompt.md")
        except FileNotFoundError as e:
            raise RuntimeError(f"Cannot load segmentation prompt: {e}")
        
        final_prompt = prompt_template.replace(
            "<<<INSERT_CHUNK_BLOCK_HERE>>>", 
            current_chunk_messages
        )
        
        # Generate case segments using LLM
        try:
            # Use structured output for OpenAI models, fallback to JSON parsing for Claude
            if llm_client.provider == "openai" and CasesSegmentationListLLMRes:
                # Structured output with LLM-compatible schema (uses List[int] for msg_list)
                # Generate contextual call label with timestamp
                channel_name = Utils.format_channel_for_display(channel_url)
                call_label = f"case_segmentation_{channel_name}_chunk_{chunk_id}"
                
                structured_response = llm_client.generate_structured(
                    final_prompt, 
                    CasesSegmentationListLLMRes, 
                    call_label=call_label
                )
                
            # Convert LLM response to dict format and return raw cases
            raw_cases = [case.model_dump() for case in structured_response.complete_cases]
            return raw_cases
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate case segments for chunk {chunk_id}: {e}")

