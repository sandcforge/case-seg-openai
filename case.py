#!/usr/bin/env python3
"""
Case data models and structures for customer support message segmentation.

This module contains:
- Case data structures and related metadata
- Pydantic models for LLM structured output
- Case review and processing models
"""

import pandas as pd  # type: ignore
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

# Pydantic imports for LLM-compatible classes
from pydantic import BaseModel, Field  # type: ignore


# ----------------------------
# Case Data Structures
# ----------------------------

@dataclass
class MetaInfo:
    """Meta information structure for each case"""
    tracking_numbers: List[str] = field(default_factory=list)
    order_numbers: List[str] = field(default_factory=list)
    user_names: List[str] = field(default_factory=list)


@dataclass
class Case:
    """Individual case structure for case segmentation output"""
    case_id: Optional[str] = None  # Case ID (assigned during processing)
    msg_index_list: List[int] = field(default_factory=list)  # List of msg_ch_idx values
    messages: Optional['pd.DataFrame'] = None  # Related messages DataFrame
    summary: str = "N/A"
    status: str = "ongoing"  # open | ongoing | resolved | blocked
    pending_party: str = "N/A"  # seller|platform|N/A
    confidence: float = 0.0
    meta: Optional[MetaInfo] = None
    
    def __post_init__(self):
        """Initialize meta if not provided"""
        if self.meta is None:
            self.meta = MetaInfo()
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'case_id': self.case_id,
            'msg_index_list': self.msg_index_list,  # Now just a list of integers
            'summary': self.summary,
            'status': self.status,
            'pending_party': self.pending_party,
            'confidence': self.confidence,
            'meta': {
                'tracking_numbers': self.meta.tracking_numbers,
                'order_numbers': self.meta.order_numbers,
                'user_names': self.meta.user_names
            } if self.meta else {}
        }


# ----------------------------
# LLM-Compatible Models
# ----------------------------

class CaseItemForLLM(BaseModel):
    """LLM-compatible case structure using List[int] for msg_index_list"""
    model_config = {"extra": "forbid"}
    
    msg_index_list: List[int]  # List of message indices instead of DataFrame
    summary: str
    status: str  # open | ongoing | resolved | blocked
    pending_party: str  # seller|platform|N/A
    confidence: float
    meta: MetaInfo


class CasesSegmentationResponseForLLM(BaseModel):
    """LLM-compatible response structure for case segmentation"""
    model_config = {"extra": "forbid"}  # Ensures additionalProperties: false
    
    complete_cases: List[CaseItemForLLM]


# ----------------------------
# Case Review Models
# ----------------------------

class CaseReviewInput(BaseModel):
    """Input structure for case review"""
    model_config = {"extra": "forbid"}
    cases: List[CaseItemForLLM] = Field(..., description="相关的cases列表")
    overlap_msg_ids: List[int] = Field(..., description="重叠区域的消息ID")
    all_messages: str = Field(..., description="所有相关消息的文本")


class ReviewAction(BaseModel):
    """Single review action"""
    model_config = {"extra": "forbid"}
    action_type: Literal["merge", "split", "adjust_boundary", "no_change"] = Field(..., description="操作类型")
    target_cases: List[int] = Field(..., description="目标case的索引")
    new_msg_assignment: Dict[int, int] = Field(..., description="新的消息分配 {msg_id: case_index}")
    reason: str = Field(..., description="操作原因")


class CaseReviewResponse(BaseModel):
    """Response structure for case review"""
    model_config = {"extra": "forbid"}
    review_actions: List[ReviewAction] = Field(..., description="review操作列表")
    updated_cases: List[CaseItemForLLM] = Field(..., description="更新后的cases")
    confidence: float = Field(..., description="review结果的置信度", ge=0.0, le=1.0)