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
from typing import List, Optional, Dict, Any, Literal, TYPE_CHECKING
from datetime import datetime

# Pydantic imports for LLM-compatible classes
from pydantic import BaseModel, Field  # type: ignore

# Import Utils for message formatting
from utils import Utils

if TYPE_CHECKING:
    from llm_client import LLMClient


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
    segmentation_confidence: float = 0.0
    meta: Optional[MetaInfo] = None
    channel_url: str = ""  # Channel URL this case belongs to
    # Classification fields
    main_category: str = "unknown"  # 主分类
    sub_category: str = "unknown"  # 子分类
    classification_reasoning: str = "N/A"  # 分类理由
    classification_confidence: float = 0.0  # 分类置信度
    classification_indicators: List[str] = field(default_factory=list)  # 关键指标
    has_classification: bool = False  # 是否已完成分类
    # SOP fields
    has_sop: bool = False  # 是否已找到相关SOP
    sop_content: str = "N/A"  # SOP内容
    sop_url: str = "N/A"  # SOP链接
    sop_score: float = 0.0  # SOP评分

    def __post_init__(self):
        """Initialize meta if not provided"""
        if self.meta is None:
            self.meta = MetaInfo()
    
    @property
    def global_msg_id_list(self) -> List[str]:
        """Get list of Message IDs from the messages DataFrame"""
        if self.messages is None or self.messages.empty:
            return []
        if 'Message ID' not in self.messages.columns:
            return []
        return self.messages['Message ID'].tolist()
    
    @property
    def start_time(self) -> Optional[str]:
        """Get start time from first message in the case"""
        if self.messages is None or self.messages.empty:
            return None
        if 'Created Time' not in self.messages.columns:
            return None
        df_sorted = self.messages.sort_values('Created Time')
        first_time = pd.to_datetime(df_sorted.iloc[0]['Created Time'])
        return first_time.strftime('%Y-%m-%d %H:%M:%S')
    
    @property
    def end_time(self) -> Optional[str]:
        """Get end time from last message in the case"""
        if self.messages is None or self.messages.empty:
            return None
        if 'Created Time' not in self.messages.columns:
            return None
        df_sorted = self.messages.sort_values('Created Time')
        last_time = pd.to_datetime(df_sorted.iloc[-1]['Created Time'])
        return last_time.strftime('%Y-%m-%d %H:%M:%S')
    
    @property
    def usr_msg_num(self) -> int:
        """Count of user messages (non-customer_service role)"""
        if self.messages is None or self.messages.empty:
            return -1
        if 'role' not in self.messages.columns:
            return -1
        return len(self.messages[self.messages['role'] != 'customer_service'])
    
    @property
    def total_msg_num(self) -> int:
        """Total count of messages"""
        if self.messages is None or self.messages.empty:
            return -1
        return len(self.messages)
    
    @property
    def handle_time(self) -> int:
        """Time between first and last message in minutes"""
        if self.messages is None or self.messages.empty:
            return -1
        if 'Created Time' not in self.messages.columns:
            return -1
        df_sorted = self.messages.sort_values('Created Time')
        first_time = pd.to_datetime(df_sorted.iloc[0]['Created Time'])
        last_time = pd.to_datetime(df_sorted.iloc[-1]['Created Time'])
        return int((last_time - first_time).total_seconds() / 60)
    
    @property
    def first_res_time(self) -> int:
        """Support response time in minutes, -1 if not applicable"""
        if self.messages is None or self.messages.empty:
            return -1
        if 'Created Time' not in self.messages.columns or 'role' not in self.messages.columns:
            return -1
        
        df_sorted = self.messages.sort_values('Created Time')
        if len(df_sorted) == 0:
            return -1
        
        first_message = df_sorted.iloc[0]
        
        # If support initiates conversation, return -1
        if first_message['role'] == 'customer_service':
            return -1
        
        # First message is from user, find first support response
        support_messages = df_sorted[df_sorted['role'] == 'customer_service']
        
        if len(support_messages) > 0:
            # Found support response
            first_user_time = pd.to_datetime(first_message['Created Time'])
            first_support_time = pd.to_datetime(support_messages.iloc[0]['Created Time'])
            return int((first_support_time - first_user_time).total_seconds() / 60)
        else:
            # No support response, use handle_time
            return self.handle_time
    
    @property
    def first_contact_resolution(self) -> int:
        """1=resolved within 8h, 0=not, -1=not processed"""
        if self.messages is None or self.messages.empty:
            return -1
        
        handle_time_val = self.handle_time
        if handle_time_val == -1:
            return -1
        
        # Check if resolved within 8 hours (480 minutes)
        if self.status == "resolved" and handle_time_val <= 480:
            return 1
        else:
            return 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'case_id': self.case_id,
            'msg_index_list': self.msg_index_list,  # Now just a list of integers
            'global_msg_id_list': self.global_msg_id_list,  # List of Message IDs
            'summary': self.summary,
            'status': self.status,
            'pending_party': self.pending_party,
            'segmentation_confidence': self.segmentation_confidence,
            'channel_url': self.channel_url,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'main_category': self.main_category,
            'sub_category': self.sub_category,
            'classification_reasoning': self.classification_reasoning,
            'classification_confidence': self.classification_confidence,
            'classification_indicators': self.classification_indicators,
            'has_classification': self.has_classification,
            'has_sop': self.has_sop,
            'sop_content': self.sop_content,
            'sop_url': self.sop_url,
            'sop_score': self.sop_score,
            'first_res_time': self.first_res_time,
            'handle_time': self.handle_time,
            'first_contact_resolution': self.first_contact_resolution,
            'usr_msg_num': self.usr_msg_num,
            'total_msg_num': self.total_msg_num,
            'meta': {
                'tracking_numbers': self.meta.tracking_numbers,
                'order_numbers': self.meta.order_numbers,
                'user_names': self.meta.user_names
            } if self.meta else {}
        }
    
    def classify_case(self, llm_client: 'LLMClient') -> 'CaseClassificationLLMRes':
        """Classify the case using LLM based on the messages DataFrame"""
        if self.messages is None or self.messages.empty:
            raise ValueError("Cannot classify case: no messages available")
        
        # Format all messages in this case using the Utils method
        messages_text = Utils.format_messages_for_prompt(self.messages)
        
        # Load the classification prompt template
        try:
            prompt_template = llm_client.load_prompt("case_classification_prompt.md")
        except FileNotFoundError as e:
            raise RuntimeError(f"Cannot load case classification prompt: {e}")
        
        # Replace placeholders in the template
        classification_prompt = prompt_template.replace(
            "<<<INSERT_CASE_SUMMARY>>>", self.summary
        ).replace(
            "<<<INSERT_CASE_STATUS>>>", self.status
        ).replace(
            "<<<INSERT_FORMATTED_MESSAGES>>>", messages_text
        )

        # Use structured output for classification
        call_label = f"case_classification_{self.case_id}"
        
        classification_response = llm_client.generate_structured(
            classification_prompt,
            CaseClassificationLLMRes,
            call_label=call_label
        )
        
        # Update case object fields in-place with classification results
        # classification_response is a dict, not a Pydantic model
        self.main_category = classification_response['main_category']
        self.sub_category = classification_response['sub_category']
        self.classification_reasoning = classification_response['reasoning']
        self.classification_confidence = classification_response['confidence']
        self.classification_indicators = classification_response['key_indicators']
        self.has_classification = True

        return classification_response

    def find_sop(self) -> Dict[str, Any]:
        """
        Find the most relevant SOP for this case by calling Aloy API

        Returns:
            API response dict containing SOP information

        Raises:
            ValueError: If no messages available
            RuntimeError: If API call fails
        """
        if self.messages is None or self.messages.empty:
            raise ValueError("Cannot find SOP: no messages available")

        # Format messages as chat logs string
        chat_logs = Utils.format_messages_for_prompt(self.messages)

        # Call SOP API (returns parsed dict with sop_content, sop_url, sop_score)
        data = Utils.call_sop_api(chat_logs)

        # Update case fields with parsed data
        if data.get('sop_content') and data.get('sop_content') != 'N/A':
            self.has_sop = True
            self.sop_content = data.get('sop_content', 'N/A')
            self.sop_url = data.get('sop_url', 'N/A')

            # Convert sop_score to float
            try:
                self.sop_score = float(data.get('sop_score', 0.0))
            except (ValueError, TypeError):
                self.sop_score = 0.0
        else:
            self.has_sop = False
            self.sop_content = 'N/A'
            self.sop_url = 'N/A'
            self.sop_score = 0.0

        return data


# ----------------------------
# LLM-Compatible Models
# ----------------------------

class CaseSegmentationLLMRes(BaseModel):
    """LLM-compatible case structure using List[int] for msg_index_list"""
    model_config = {"extra": "forbid"}
    
    msg_index_list: List[int]  # List of message indices instead of DataFrame
    summary: str
    status: str  # open | ongoing | resolved | blocked
    pending_party: str  # seller|platform|N/A
    segmentation_confidence: float
    meta: MetaInfo


class CasesSegmentationListLLMRes(BaseModel):
    """LLM-compatible response structure for case segmentation"""
    model_config = {"extra": "forbid"}  # Ensures additionalProperties: false
    
    complete_cases: List[CaseSegmentationLLMRes]


# ----------------------------
# Case Review Models
# ----------------------------

class CaseReviewInput(BaseModel):
    """Input structure for case review"""
    model_config = {"extra": "forbid"}
    cases: List[CaseSegmentationLLMRes] = Field(..., description="相关的cases列表")
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
    updated_cases: List[CaseSegmentationLLMRes] = Field(..., description="更新后的cases")
    review_confidence: float = Field(..., description="review结果的置信度", ge=0.0, le=1.0)


# ----------------------------
# Case Classification Models
# ----------------------------

class CaseClassificationLLMRes(BaseModel):
    """Response structure for case classification"""
    model_config = {"extra": "forbid"}
    main_category: str = Field(..., description="主分类 (Order, Shipment, Payment, Product, Account, Technical)")
    sub_category: str = Field(..., description="子分类")
    reasoning: str = Field(..., description="分类理由")
    confidence: float = Field(..., description="分类置信度", ge=0.0, le=1.0)
    key_indicators: List[str] = Field(..., description="关键指标词汇或短语")