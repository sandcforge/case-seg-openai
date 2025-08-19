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
import pandas as pd # type: ignore
import pytz # type: ignore
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Set, Tuple
from dotenv import load_dotenv # type: ignore
import anthropic # type: ignore
from datetime import datetime
import copy
from collections import defaultdict

# Pydantic and OpenAI imports
from typing import List, Optional, Literal, Tuple
from pydantic import BaseModel, Field  # type: ignore

import openai # type: ignore

# Load environment variables
load_dotenv()


# ----------------------------
# Pydantic Models for Structured Output
# ----------------------------


class CaseItem(BaseModel):
    """Individual case structure for case segmentation output"""
    model_config = {"extra": "forbid"}  # Ensures additionalProperties: false
    
    msg_list: List[int]
    summary: str
    status: str  # open | ongoing | resolved | blocked
    pending_party: str  # seller | platform | N/A
    last_update: str  # ISO timestamp or N/A
    is_active_case: bool
    confidence: float

class CasesSegmentationResponse(BaseModel):
    """Complete response structure for case segmentation"""
    model_config = {"extra": "forbid"}  # Ensures additionalProperties: false
    
    complete_cases: List[CaseItem]
    total_messages_analyzed: int
    llm_duration_seconds: Optional[float] = None


class CaseAnchorRules(BaseModel):
    """Case anchor rules structure"""
    model_config = {"extra": "forbid"}
    priority_order: str = Field(..., description="优先级规则说明")
    multi_order_rule: str = Field(..., description="多订单处理规则")
    default_scope_rules: str = Field(..., description="默认范围规则")

class AnchorInfo(BaseModel):
    """Anchor information structure"""
    model_config = {"extra": "forbid"}
    tracking: List[str] = Field(..., description="物流单号列表")
    order_ids: List[str] = Field(..., description="订单号列表")
    buyers: List[str] = Field(..., description="买家标识列表")
    carrier: Literal["UPS", "FedEx", "USPS", "N/A"] = Field(
        ..., description="承运商"
    )

class Amounts(BaseModel):
    """Monetary amounts involved in the case"""
    model_config = {"extra": "forbid"}
    credit_to_seller: Optional[float] = Field(
        None, description="需要划给卖家的信用/补偿金额"
    )
    refund_to_buyer: Optional[float] = Field(
        None, description="需要退还给买家的金额"
    )

class ActiveCaseHint(BaseModel):
    """Individual active case hint structure"""
    model_config = {"extra": "forbid"}
    topic: str
    program: str
    scope: str
    anchor: AnchorInfo
    status: str
    shipping_state: str
    last_action: str
    last_update: str  # 建议使用 ISO8601 字符串或直接 datetime
    pending_party: str
    amounts: Amounts  # ← 由 Dict 改为定形对象，避免 schema 不一致
    returns_to_previous_topic: bool
    possible_new_session: bool
    keywords: List[str]
    evidence_msg_ch_idx: List[int]

class TimeWindow(BaseModel):
    """Start/End window as ISO8601 strings"""
    model_config = {"extra": "forbid"}
    start_iso: str = Field(..., description="开始时间，ISO8601")
    end_iso: str = Field(..., description="结束时间，ISO8601")


class MetaInfo(BaseModel):
    """Meta information structure"""
    model_config = {"extra": "forbid"}
    overlap: int
    channel: str
    time_window: TimeWindow

class GuidanceInfo(BaseModel):
    """Guidance information structure"""
    model_config = {"extra": "forbid"}
    role_normalization: str
    pronoun_resolution: str
    carrier_detection: str
    resolved_status_rule: str

class TailSummaryResponse(BaseModel):
    """Complete response structure for tail summary generation"""
    model_config = {"extra": "forbid"}  # => additionalProperties=false
    case_anchor_rules: CaseAnchorRules
    active_case_hints: List[ActiveCaseHint]
    meta: MetaInfo
    guidance: GuidanceInfo

# ----------------------------
# Repair Chunk Output Utilities
# ----------------------------


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



def format_channel_for_display(channel_url: str) -> str:
    """
    Format channel URL for display: show channel_ + last 5 characters
    Example: sendbird_group_channel_215482988_b374305ff3e440674e786d63916f1d5aacda8249 -> channel_da8249
    """
    if len(channel_url) <= 5:
        return channel_url
    return f"channel_{channel_url[-5:]}"


def validate_global_assignment(
    global_cases: List[Dict[str, Any]], 
    total_messages: int,
    channel_name: str = ""
) -> Dict[str, Any]:
    """
    验证全局案例分配的完整性：
    - 检查消息覆盖率（0遗漏）
    - 检查重复分配（0重复）
    - 生成详细的分配报告
    
    Args:
        global_cases: 全局cases列表
        total_messages: 期望的总消息数
        channel_name: 频道名称（用于日志）
    
    Returns:
        验证报告字典
    """
    print(f"\n🔍 Validating global assignment for {channel_name}...")
    
    # 收集所有已分配的消息
    all_assigned_msgs = []
    case_stats = []
    
    for case_idx, case in enumerate(global_cases):
        msg_list = case.get("msg_list", [])
        all_assigned_msgs.extend(msg_list)
        
        case_stats.append({
            "global_case_id": case.get("global_case_id", case_idx),
            "msg_count": len(msg_list),
            "msg_range": f"[{min(msg_list) if msg_list else 'N/A'}, {max(msg_list) if msg_list else 'N/A'}]",
            "summary_preview": case.get("summary", "")[:50] + "..." if len(case.get("summary", "")) > 50 else case.get("summary", ""),
            "status": case.get("status", "unknown")
        })
    
    # 分析分配情况
    assigned_set = set(all_assigned_msgs)
    expected_set = set(range(total_messages))
    
    # 检查重复分配
    duplicates = []
    msg_count = defaultdict(int)
    for msg in all_assigned_msgs:
        msg_count[msg] += 1
        if msg_count[msg] > 1:
            duplicates.append(msg)
    
    # 检查遗漏
    missing = sorted(list(expected_set - assigned_set))
    
    # 检查超出范围
    out_of_range = sorted([msg for msg in assigned_set if msg >= total_messages or msg < 0])
    
    # 生成统计
    coverage_rate = len(assigned_set & expected_set) / total_messages * 100 if total_messages > 0 else 0
    
    report = {
        "channel": channel_name,
        "total_cases": len(global_cases),
        "total_messages_expected": total_messages,
        "total_messages_assigned": len(all_assigned_msgs),
        "unique_messages_assigned": len(assigned_set),
        "coverage_rate": round(coverage_rate, 2),
        "duplicates": {
            "count": len(set(duplicates)),
            "messages": sorted(list(set(duplicates)))[:20],  # 限制显示数量
            "total_duplicate_assignments": len(duplicates)
        },
        "missing": {
            "count": len(missing),
            "messages": missing[:20]  # 限制显示数量
        },
        "out_of_range": {
            "count": len(out_of_range),
            "messages": out_of_range[:10]
        },
        "case_stats": case_stats,
        "is_valid": len(missing) == 0 and len(set(duplicates)) == 0 and len(out_of_range) == 0
    }
    
    # 打印报告
    print(f"  📊 Cases: {report['total_cases']}")
    print(f"  📈 Coverage: {report['coverage_rate']:.1f}% ({report['unique_messages_assigned']}/{report['total_messages_expected']})")
    
    if report['duplicates']['count'] > 0:
        print(f"  ⚠️  Duplicates: {report['duplicates']['count']} messages, {report['duplicates']['total_duplicate_assignments']} total assignments")
        print(f"     Sample: {report['duplicates']['messages'][:5]}")
    
    if report['missing']['count'] > 0:
        print(f"  ❌ Missing: {report['missing']['count']} messages")
        print(f"     Sample: {report['missing']['messages'][:5]}")
    
    if report['out_of_range']['count'] > 0:
        print(f"  🚫 Out of range: {report['out_of_range']['count']} messages")
        print(f"     Sample: {report['out_of_range']['messages'][:5]}")
    
    if report['is_valid']:
        print(f"  ✅ Validation PASSED - Perfect assignment!")
    else:
        print(f"  ❌ Validation FAILED - Assignment issues detected")
    
    return report


def repair_global_assignment(
    global_cases: List[Dict[str, Any]], 
    total_messages: int,
    channel_name: str = ""
) -> List[Dict[str, Any]]:
    """
    修复全局分配中的问题：
    - 去除重复分配（保留第一个分配）
    - 将遗漏消息分配到合适的case或创建misc case
    
    Args:
        global_cases: 需要修复的全局cases
        total_messages: 总消息数
        channel_name: 频道名称
        
    Returns:
        修复后的全局cases
    """
    print(f"\n🔧 Repairing global assignment for {channel_name}...")
    
    # 深拷贝避免修改原数据
    repaired_cases = copy.deepcopy(global_cases)
    
    # Step 1: 去重 - 保留第一次出现的分配
    msg_to_first_case = {}
    removals = []
    
    for case_idx, case in enumerate(repaired_cases):
        msg_list = case.get("msg_list", [])
        keep_msgs = []
        
        for msg in msg_list:
            if msg in msg_to_first_case:
                # 重复分配，记录移除
                removals.append({
                    "msg": msg,
                    "from_case": case_idx,
                    "kept_in_case": msg_to_first_case[msg]
                })
            else:
                # 首次分配，保留
                msg_to_first_case[msg] = case_idx
                keep_msgs.append(msg)
        
        case["msg_list"] = sorted(keep_msgs)
    
    if removals:
        print(f"  🔧 Removed {len(removals)} duplicate assignments")
    
    # Step 2: 处理遗漏消息
    assigned_msgs = set(msg_to_first_case.keys())
    expected_msgs = set(range(total_messages))
    missing_msgs = sorted(list(expected_msgs - assigned_msgs))
    
    if missing_msgs:
        print(f"  🔧 Found {len(missing_msgs)} missing messages")
        
        # 创建misc case处理遗漏消息
        misc_case = {
            "global_case_id": max([c.get("global_case_id", i) for i, c in enumerate(repaired_cases)], default=-1) + 1,
            "msg_list": missing_msgs,
            "summary": f"auto_repaired: {len(missing_msgs)} previously unassigned messages",
            "status": "open",
            "pending_party": "N/A",
            "last_update": "N/A",
            "is_active_case": False,
            "confidence": 0.2,
            "anchors": {}
        }
        
        repaired_cases.append(misc_case)
        print(f"  ➕ Created repair case {misc_case['global_case_id']} for {len(missing_msgs)} missing messages")
    
    # Step 3: 清理空cases
    non_empty_cases = [case for case in repaired_cases if case.get("msg_list")]
    removed_empty = len(repaired_cases) - len(non_empty_cases)
    
    if removed_empty > 0:
        print(f"  🧹 Removed {removed_empty} empty cases")
    
    # Step 4: 重新分配global_case_id确保连续性
    for i, case in enumerate(non_empty_cases):
        case["global_case_id"] = i
    
    print(f"  ✅ Repair completed: {len(non_empty_cases)} final cases")
    
    return non_empty_cases


@dataclass
class Chunk:
    """Data structure for a single chunk of messages"""
    chunk_id: int                    # Sequential chunk ID (0, 1, 2, ...)
    channel_url: str                 # Channel this chunk belongs to
    start_idx: int                   # Start index in the channel (inclusive)
    end_idx: int                     # End index in the channel (exclusive) - half-open interval [start, end)
    messages: pd.DataFrame           # DataFrame slice with messages in this chunk
    has_overlap_with_previous: bool  # Whether this chunk overlaps with previous chunk
    overlap_size: int                # Number of overlapping messages with previous chunk
    tail_summary: Optional[str] = None  # Generated tail summary for next chunk
    
    @property
    def total_messages(self) -> int:
        """Number of messages in this chunk (calculated from end_idx - start_idx)"""
        return self.end_idx - self.start_idx

    def get_message_indices(self) -> List[int]:
        """Get list of msg_ch_idx values for messages in this chunk"""
        return self.messages['msg_ch_idx'].tolist()
    
    def format_one_msg_for_prompt(self, row) -> str:
        """Format a single message row as: msg_ch_idx | sender_id | role | timestamp | text"""
        # Handle NaN messages and replace newlines with spaces to keep one line per message
        message_text = str(row['Message']).replace('\n', ' ').replace('\r', ' ')
        if message_text == 'nan':
            message_text = ''
        
        return f"{row['msg_ch_idx']} | {row['Sender ID']} | {row['role']} | {row['Created Time']} | {message_text}"
    
    def format_all_messages_for_prompt(self) -> str:
        """Format chunk messages as: message_index | sender id | role | timestamp | text"""
        formatted_lines = []
        for _, row in self.messages.iterrows():
            formatted_lines.append(self.format_one_msg_for_prompt(row))
        return '\n'.join(formatted_lines)
    
    def generate_tail_summary(self, 
                            current_messages: str,
                            overlap_size: int,
                            llm_client: 'LLMClient', 
                            previous_context: str = "") -> Dict[str, Any]:
        """Generate tail summary using LLM for the next chunk"""
        # Load the prompt template
        try:
            prompt_template = llm_client.load_prompt("tail_summary_prompt.md")
        except FileNotFoundError as e:
            raise RuntimeError(f"Cannot load tail summary prompt: {e}")
        
        # Get time window from messages
        if len(self.messages) > 0:
            start_time = str(self.messages.iloc[0]['Created Time'])
            end_time = str(self.messages.iloc[-1]['Created Time'])
        else:
            start_time = "N/A"
            end_time = "N/A"
        
        # Prepare previous context JSON (default to empty object if none provided)
        previous_context_json = previous_context if previous_context else "{}"
        
        # Replace placeholders in the new prompt format
        final_prompt = prompt_template.replace(
            "{PUT_PREVIOUS_CONTEXT_SUMMARY_JSON_HERE}", 
            previous_context_json
        ).replace(
            "PUT_CURRENT_CHUNK_MESSAGE_LINES_HERE", 
            current_messages
        ).replace(
            "PUT_OVERLAP_INT_HERE", 
            str(overlap_size)
        ).replace(
            "PUT_CHANNEL_ID_OR_URL_OR_NA_HERE", 
            self.channel_url
        ).replace(
            "PUT_START_ISO", 
            start_time
        ).replace(
            "PUT_END_ISO", 
            end_time
        )
        
        # Generate tail summary using LLM with structured output
        try:
            # Use structured output for OpenAI models
            structured_response = llm_client.generate_structured(
                final_prompt, 
                TailSummaryResponse, 
                call_label="tail_summary"
            )
            # Convert Pydantic response to dict for compatibility
            result = structured_response.model_dump()
            
            # Store as JSON string for later use (if needed)
            import json
            self.tail_summary = json.dumps(result, ensure_ascii=False)
            
            # Return the dict directly
            return result
                
        except Exception as e:
            raise RuntimeError(f"Failed to generate tail summary for chunk {self.chunk_id}: {e}")

    def generate_case_segments(self, 
                             current_chunk_messages: str, 
                             previous_chunk_tail_summary: Optional[Dict[str, Any]], 
                             llm_client: 'LLMClient') -> Dict[str, Any]:
        """Generate case segments using LLM for current chunk messages"""
        # Load the segmentation prompt template
        try:
            prompt_template = llm_client.load_prompt("segmentation_prompt.md")
        except FileNotFoundError as e:
            raise RuntimeError(f"Cannot load segmentation prompt: {e}")
        
        # Replace placeholders in prompt template
        # Handle previous context (now dict format from generate_tail_summary)
        if previous_chunk_tail_summary is None:
            context_text = "No previous context"
        else:
            # Convert dict to JSON string format for prompt
            import json
            if isinstance(previous_chunk_tail_summary, dict):
                context_text = json.dumps(previous_chunk_tail_summary, ensure_ascii=False, indent=2)
            else:
                # Backwards compatibility for string format
                context_text = str(previous_chunk_tail_summary)
        
        final_prompt = prompt_template.replace(
            "<<<INSERT_PREVIOUS_CONTEXT_SUMMARY_BLOCK_HERE>>>", 
            context_text
        ).replace(
            "<<<INSERT_CHUNK_BLOCK_HERE>>>", 
            current_chunk_messages
        )
        
        # Generate case segments using LLM
        try:
            # Use structured output for OpenAI models, fallback to JSON parsing for Claude
            if llm_client.provider == "openai" and CasesSegmentationResponse:
                # Structured output with Pydantic schema
                structured_response = llm_client.generate_structured(
                    final_prompt, 
                    CasesSegmentationResponse, 
                    call_label="case_segmentation"
                )
                # Convert Pydantic response to dict for compatibility
                result = structured_response.model_dump()
                
            
            # 使用 previous context (现在是字典)
            prev_context = None
            if previous_chunk_tail_summary and isinstance(previous_chunk_tail_summary, dict):
                if "active_case_hints" in previous_chunk_tail_summary:
                    prev_context = {"ACTIVE_CASE_HINTS": previous_chunk_tail_summary["active_case_hints"]}
            
            # 直接使用修复函数
            repair_result = self.repair_case_segment_output(
                cases=result.get('complete_cases', []),
                prev_context=prev_context
            )
            
            # 更新result结构
            result['complete_cases'] = repair_result['cases_out']
            
            # 报告修复情况
            report = repair_result['report']
            provisionals = repair_result['provisionals']
            
            if provisionals:
                print(f"🔧 Applied {len(provisionals)} repair actions for chunk {self.chunk_id}:")
                for prov in provisionals:
                    if prov['type'] == 'duplicate_resolution':
                        print(f"  ➜ Resolved duplicate msg {prov['msg_idx']}: kept in case {prov['chosen_case']}")
                    elif prov['type'] == 'auto_attach':
                        print(f"  ➕ Auto-attached msg {prov['msg_idx']} to case {prov['attached_to']}")
                    elif prov['type'] == 'misc_bucket':
                        print(f"  📦 Created misc case for {len(prov['msg_idxs'])} unassigned messages")
            
            # 最终验证
            if report['missing_msgs'] == 0 and report['duplicates_after'] == 0:
                print(f"✅ Chunk {self.chunk_id} repair completed: 100% coverage achieved")
                print(f"   Final: {report['covered_msgs']}/{report['total_msgs']} messages in {report['total_cases_out']} cases")
            else:
                print(f"⚠️ Chunk {self.chunk_id} repair incomplete:")
                print(f"   Missing: {report['missing_msgs']}, Duplicates: {report['duplicates_after']}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate case segments for chunk {self.chunk_id}: {e}")

    def repair_case_segment_output(self, 
                                 cases: List[Dict[str, Any]], 
                                 prev_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        对单个 chunk 的 LLM 分段结果进行修复 & 校验（不修改入参）。
        - 去重：同一 msg 出现在多个 case，只保留一个（可解释的择一规则）
        - 未分配：必须挂靠到合理的 case（空消息优先挂靠到相同sender的最近消息）
        - 补齐字段、排序稳定、自检报告

        Args
        ----
        cases : List[Dict]      # LLM 输出的 cases
        prev_context : Optional[Dict]  # 上一块尾部摘要，用于承接判断

        Returns
        -------
        {
          "cases_out": List[Dict],
          "provisionals": List[Dict],   # 去重/自动挂靠的记录，便于后续复核
          "report": { ... }             # 自检统计
        }
        """
        # 内部helper函数
        def _is_empty_message(msg_idx: int) -> bool:
            """检查消息内容是否为空或空白"""
            if msg_idx >= len(self.messages):
                return True
            message = self.messages.iloc[msg_idx]
            text = str(message.get('Text', '')).strip()
            return len(text) == 0
        
        def _find_nearest_same_sender_case(msg_idx: int, cases: List[Dict]) -> Optional[int]:
            """查找包含最近的相同sender_id消息的case"""
            if msg_idx >= len(self.messages):
                return None
            
            target_sender = self.messages.iloc[msg_idx].get('Sender ID', '')
            if not target_sender:
                return None
            
            # 构建消息到case的映射
            msg_to_case = {}
            for case_idx, case in enumerate(cases):
                for msg_id in case.get('msg_list', []):
                    msg_to_case[msg_id] = case_idx
            
            # 寻找最近的相同sender消息
            best_distance = float('inf')
            best_case_id = None
            
            for check_msg_idx, case_idx in msg_to_case.items():
                if check_msg_idx < len(self.messages):
                    check_sender = self.messages.iloc[check_msg_idx].get('Sender ID', '')
                    if check_sender == target_sender:
                        distance = abs(check_msg_idx - msg_idx)
                        if distance < best_distance:
                            best_distance = distance
                            best_case_id = case_idx
            
            return best_case_id
        
        def _attach_unassigned_smart(msg_idx: int, cases: List[Dict]) -> Optional[int]:
            """智能挂靠逻辑（基于原_attach_unassigned_simple）"""
            scored = []
            for cid, c in enumerate(cases):
                # 仅考虑 active 的或 open/ongoing/blocked 的
                if not c.get("is_active_case", False) and c.get("status") == "resolved":
                    continue
                scored.append((
                    1 if self._hits_active_hints(c, prev_context) else 0,
                    self._anchor_strength(c),
                    self._proximity_score(msg_idx, c),
                    float(c.get("confidence", 0.0)),
                    -cid,
                    cid
                ))
            if not scored:
                return None
            scored.sort(reverse=True)
            # 如果完全没有上下文命中且锚点/贴近都很弱，可以阈值丢弃，避免误挂
            best = scored[0]
            cont, anc, prox, _, _, cid = best
            if (cont == 0 and anc == 0 and prox < 0.25):  # 贴近度阈值可调
                return None
            return cid
        
        def _attach_to_any_nearest_case(msg_idx: int, cases: List[Dict]) -> int:
            """终极兜底：挂靠到任何最近的case"""
            if not cases:
                return 0  # 如果没有cases，返回第一个（这种情况理论上不应该发生）
            
            # 找到包含最近消息的case
            best_distance = float('inf')
            best_case_id = 0
            
            for case_idx, case in enumerate(cases):
                for msg_id in case.get('msg_list', []):
                    distance = abs(msg_id - msg_idx)
                    if distance < best_distance:
                        best_distance = distance
                        best_case_id = case_idx
            
            return best_case_id
        
        def _attach_to_case(msg_idx: int, case_id: int, cases: List[Dict], provisionals: List[Dict], reason: str):
            """执行挂靠操作"""
            if case_id < len(cases):
                cases[case_id]["msg_list"].append(msg_idx)
                cases[case_id]["msg_list"] = sorted(set(cases[case_id]["msg_list"]))
                provisionals.append({
                    "type": "auto_attach",
                    "msg_idx": msg_idx,
                    "attached_to": case_id,
                    "reason": reason
                })
        
        # 使用自己的消息ID列表
        chunk_msg_ids = self.get_message_indices()
        
        out = copy.deepcopy(cases)

        # 1) 规范化 & 补齐字段
        for idx in range(len(out)):
            out[idx] = self._ensure_case_schema(out[idx])

        # 2) case 内排序稳定 + 去空 case
        out = [c for c in out if c["msg_list"]]
        out.sort(key=lambda c: (c["msg_list"][0], c.get("confidence", 0.0) * -1))

        # 3) 建立 msg -> cases 反查
        msg_to_cases = defaultdict(list)
        for cid, c in enumerate(out):
            for i in c["msg_list"]:
                msg_to_cases[i].append(cid)

        provisionals = []

        # 4) 去重：每条 msg 只保留一个 case
        for i, cids in list(msg_to_cases.items()):
            if len(cids) <= 1:
                continue
            winner = self._choose_one_for_duplicate(i, out, cids, prev_context)
            losers = [cid for cid in cids if cid != winner]
            # 从 loser 中移除该 msg
            for cid in losers:
                ml = out[cid]["msg_list"]
                if i in ml:
                    out[cid]["msg_list"] = [x for x in ml if x != i]
            provisionals.append({
                "type": "duplicate_resolution",
                "msg_idx": i,
                "chosen_case": winner,
                "rejected_cases": losers,
                "reason": "anchor > continuation > confidence > proximity > case_id"
            })

        # 5) 再次清理空 case
        out = [c for c in out if c["msg_list"]]
        # 重新构建 msg 索引
        msg_to_cases.clear()
        for cid, c in enumerate(out):
            for i in c["msg_list"]:
                msg_to_cases[i].append(cid)

        # 6) 识别未分配
        chunk_set = set(int(x) for x in chunk_msg_ids)
        assigned = set(msg_to_cases.keys())
        unassigned = sorted(list(chunk_set - assigned))

        # 7) 挂靠未分配（必须全部挂靠，三层优先级）
        for i in unassigned:
            cid = None
            reason = ""
            
            # 第一优先级：空消息挂靠到相同sender的最近消息
            if _is_empty_message(i):
                cid = _find_nearest_same_sender_case(i, out)
                if cid is not None:
                    reason = "empty_message_same_sender"
            
            # 第二优先级：智能挂靠逻辑
            if cid is None:
                cid = _attach_unassigned_smart(i, out)
                if cid is not None:
                    reason = "smart_attachment"
            
            # 第三优先级：终极兜底挂靠
            if cid is None:
                cid = _attach_to_any_nearest_case(i, out)
                reason = "nearest_fallback"
            
            # 执行挂靠（cid保证不为None）
            _attach_to_case(i, cid, out, provisionals, reason)

        # 8) 最终排序稳定（按最小 msg 升序）
        out.sort(key=lambda c: c["msg_list"][0])

        # 9) 自检报告
        # 9.1 统计重复与覆盖率
        final_msg_to_cases = defaultdict(int)
        for c in out:
            for i in c["msg_list"]:
                final_msg_to_cases[i] += 1
        duplicates_after = [i for i, cnt in final_msg_to_cases.items() if cnt > 1]
        covered = set(final_msg_to_cases.keys())
        missing_after = sorted(list(chunk_set - covered))

        report = {
            "total_cases_in": len(cases),
            "total_cases_out": len(out),
            "total_msgs": len(chunk_set),
            "covered_msgs": len(covered),
            "missing_msgs": len(missing_after),
            "duplicates_after": len(duplicates_after),
            "duplicates_after_list": duplicates_after[:50],  # 截断，避免过大
        }

        return {
            "cases_out": out,
            "provisionals": provisionals,
            "report": report
        }
    
    # =========================
    # Helper methods for case processing
    # =========================
    
    def _ensure_case_schema(self, c: Dict[str, Any]) -> Dict[str, Any]:
        """补齐字段、规范类型，不改入参（在外层会 deepcopy）"""
        # Import from ChannelSegmenter constants
        REQUIRED_FIELDS_DEFAULTS = {
            "summary": "N/A",
            "status": "ongoing",
            "pending_party": "N/A",
            "last_update": "N/A",
            "is_active_case": False,
            "confidence": 0.0,
            "anchors": {}
        }
        ANCHOR_KEYS_LAX = ("tracking", "order", "order_ids", "buyer", "buyers", "topic")
        
        if "msg_list" not in c or not isinstance(c["msg_list"], list):
            c["msg_list"] = []
        # 统一整型 + 升序去重
        c["msg_list"] = sorted({int(x) for x in c["msg_list"]})

        for k, v in REQUIRED_FIELDS_DEFAULTS.items():
            c.setdefault(k, copy.deepcopy(v))

        # 规范 anchors
        if not isinstance(c["anchors"], dict):
            c["anchors"] = {}
        for k in ANCHOR_KEYS_LAX:
            v = c["anchors"].get(k)
            if v is None:
                continue
            # 统一为 list[str]
            if isinstance(v, (str, int)):
                c["anchors"][k] = [str(v)]
            elif isinstance(v, list):
                c["anchors"][k] = [str(x) for x in v if x is not None]
            else:
                c["anchors"][k] = [str(v)]

        # 规范 last_update（容错 ISO，不可解析则保留原值）
        lu = c.get("last_update", "N/A")
        if isinstance(lu, str) and lu not in ("", "N/A"):
            try:
                # 尝试解析；再统一成 ISO 格式
                dt = datetime.fromisoformat(lu.replace("Z", "+00:00"))
                c["last_update"] = dt.isoformat().replace("+00:00","Z")
            except Exception:
                pass

        # 置信度裁剪
        try:
            c["confidence"] = float(c.get("confidence", 0.0))
        except Exception:
            c["confidence"] = 0.0
        c["confidence"] = max(0.0, min(1.0, c["confidence"]))

        # 状态合法性
        if c.get("status") not in ("open", "ongoing", "resolved", "blocked"):
            c["status"] = "ongoing"

        # is_active_case 合法性
        c["is_active_case"] = bool(c.get("is_active_case", False))
        return c
    
    def _anchor_strength(self, case: Dict[str, Any]) -> int:
        # tracking(4) > order(3) > buyer(2) > topic(1)
        anc = case.get("anchors", {})
        if anc.get("tracking"): return 4
        if anc.get("order") or anc.get("order_ids"): return 3
        if anc.get("buyer") or anc.get("buyers"): return 2
        if anc.get("topic"): return 1
        return 0

    def _hits_active_hints(self, case: Dict[str, Any], prev_context: Optional[Dict[str, Any]]) -> bool:
        if not prev_context: return False
        hints = prev_context.get("ACTIVE_CASE_HINTS", [])
        if not hints: return False
        anc = case.get("anchors", {})
        # Use class constant from ChannelSegmenter 
        ANCHOR_KEYS_LAX = ("tracking", "order", "order_ids", "buyer", "buyers", "topic")
        for h in hints:
            for k in ANCHOR_KEYS_LAX:
                if set(anc.get(k, [])) & set(h.get(k, [])):
                    return True
        return False

    def _proximity_score(self, i: int, case: Dict[str, Any]) -> float:
        ml = case.get("msg_list", [])
        if not ml: return 0.0
        dist = min(abs(i - m) for m in ml)
        return 1.0 / (1 + dist)  # 1, 0.5, 0.33, ...

    def _choose_one_for_duplicate(self, i: int, cases: List[Dict[str, Any]], cids: List[int], prev_context: Optional[Dict[str, Any]]) -> int:
        # 规则：anchor_strength > 承接(prev_context) > confidence > proximity > 较小 case_id（稳定）
        scored = []
        for cid in cids:
            c = cases[cid]
            scored.append((
                self._anchor_strength(c),
                1 if self._hits_active_hints(c, prev_context) else 0,
                float(c.get("confidence", 0.0)),
                self._proximity_score(i, c),
                -cid,  # 反向用于最后的稳定 tie-break（越小优先）
                cid
            ))
        scored.sort(reverse=True)
        return scored[0][-1]


class LLMClient:
    """LLM client supporting both Claude (Anthropic) and OpenAI models"""
    
    def __init__(self, model: str = "gpt-5-2025-08-01"):
        self.model = model
        self.provider, env_key_name = self._get_provider_and_key(model)
        
        # Load API key from environment
        self.api_key = os.getenv(env_key_name)
        if not self.api_key:
            raise ValueError(f"{env_key_name} not found. Please set it in .env file")
        
        # Initialize appropriate client
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_provider_and_key(self, model: str) -> tuple[str, str]:
        """Determine provider and environment key name based on model prefix"""
        model_lower = model.lower()
        
        if model_lower.startswith('claude-'):
            return "anthropic", "ANTHROPIC_API_KEY"
        elif model_lower.startswith('gpt-') or 'gpt' in model_lower:
            return "openai", "OPENAI_API_KEY"
        elif model_lower.startswith('gemini-'):
            return "google", "GOOGLE_API_KEY"  # Future support
        else:
            raise ValueError(f"Cannot determine provider from model name: {model}. Supported prefixes: claude-, gpt-, gemini-")
    
    def generate(self, prompt: str, call_label: str = "unknown", max_tokens: int = 12000) -> str:
        """Generate response using appropriate API with debug logging"""
        import time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()
        
        # Create debug output directory if it doesn't exist
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Generate debug log filename
        debug_file = os.path.join(debug_dir, f"{call_label}_{timestamp}.log")
        
        try:
            # Log the request
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== LLM CALL DEBUG LOG ===\n")
                f.write(f"Start Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Call Label: {call_label}\n")
                f.write(f"Model: {self.model} ({self.provider.title()})\n")
                f.write(f"Max Tokens: {max_tokens}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write("\n=== PROMPT ===\n")
                f.write(prompt)
                f.write("\n\n")
            
            # Make the API call
            if self.provider == "openai":
                response = self._call_openai(prompt, max_tokens)
            else:
                response = self._call_anthropic(prompt, max_tokens)
            
            end_time = time.time()
            duration_seconds = end_time - start_time
            
            # Log the successful response
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=== RESPONSE ===\n")
                f.write(response)
                f.write(f"\n\nResponse Length: {len(response)} characters\n")
                f.write(f"End Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"LLM Call Duration: {duration_seconds:.2f} seconds\n")
                f.write("\n=== STATUS ===\n")
                f.write("Success: LLM call completed successfully\n")
            
            print(f"Debug log saved: {debug_file}")
            return response
            
        except Exception as e:
            end_time = time.time()
            duration_seconds = end_time - start_time
            
            # Log the error
            try:
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write("=== ERROR ===\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Error Type: {type(e).__name__}\n")
                    f.write(f"End Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"LLM Call Duration: {duration_seconds:.2f} seconds\n")
                    f.write("\n=== STATUS ===\n")
                    f.write(f"Failed: LLM call failed - {str(e)}\n")
            except:
                pass  # If debug logging fails, don't break the main functionality
            
            raise RuntimeError(f"LLM generation failed ({call_label}): {e}")
    
    def generate_structured(self, prompt: str, response_format, call_label: str = "unknown", max_tokens: int = 1200):
        """Generate structured response using OpenAI with JSON schema"""
        if self.provider != "openai":
            raise RuntimeError("Structured output is only supported for OpenAI models")
        
        import time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()
        
        # Create debug output directory if it doesn't exist
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Generate debug log filename
        debug_file = os.path.join(debug_dir, f"{call_label}_{timestamp}.log")
        
        try:
            # Log the request
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== LLM STRUCTURED CALL DEBUG LOG ===\n")
                f.write(f"Start Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Call Label: {call_label}\n")
                f.write(f"Model: {self.model} (OpenAI Structured)\n")
                f.write(f"Max Tokens: {max_tokens}\n")
                f.write(f"Response Format: {response_format.__name__}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write("\n=== PROMPT ===\n")
                f.write(prompt)
                f.write("\n\n")
            
            # Make the structured API call
            response = self.client.responses.parse(
                model=self.model,
                input=[{"role": "user", "content": prompt}],
                text_format=response_format,
            )

            end_time = time.time()
            duration_seconds = end_time - start_time
            parsed_response = response.output_parsed
            
            # Log the successful response
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=== RESPONSE (RAW JSON) ===\n")
                f.write(parsed_response.model_dump_json(indent=2))
                f.write(f"\n\nResponse Length: {len(parsed_response.model_dump_json(indent=2))} characters\n")
                f.write(f"End Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"LLM Call Duration: {duration_seconds:.2f} seconds\n")
                f.write("\n=== STATUS ===\n")
                f.write("Success: Structured LLM call completed successfully\n")
            
            print(f"Debug log saved: {debug_file}")
            return parsed_response
            
        except Exception as e:
            end_time = time.time()
            duration_seconds = end_time - start_time
            
            # Log the error
            try:
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write("=== ERROR ===\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Error Type: {type(e).__name__}\n")
                    f.write(f"End Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"LLM Call Duration: {duration_seconds:.2f} seconds\n")
                    f.write("\n=== STATUS ===\n")
                    f.write(f"Failed: Structured LLM call failed - {str(e)}\n")
            except:
                pass  # If debug logging fails, don't break the main functionality
            
            raise RuntimeError(f"Structured LLM generation failed ({call_label}): {e}")
    
    def _call_openai(self, prompt: str, max_tokens: int) -> str:
        """Call OpenAI API"""
        # GPT-5 models use max_completion_tokens instead of max_tokens
        if "gpt-5" in self.model.lower():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, max_tokens: int) -> str:
        """Call Anthropic API"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    def load_prompt(self, filename: str) -> str:
        """Load prompt template from prompts directory"""
        prompt_path = os.path.join("prompts", filename)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


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
    
    def filter_deleted_rows(self) -> None:
        """Filter out rows where Deleted = True"""
        if 'Deleted' in self.df.columns:
            original_count = len(self.df)
            self.df = self.df[self.df['Deleted'] != True].reset_index(drop=True)
            filtered_count = original_count - len(self.df)
            print(f"Filtered out {filtered_count} deleted rows ({len(self.df)} remaining)")
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
        
        # Save output
        output_path = self.save_output()
        
        print(f"Processing complete! Output saved to {output_path}")
        print(f"Processed {len(self.df_clean)} messages across {self.df_clean['Channel URL'].nunique()} channels")
        
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
            
            print(f"  Channel: {format_channel_for_display(channel_url)} - {len(channel_df)} messages")
        
        return channel_data_list


class ChannelSegmenter:
    """
    Segments processed messages into overlapping chunks for LLM analysis.
    Assumes single channel input.
    
    Features:
    - Half-open intervals: Uses [start, end) to avoid boundary duplication
    - Overlap validation: Ensures overlap < chunk_size/3 for optimal coverage
    - Chunk tracking: Maintains overlap metadata for context continuity
    - Case merging: Handles pairwise merge and global aggregation
    """
    
    # Case schema and anchor constants
    REQUIRED_FIELDS_DEFAULTS = {
        "summary": "N/A",
        "status": "ongoing",            # 缺省设为 ongoing，便于保守承接
        "pending_party": "N/A",
        "last_update": "N/A",
        "is_active_case": False,
        "confidence": 0.0,
        "anchors": {}
    }
    
    ANCHOR_KEYS_STRICT = ("tracking", "order", "buyer", "topic")
    ANCHOR_KEYS_LAX = ("tracking", "order", "order_ids", "buyer", "buyers", "topic")
    
    def __init__(self, df_clean: pd.DataFrame, chunk_size: int = 80, overlap: int = 20, review_gap_threshold: float = 0.05):
        self.df_clean = df_clean
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.review_gap_threshold = review_gap_threshold
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
        """Generate chunks for single channel"""
        self.chunks = []
        total_messages = len(self.df_clean)
        
        if total_messages == 0:
            return self.chunks
        
        # Assume single channel input - get the channel URL
        channel_url = self.df_clean['Channel URL'].iloc[0] if len(self.df_clean) > 0 else "unknown"
        
        # Reset index to ensure continuous indexing within channel
        channel_df = self.df_clean.reset_index(drop=True)
        
        # Calculate number of chunks needed
        import math
        if self.overlap >= self.chunk_size:
            num_chunks = math.ceil(total_messages / self.chunk_size)
        else:
            num_chunks = max(1, math.ceil((total_messages - self.overlap) / (self.chunk_size - self.overlap)))
        
        for i in range(num_chunks):
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
            
            # Create chunk with DataFrame slice
            chunk_messages = channel_df.iloc[start_idx:end_idx].copy()
            
            chunk = Chunk(
                chunk_id=i,
                channel_url=channel_url,
                start_idx=start_idx,
                end_idx=end_idx,
                messages=chunk_messages,
                has_overlap_with_previous=has_overlap_with_previous,
                overlap_size=overlap_size
            )
            
            print(f"Generated chunk {i}: [{start_idx}, {end_idx}), "
                  f"{len(chunk_messages)} messages, channel: {format_channel_for_display(channel_url)}")
            self.chunks.append(chunk)
        
        print(f"Generated {len(self.chunks)} chunks for single channel")
        return self.chunks
    
    def process_all_chunks_with_merge(self, llm_client: 'LLMClient') -> Dict[str, Any]:
        """
        处理所有chunks并执行merge操作，返回全局cases
        """
        print(f"\n=== Processing {len(self.chunks)} chunks with merge pipeline ===")
        
        if not self.chunks:
            return {"global_cases": [], "local_to_global": {}, "total_messages": 0}
        
        chunks = self.chunks
        
        if len(chunks) == 1:
            # 单chunk情况
            chunk = chunks[0]
            print(f"Single chunk {chunk.chunk_id}: processing without merge")
            
            current_messages = chunk.format_all_messages_for_prompt()
            case_results = chunk.generate_case_segments(
                current_chunk_messages=current_messages,
                previous_chunk_tail_summary=None,
                llm_client=llm_client
            )
            
            return {
                "global_cases": case_results.get('complete_cases', []),
                "local_to_global": {f"0#{i}": i for i in range(len(case_results.get('complete_cases', [])))},
                "total_messages": chunk.total_messages,
                "chunks_processed": 1
            }
        
        # 多chunk情况 - 分两阶段处理
        print(f"Multi-chunk processing: {len(chunks)} chunks")
        
        # Stage 1: LLM调用阶段 - 处理每个chunk获取case segmentation和tail summary
        chunk_cases = []
        tail_summaries = []
        
        for i, chunk in enumerate(chunks):
            print(f"\n--- Processing chunk {chunk.chunk_id} ({i+1}/{len(chunks)}) ---")
            current_messages = chunk.format_all_messages_for_prompt()
            
            # 使用前一个chunk的tail summary
            previous_tail_summary = tail_summaries[i-1] if i > 0 else None
            
            case_results = chunk.generate_case_segments(
                current_chunk_messages=current_messages,
                previous_chunk_tail_summary=previous_tail_summary,
                llm_client=llm_client
            )
            
            chunk_cases.append(case_results.get('complete_cases', []))
            
            # 生成tail summary
            if i < len(chunks) - 1:  # 不是最后一个chunk
                tail_summary = chunk.generate_tail_summary(
                    current_messages=current_messages,
                    overlap_size=self.overlap,
                    llm_client=llm_client
                )
                tail_summaries.append(tail_summary)
        
        # Stage 2: 数据处理阶段 - 执行merge pipeline
        return self.execute_merge_pipeline(chunk_cases, tail_summaries, chunks)
    
    def execute_merge_pipeline(
        self,
        chunk_cases: List[List[Dict[str, Any]]],
        tail_summaries: List[Dict[str, Any]],
        chunks: List[Chunk]
    ) -> Dict[str, Any]:
        """
        执行merge pipeline的数据处理阶段（不包含LLM调用）
        
        Args:
            chunk_cases: 每个chunk的case分割结果
            tail_summaries: 每个chunk的tail summary
            chunks: chunk列表
            
        Returns:
            包含global_cases, validation_report等的处理结果
        """
        print(f"Executing merge pipeline for {len(chunks)} chunks")
        
        # Stage 2: 执行pairwise merge
        uf_parents = []
        merged_cases = chunk_cases.copy()
        
        for i in range(len(chunks) - 1):
            print(f"\n--- Merging chunk {chunks[i].chunk_id} + {chunks[i+1].chunk_id} ---")
            
            # 计算重叠区域
            overlap_ids = self._get_overlap_ids(chunks[i], chunks[i+1])
            
            if not overlap_ids:
                print("No overlap found, skipping merge")
                continue
            
            # 执行merge
            # 使用 tail summary (现在是字典)
            prev_context = None
            if i < len(tail_summaries) and tail_summaries[i]:
                tail_data = tail_summaries[i]
                if isinstance(tail_data, dict) and "active_case_hints" in tail_data:
                    prev_context = {"ACTIVE_CASE_HINTS": tail_data["active_case_hints"]}
            
            merge_result = self.merge_overlap(
                cases_k=merged_cases[i],
                cases_k1=merged_cases[i+1],
                prev_context=prev_context,
                overlap_ids=overlap_ids
            )
            
            # 更新merged cases
            merged_cases[i] = merge_result["cases_k_out"]
            merged_cases[i+1] = merge_result["cases_k1_out"]
            
            # 收集union-find结果
            uf_parents.append(merge_result["uf_parent"])
            
            # 报告merge结果
            if merge_result["conflicts"]:
                print(f"  Found {len(merge_result['conflicts'])} conflicts requiring review")
            if merge_result["errors"]:
                print(f"  Errors: {merge_result['errors']}")
        
        # Stage 3: 修复每个chunk
        repaired_cases = []
        for i, chunk in enumerate(chunks):
            # 使用 tail summary (现在是字典)
            prev_context = None
            if i > 0 and i-1 < len(tail_summaries) and tail_summaries[i-1]:
                tail_data = tail_summaries[i-1]
                if isinstance(tail_data, dict) and "active_case_hints" in tail_data:
                    prev_context = {"ACTIVE_CASE_HINTS": tail_data["active_case_hints"]}
            
            repair_result = chunks[i].repair_case_segment_output(
                cases=merged_cases[i],
                prev_context=prev_context
            )
            
            repaired_cases.append(repair_result["cases_out"])
        
        # Stage 4: 全局聚合
        _, local_to_global = self.build_global_mapping(uf_parents, repaired_cases)
        global_cases = self.aggregate_global_cases(repaired_cases, local_to_global)
        
        # 计算实际的channel消息总数（去重chunk重叠）
        channel_msg_indices = set()
        for chunk in chunks:
            channel_msg_indices.update(chunk.get_message_indices())
        total_unique_messages = len(channel_msg_indices)
        
        # Stage 5: 全局验证和修复
        channel_short_name = format_channel_for_display(chunks[0].channel_url) if chunks else "unknown"
        validation_report = self.validate_global_assignment(
            global_cases, 
            total_unique_messages, 
            channel_short_name
        )
        
        # 如果验证失败，尝试修复
        if not validation_report["is_valid"]:
            print(f"\n🔧 Attempting to repair assignment issues...")
            global_cases = self.repair_global_assignment(
                global_cases, 
                total_unique_messages, 
                channel_short_name
            )
            
            # 重新验证修复结果
            final_validation = self.validate_global_assignment(
                global_cases, 
                total_unique_messages, 
                f"{channel_short_name}(repaired)"
            )
            
            if final_validation["is_valid"]:
                print(f"✅ Repair successful!")
            else:
                print(f"⚠️  Repair partially successful, some issues remain")
        
        print(f"\n✅ Merge pipeline complete:")
        print(f"   {len(chunks)} chunks → {len(global_cases)} global cases")
        print(f"   Channel messages: {total_unique_messages}")
        print(f"   Assignment quality: {'Perfect' if validation_report.get('is_valid', False) else 'Needs attention'}")
        
        return {
            "global_cases": global_cases,
            "local_to_global": local_to_global,
            "total_messages": total_unique_messages,
            "chunks_processed": len(chunks),
            "validation_report": validation_report
        }

    def _get_overlap_ids(self, chunk_k: Chunk, chunk_k1: Chunk) -> Set[int]:
        """计算两个chunk的重叠消息ID"""
        # 获取两个chunk的消息ID集合
        k_ids = set(chunk_k.get_message_indices())
        k1_ids = set(chunk_k1.get_message_indices())
        
        # 返回交集
        overlap = k_ids & k1_ids
        print(f"  Overlap: {len(overlap)} messages {sorted(list(overlap))[:10]}{'...' if len(overlap) > 10 else ''}")
        return overlap
    
    
    
    # =========================
    # Case Schema and Anchor Utilities
    # =========================
    
    def _ensure_case_schema(self, c: Dict[str, Any]) -> Dict[str, Any]:
        """补齐字段、规范类型，不改入参（在外层会 deepcopy）"""
        if "msg_list" not in c or not isinstance(c["msg_list"], list):
            c["msg_list"] = []
        # 统一整型 + 升序去重
        c["msg_list"] = sorted({int(x) for x in c["msg_list"]})

        for k, v in self.REQUIRED_FIELDS_DEFAULTS.items():
            c.setdefault(k, copy.deepcopy(v))

        # 规范 anchors
        if not isinstance(c["anchors"], dict):
            c["anchors"] = {}
        for k in self.ANCHOR_KEYS_LAX:
            v = c["anchors"].get(k)
            if v is None:
                continue
            # 统一为 list[str]
            if isinstance(v, (str, int)):
                c["anchors"][k] = [str(v)]
            elif isinstance(v, list):
                c["anchors"][k] = [str(x) for x in v if x is not None]
            else:
                c["anchors"][k] = [str(v)]

        # 规范 last_update（容错 ISO，不可解析则保留原值）
        lu = c.get("last_update", "N/A")
        if isinstance(lu, str) and lu not in ("", "N/A"):
            try:
                # 尝试解析；再统一成 ISO 格式
                dt = datetime.fromisoformat(lu.replace("Z", "+00:00"))
                c["last_update"] = dt.isoformat().replace("+00:00","Z")
            except Exception:
                pass

        # 置信度裁剪
        try:
            c["confidence"] = float(c.get("confidence", 0.0))
        except Exception:
            c["confidence"] = 0.0
        c["confidence"] = max(0.0, min(1.0, c["confidence"]))

        # 状态合法性
        if c.get("status") not in ("open", "ongoing", "resolved", "blocked"):
            c["status"] = "ongoing"

        # is_active_case 合法性
        c["is_active_case"] = bool(c.get("is_active_case", False))
        return c
    
    def _anchor_strength(self, case: Dict[str, Any]) -> int:
        # tracking(4) > order(3) > buyer(2) > topic(1)
        anc = case.get("anchors", {})
        if anc.get("tracking"): return 4
        if anc.get("order") or anc.get("order_ids"): return 3
        if anc.get("buyer") or anc.get("buyers"): return 2
        if anc.get("topic"): return 1
        return 0

    def _hits_active_hints(self, case: Dict[str, Any], prev_context: Optional[Dict[str, Any]]) -> bool:
        if not prev_context: return False
        hints = prev_context.get("ACTIVE_CASE_HINTS", [])
        if not hints: return False
        anc = case.get("anchors", {})
        for h in hints:
            for k in self.ANCHOR_KEYS_LAX:
                if set(anc.get(k, [])) & set(h.get(k, [])):
                    return True
        return False

    def _proximity_score(self, i: int, case: Dict[str, Any]) -> float:
        ml = case.get("msg_list", [])
        if not ml: return 0.0
        dist = min(abs(i - m) for m in ml)
        return 1.0 / (1 + dist)  # 1, 0.5, 0.33, ...

    def _choose_one_for_duplicate(self, i: int, cases: List[Dict[str, Any]], cids: List[int], prev_context: Optional[Dict[str, Any]]) -> int:
        # 规则：anchor_strength > 承接(prev_context) > confidence > proximity > 较小 case_id（稳定）
        scored = []
        for cid in cids:
            c = cases[cid]
            scored.append((
                self._anchor_strength(c),
                1 if self._hits_active_hints(c, prev_context) else 0,
                float(c.get("confidence", 0.0)),
                self._proximity_score(i, c),
                -cid,  # 反向用于最后的稳定 tie-break（越小优先）
                cid
            ))
        scored.sort(reverse=True)
        return scored[0][-1]
    
    def _score_merge_candidate(self, msg_idx: int, case: Dict[str, Any], chunk_idx: int, prev_context: Optional[Dict[str, Any]]) -> float:
        """计算merge候选case的评分"""
        s = 0.0
        s += 0.40 * (1.0 if self._hits_active_hints(case, prev_context) else 0.0)                    # 承接
        s += 0.25 * (self._anchor_strength(case) / 4.0)                                             # 锚点强度
        s += 0.20 * float(max(0.0, min(1.0, case.get("confidence", 0.0))))                    # 置信度
        s += 0.10 * self._proximity_score(msg_idx, case)                                            # 贴近度
        s += 0.05 * (1.0 if chunk_idx == 1 else 0.0)                                          # 后块偏置
        return s

    def _anchor_equivalent(self, c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
        """检查两个case的锚点等价性"""
        a1, a2 = c1.get("anchors", {}), c2.get("anchors", {})
        
        # tracking级别等价
        if set(a1.get("tracking", [])) & set(a2.get("tracking", [])):
            return True
        
        # order级别等价  
        orders1 = set(a1.get("order", []) + a1.get("order_ids", []))
        orders2 = set(a2.get("order", []) + a2.get("order_ids", []))
        if orders1 & orders2:
            return True
        
        return False
    
    # =========================
    # Core Merge Methods
    # =========================
    
    def merge_overlap(
        self,
        cases_k: List[Dict[str, Any]],
        cases_k1: List[Dict[str, Any]],
        prev_context: Optional[Dict[str, Any]],
        overlap_ids: Set[int]
    ) -> Dict[str, Any]:
        """
        对 chunk k 与 k+1 的 overlap 部分进行冲突裁决、去重、并案。
        
        Args:
            cases_k: chunk k的cases
            cases_k1: chunk k+1的cases
            prev_context: 前序上下文（tail summary）
            overlap_ids: 重叠的消息ID集合
        
        Returns:
            {
              "owner": { msg_idx: {"chosen": CaseRef, "candidates": [CaseRef...], "scores": {CaseRef: score}}},
              "cases_k_out": [...],    # 深拷贝后、已做剔除
              "cases_k1_out": [...],
              "uf_parent": { uf_key: root_key, ... },
              "conflicts": [ {msg_idx, chosen:CaseRef, alt:CaseRef, score_gap, reason}, ... ],
              "errors": [ ... ]        # 非致命问题
            }
        """
        # 深拷贝，避免副作用
        cases0 = copy.deepcopy(cases_k)
        cases1 = copy.deepcopy(cases_k1)

        uf = UnionFind()
        owner: Dict[int, Dict[str, Any]] = {}
        conflicts: List[Dict[str, Any]] = []
        errors: List[str] = []
        pending_removals = defaultdict(list)  # (chunk_idx, case_id) -> [msg_idx,...]

        for i in sorted(overlap_ids):
            candidates: List[Tuple[CaseRef, Dict[str, Any], float]] = []
            
            # 收集chunk k的候选
            for cid, c in enumerate(cases0):
                if i in c.get("msg_list", []):
                    ref = CaseRef(0, cid)
                    score = self._score_merge_candidate(i, c, 0, prev_context)
                    candidates.append((ref, c, score))
            
            # 收集chunk k+1的候选
            for cid, c in enumerate(cases1):
                if i in c.get("msg_list", []):
                    ref = CaseRef(1, cid)
                    score = self._score_merge_candidate(i, c, 1, prev_context)
                    candidates.append((ref, c, score))

            # 统一 owner 结构（即使 0 或 1 候选）
            owner[i] = {"chosen": None, "candidates": [c[0] for c in candidates], "scores": {}}
            for ref, _, sc in candidates:
                owner[i]["scores"][ref] = sc

            if len(candidates) == 0:
                # 无人认领：留给未分配处理器
                continue

            # 按分数排序
            candidates.sort(key=lambda x: x[2], reverse=True)
            chosen_ref, chosen_case, best_score = candidates[0]

            # 完全平分时的细化 tie-break：锚点强度 > 后块偏置 > (chunk, case_id)
            if len(candidates) > 1 and abs(best_score - candidates[1][2]) < 1e-12:
                def tie_key(t):
                    ref, case, _ = t
                    return (
                        self._anchor_strength(case),
                        1 if ref.chunk_idx == 1 else 0,
                        ref.chunk_idx,
                        ref.case_id
                    )
                candidates.sort(key=tie_key, reverse=True)
                chosen_ref, chosen_case, best_score = candidates[0]

            owner[i]["chosen"] = chosen_ref

            # 标记需要从其他候选中移除该 msg（稍后统一应用）
            for ref, case, sc in candidates[1:]:
                pending_removals[(ref.chunk_idx, ref.case_id)].append(i)

            # 并案：若锚点等价，Union 成同一全局案
            for ref, case, sc in candidates[1:]:
                if self._anchor_equivalent(chosen_case, case):
                    uf.union(chosen_ref.uf_key(), ref.uf_key())

            # 进入复核队列：分差接近
            if len(candidates) > 1 and (best_score - candidates[1][2]) < self.review_gap_threshold:
                conflicts.append({
                    "msg_idx": i,
                    "chosen": chosen_ref,
                    "alt": candidates[1][0],
                    "score_gap": round(best_score - candidates[1][2], 4),
                    "reason": "small score gap in overlap"
                })

        # 应用剔除（统一修改，避免副作用/读写竞态）
        for (ck, cid), to_remove in pending_removals.items():
            target_cases = cases0 if ck == 0 else cases1
            if 0 <= cid < len(target_cases):
                ml = target_cases[cid].get("msg_list", [])
                keep = sorted(set(ml) - set(to_remove))
                target_cases[cid]["msg_list"] = keep
            else:
                errors.append(f"invalid case index to remove: chunk={ck}, case={cid}")

        return {
            "owner": owner,
            "cases_k_out": cases0,
            "cases_k1_out": cases1,
            "uf_parent": uf.parent,
            "conflicts": conflicts,
            "errors": errors
        }
    
    def build_global_mapping(
        self,
        pairwise_uf_parents: List[Dict[str, str]],
        chunk_cases_list: List[List[Dict[str, Any]]]
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        把多个 pairwise merge 的 uf_parent 合并为一个全局 UF 映射。
        返回：
          - uf_parent_merged: 最终 UF 的 parent 映射
          - local_to_global_id: "{chunk_idx}#{case_id}" -> global_case_id (int)
        """
        uf = UnionFind()
        # 先将所有局部 case 注册
        for chunk_idx, cases in enumerate(chunk_cases_list):
            for cid, _ in enumerate(cases):
                uf.find(f"{chunk_idx}#{cid}")

        # 合并所有 pairwise union 结果
        for parent_map in pairwise_uf_parents:
            for node, parent in parent_map.items():
                uf.union(node, parent)

        # 归并到连续的全局 id
        root_to_gid: Dict[str, int] = {}
        local_to_global: Dict[str, int] = {}
        gid = 0
        for chunk_idx, cases in enumerate(chunk_cases_list):
            for cid, _ in enumerate(cases):
                key = f"{chunk_idx}#{cid}"
                root = uf.find(key)
                if root not in root_to_gid:
                    root_to_gid[root] = gid
                    gid += 1
                local_to_global[key] = root_to_gid[root]

        return uf.parent, local_to_global
    
    def aggregate_global_cases(
        self,
        chunk_cases_list: List[List[Dict[str, Any]]],
        local_to_global: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        把 (chunk, case) 聚合为全局 case：
        - msg_list 合并去重升序
        - 状态/最后时间/是否 active：取"更晚/更强"的（可按需自定义）
        - summary：简单策略为拼接最近版本或保留信息量较大的（此处取最后出现的）
        """
        global_buckets: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # gid -> [(chunk_idx, case_id)]
        for chunk_idx, cases in enumerate(chunk_cases_list):
            for cid, _ in enumerate(cases):
                key = f"{chunk_idx}#{cid}"
                gid = local_to_global[key]
                global_buckets[gid].append((chunk_idx, cid))

        globals_out: List[Dict[str, Any]] = []
        for gid, refs in sorted(global_buckets.items(), key=lambda x: x[0]):
            all_msgs: Set[int] = set()
            last_case: Optional[Dict[str, Any]] = None

            # 简单的"最后写优先"
            for chunk_idx, cid in sorted(refs):
                c = chunk_cases_list[chunk_idx][cid]
                all_msgs.update(c.get("msg_list", []))
                last_case = c

            if last_case is None:
                continue

            # 你可以在此自定义更复杂的汇总策略
            globals_out.append({
                "global_case_id": gid,
                "msg_list": sorted(all_msgs),
                "summary": last_case.get("summary", "N/A"),
                "status": last_case.get("status", "N/A"),
                "pending_party": last_case.get("pending_party", "N/A"),
                "last_update": last_case.get("last_update", "N/A"),
                "is_active_case": bool(last_case.get("is_active_case", False)),
                "confidence": float(last_case.get("confidence", 0.0)),
                "anchors": last_case.get("anchors", {})
            })

        return globals_out

    def validate_global_assignment(
        self,
        global_cases: List[Dict[str, Any]], 
        total_messages: int,
        channel_name: str = ""
    ) -> Dict[str, Any]:
        """
        验证全局案例分配的完整性：
        - 检查消息覆盖率（0遗漏）
        - 检查重复分配（0重复）
        - 生成详细的分配报告
        
        Args:
            global_cases: 全局cases列表
            total_messages: 期望的总消息数
            channel_name: 频道名称（用于日志）
        
        Returns:
            验证报告字典
        """
        print(f"\n🔍 Validating global assignment for {channel_name}...")
        
        # 收集所有已分配的消息
        all_assigned_msgs = []
        case_stats = []
        
        for case_idx, case in enumerate(global_cases):
            msg_list = case.get("msg_list", [])
            all_assigned_msgs.extend(msg_list)
            
            case_stats.append({
                "global_case_id": case.get("global_case_id", case_idx),
                "msg_count": len(msg_list),
                "msg_range": f"[{min(msg_list) if msg_list else 'N/A'}, {max(msg_list) if msg_list else 'N/A'}]",
                "summary_preview": case.get("summary", "")[:50] + "..." if len(case.get("summary", "")) > 50 else case.get("summary", ""),
                "status": case.get("status", "unknown")
            })
        
        # 分析分配情况
        assigned_set = set(all_assigned_msgs)
        expected_set = set(range(total_messages))
        
        # 检查重复分配
        duplicates = []
        msg_count = defaultdict(int)
        for msg in all_assigned_msgs:
            msg_count[msg] += 1
            if msg_count[msg] > 1:
                duplicates.append(msg)
        
        # 检查遗漏
        missing = sorted(list(expected_set - assigned_set))
        
        # 检查超出范围
        out_of_range = sorted([msg for msg in assigned_set if msg >= total_messages or msg < 0])
        
        # 生成统计
        coverage_rate = len(assigned_set & expected_set) / total_messages * 100 if total_messages > 0 else 0
        
        report = {
            "channel": channel_name,
            "total_cases": len(global_cases),
            "total_messages_expected": total_messages,
            "total_messages_assigned": len(all_assigned_msgs),
            "unique_messages_assigned": len(assigned_set),
            "coverage_rate": round(coverage_rate, 2),
            "duplicates": {
                "count": len(set(duplicates)),
                "messages": sorted(list(set(duplicates)))[:20],  # 限制显示数量
                "total_duplicate_assignments": len(duplicates)
            },
            "missing": {
                "count": len(missing),
                "messages": missing[:20]  # 限制显示数量
            },
            "out_of_range": {
                "count": len(out_of_range),
                "messages": out_of_range[:10]
            },
            "case_stats": case_stats,
            "is_valid": len(missing) == 0 and len(set(duplicates)) == 0 and len(out_of_range) == 0
        }
        
        # 打印报告
        print(f"  📊 Cases: {report['total_cases']}")
        print(f"  📈 Coverage: {report['coverage_rate']:.1f}% ({report['unique_messages_assigned']}/{report['total_messages_expected']})")
        
        if report['duplicates']['count'] > 0:
            print(f"  ⚠️  Duplicates: {report['duplicates']['count']} messages, {report['duplicates']['total_duplicate_assignments']} total assignments")
            print(f"     Sample: {report['duplicates']['messages'][:5]}")
        
        if report['missing']['count'] > 0:
            print(f"  ❌ Missing: {report['missing']['count']} messages")
            print(f"     Sample: {report['missing']['messages'][:5]}")
        
        if report['out_of_range']['count'] > 0:
            print(f"  🚫 Out of range: {report['out_of_range']['count']} messages")
            print(f"     Sample: {report['out_of_range']['messages'][:5]}")
        
        if report['is_valid']:
            print(f"  ✅ Validation PASSED - Perfect assignment!")
        else:
            print(f"  ❌ Validation FAILED - Assignment issues detected")
        
        return report

    def repair_global_assignment(
        self,
        global_cases: List[Dict[str, Any]], 
        total_messages: int,
        channel_name: str = ""
    ) -> List[Dict[str, Any]]:
        """
        修复全局分配中的问题：
        - 去除重复分配（保留第一个分配）
        - 将遗漏消息分配到合适的case或创建misc case
        
        Args:
            global_cases: 需要修复的全局cases
            total_messages: 总消息数
            channel_name: 频道名称
            
        Returns:
            修复后的全局cases
        """
        print(f"\n🔧 Repairing global assignment for {channel_name}...")
        
        # 深拷贝避免修改原数据
        repaired_cases = copy.deepcopy(global_cases)
        
        # Step 1: 去重 - 保留第一次出现的分配
        msg_to_first_case = {}
        removals = []
        
        for case_idx, case in enumerate(repaired_cases):
            msg_list = case.get("msg_list", [])
            keep_msgs = []
            
            for msg in msg_list:
                if msg in msg_to_first_case:
                    # 重复分配，记录移除
                    removals.append({
                        "msg": msg,
                        "from_case": case_idx,
                        "kept_in_case": msg_to_first_case[msg]
                    })
                else:
                    # 首次分配，保留
                    msg_to_first_case[msg] = case_idx
                    keep_msgs.append(msg)
            
            case["msg_list"] = sorted(keep_msgs)
        
        if removals:
            print(f"  🔧 Removed {len(removals)} duplicate assignments")
        
        # Step 2: 处理遗漏消息
        assigned_msgs = set(msg_to_first_case.keys())
        expected_msgs = set(range(total_messages))
        missing_msgs = sorted(list(expected_msgs - assigned_msgs))
        
        if missing_msgs:
            print(f"  🔧 Found {len(missing_msgs)} missing messages")
            
            # 创建misc case处理遗漏消息
            misc_case = {
                "global_case_id": max([c.get("global_case_id", i) for i, c in enumerate(repaired_cases)], default=-1) + 1,
                "msg_list": missing_msgs,
                "summary": f"auto_repaired: {len(missing_msgs)} previously unassigned messages",
                "status": "open",
                "pending_party": "N/A",
                "last_update": "N/A",
                "is_active_case": False,
                "confidence": 0.2,
                "anchors": {}
            }
            
            repaired_cases.append(misc_case)
            print(f"  ➕ Created repair case {misc_case['global_case_id']} for {len(missing_msgs)} missing messages")
        
        # Step 3: 清理空cases
        non_empty_cases = [case for case in repaired_cases if case.get("msg_list")]
        removed_empty = len(repaired_cases) - len(non_empty_cases)
        
        if removed_empty > 0:
            print(f"  🧹 Removed {removed_empty} empty cases")
        
        # Step 4: 重新分配global_case_id确保连续性
        for i, case in enumerate(non_empty_cases):
            case["global_case_id"] = i
        
        print(f"  ✅ Repair completed: {len(non_empty_cases)} final cases")
        
        return non_empty_cases


def save_channel_results(channel_result: Dict[str, Any], channel_url: str, channel_idx: int, channel_df: pd.DataFrame, output_dir: str) -> None:
    """Save individual channel results to separate files"""
    import json
    
    global_cases = channel_result.get('global_cases', [])
    total_analyzed = channel_result.get('total_messages', 0)
    
    print(f"\n--- Saving Channel {channel_idx + 1} Results ---")
    print(f"Found {len(global_cases)} cases across {total_analyzed} messages")
    
    # Save channel cases to JSON
    channel_cases_file = os.path.join(output_dir, f"cases_channel_{channel_idx + 1}.json")
    save_result = {
        "channel_url": channel_url,
        "global_cases": global_cases,
        "total_messages": total_analyzed,
        "chunks_processed": channel_result.get('chunks_processed', 0)
    }
    
    with open(channel_cases_file, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False)
    print(f"Channel cases saved to: {channel_cases_file}")
    
    # Generate annotated CSV for this channel
    df_annotated = channel_df.copy()
    df_annotated['case_id'] = -1  # Default: unassigned
    
    # Map case assignments using channel's local msg_ch_idx
    assignment_stats = {"assigned": 0, "out_of_range": 0, "conflicts": 0}
    
    for case in global_cases:
        case_id = case.get('global_case_id', -1)
        for msg_ch_idx in case.get('msg_list', []):
            # Use msg_ch_idx column instead of DataFrame row index
            mask = df_annotated['msg_ch_idx'] == msg_ch_idx
            matching_rows = df_annotated[mask]
            
            if len(matching_rows) == 0:
                assignment_stats["out_of_range"] += 1
                print(f"⚠️  Warning: Message msg_ch_idx {msg_ch_idx} not found in channel data")
            else:
                # Check for conflicts
                if matching_rows['case_id'].iloc[0] != -1:
                    assignment_stats["conflicts"] += 1
                    print(f"⚠️  Warning: Message {msg_ch_idx} already assigned, reassigning to case {case_id}")
                
                # Assign case_id using msg_ch_idx-based selection
                df_annotated.loc[mask, 'case_id'] = case_id
                assignment_stats["assigned"] += 1
    
    # Save annotated CSV for this channel
    channel_segmented_file = os.path.join(output_dir, f"segmented_channel_{channel_idx + 1}.csv")
    df_annotated.to_csv(channel_segmented_file, index=False, encoding='utf-8')
    print(f"Channel annotated CSV saved to: {channel_segmented_file}")
    
    # Display assignment statistics for this channel
    assigned_count = (df_annotated['case_id'] >= 0).sum()
    coverage_rate = assigned_count / len(df_annotated) * 100 if len(df_annotated) > 0 else 0
    
    print(f"📊 Channel {channel_idx + 1} Assignment Statistics:")
    print(f"   Total messages: {len(df_annotated)}")
    print(f"   Assigned: {assigned_count} ({coverage_rate:.1f}%)")
    print(f"   Unassigned: {len(df_annotated) - assigned_count}")
    if assignment_stats["out_of_range"] > 0:
        print(f"   Out of range: {assignment_stats['out_of_range']}")
    if assignment_stats["conflicts"] > 0:
        print(f"   Conflicts resolved: {assignment_stats['conflicts']}")
    
    if coverage_rate == 100.0:
        print(f"✅ Perfect coverage achieved for channel {channel_idx + 1}!")
    else:
        print(f"⚠️  Coverage incomplete for channel {channel_idx + 1}")


def test_case_segmentation(chunks: List[Chunk], llm_client: 'LLMClient', output_dir: str) -> Dict[str, Any]:
    """Test case segmentation functionality on the first chunk"""
    if not chunks:
        print("No chunks available for case segmentation test")
        return {}
    
    print(f"\n--- Test: Case Segmentation on First Chunk ---")
    first_chunk = chunks[0]
    print(f"Processing chunk {first_chunk.chunk_id} with {first_chunk.total_messages} messages...")
    
    # Format messages for case segmentation
    current_messages = first_chunk.format_all_messages_for_prompt()
    
    # Generate case segments (no previous context for first chunk)
    print("Generating case segments using LLM...")
    case_results = first_chunk.generate_case_segments(
        current_chunk_messages=current_messages,
        previous_chunk_tail_summary=None,
        llm_client=llm_client
    )
    
    # Display results
    complete_cases = case_results.get('complete_cases', [])
    total_analyzed = case_results.get('total_messages_analyzed', 0)
    
    print(f"✅ Case segmentation complete!")
    print(f"Found {len(complete_cases)} cases in {total_analyzed} messages")
    
    # Show case summary
    for i, case in enumerate(complete_cases):
        print(f"  Case {i+1}: {case.get('summary', 'No summary')[:100]}...")
        print(f"    Status: {case.get('status', 'unknown')} | Active: {case.get('is_active_case', False)} | Messages: {len(case.get('msg_list', []))}")
    
    # Save results to JSON file
    import json
    output_file = os.path.join(output_dir, "test_case_segments.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(case_results, f, indent=2, ensure_ascii=False)
    print(f"Case segmentation test results saved to: {output_file}")
    
    return case_results


def test_tail_summary(chunks: List[Chunk], llm_client: 'LLMClient', output_dir: str, overlap_size: int) -> str:
    """Test tail summary generation functionality on the first chunk"""
    if not chunks:
        print("No chunks available for tail summary test")
        return ""
    
    print(f"\n--- Test: Tail Summary Generation on First Chunk ---")
    first_chunk = chunks[0]
    print(f"Processing chunk {first_chunk.chunk_id} with {first_chunk.total_messages} messages...")
    
    # Format messages
    current_messages = first_chunk.format_all_messages_for_prompt()
    
    # Generate tail summary (no previous context for first chunk)
    print("Generating tail summary using LLM...")
    tail_summary = first_chunk.generate_tail_summary(
        current_messages=current_messages,
        overlap_size=overlap_size,
        llm_client=llm_client
    )
    
    # Save tail summary to file
    tail_summary_file = os.path.join(output_dir, "test_tail_summary.txt")
    with open(tail_summary_file, 'w', encoding='utf-8') as f:
        import json
        f.write(json.dumps(tail_summary, ensure_ascii=False, indent=2))
    
    print(f"✅ Tail summary generation complete!")
    print(f"Summary active cases: {len(tail_summary.get('active_case_hints', []))}")
    print(f"Tail summary test results saved to: {tail_summary_file}")
    
    # Show preview of first active case hint if available
    if tail_summary.get('active_case_hints'):
        first_hint = tail_summary['active_case_hints'][0]
        print(f"First case: {first_hint.get('topic', 'N/A')} ({first_hint.get('status', 'N/A')})")
    
    return tail_summary


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
        '--test-tail-summary',
        action='store_true', 
        help='Test tail summary generation on first chunk only'
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
            print(f"Channel: {format_channel_for_display(channel_url)}")
            print(f"Messages: {len(channel_df)}")
            
            # Stage 2: Channel Segmentation for this channel
            segmenter = ChannelSegmenter(channel_df, args.chunk_size, args.overlap)
            chunks = segmenter.generate_chunks()
            
            print(f"Generated {len(chunks)} chunks with chunk_size={args.chunk_size}, overlap={args.overlap}")
            
            # Processing based on test flags
            if args.test_case_segment:
                # Test case segmentation functionality on this channel
                test_case_segmentation(chunks, llm_client, args.output_dir)
            
            if args.test_tail_summary:
                # Test tail summary functionality on this channel
                test_tail_summary(chunks, llm_client, args.output_dir, args.overlap)
            
            if not args.test_case_segment and not args.test_tail_summary:
                # Process this channel with full pipeline and save results immediately
                channel_result = segmenter.process_all_chunks_with_merge(llm_client)
                
                # Save this channel's results independently
                save_channel_results(channel_result, channel_url, channel_idx, channel_df, args.output_dir)
        
        # For test modes, exit after processing all channels
        if args.test_case_segment or args.test_tail_summary:
            return
        
        # Summary for all channels
        if not args.test_case_segment and not args.test_tail_summary:
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