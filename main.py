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


class MetaInfo(BaseModel):
    """Meta information structure for each case"""
    model_config = {"extra": "forbid"}
    tracking_numbers: List[str] = Field(default_factory=list, description="跟踪号列表")
    order_numbers: List[str] = Field(default_factory=list, description="订单号列表")
    user_names: List[str] = Field(default_factory=list, description="用户名列表")

class CaseItem(BaseModel):
    """Individual case structure for case segmentation output"""
    model_config = {"extra": "forbid"}  # Ensures additionalProperties: false
    
    msg_list: List[int]
    summary: str
    status: str  # open | ongoing | resolved | blocked
    pending_party: str  # seller|platform|N/A
    last_update: str  # ISO timestamp or N/A
    confidence: float
    meta: MetaInfo

class CasesSegmentationResponse(BaseModel):
    """Complete response structure for case segmentation"""
    model_config = {"extra": "forbid"}  # Ensures additionalProperties: false
    
    complete_cases: List[CaseItem]
    total_messages_analyzed: int
    llm_duration_seconds: Optional[float] = None



# Case Review Models
class CaseReviewInput(BaseModel):
    """Input structure for case review"""
    model_config = {"extra": "forbid"}
    cases: List[CaseItem] = Field(..., description="相关的cases列表")
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
    updated_cases: List[CaseItem] = Field(..., description="更新后的cases")
    confidence: float = Field(..., description="review结果的置信度", ge=0.0, le=1.0)


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

@dataclass
class Chunk:
    """Data structure for a single chunk of messages"""
    chunk_id: int                    # Sequential chunk ID (0, 1, 2, ...)
    channel_url: str                 # Channel this chunk belongs to
    messages: pd.DataFrame           # DataFrame slice with messages in this chunk
    
    @property
    def total_messages(self) -> int:
        """Number of messages in this chunk (calculated from DataFrame length)"""
        return len(self.messages)

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
    

    def generate_case_segments(self, 
                             current_chunk_messages: str, 
                             llm_client: 'LLMClient') -> Dict[str, Any]:
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
            if llm_client.provider == "openai" and CasesSegmentationResponse:
                # Structured output with Pydantic schema
                structured_response = llm_client.generate_structured(
                    final_prompt, 
                    CasesSegmentationResponse, 
                    call_label="case_segmentation"
                )
                # Convert Pydantic response to dict for compatibility
                result = structured_response.model_dump()
                
            
            # 直接使用修复函数（无 previous context）
            repair_result = self.repair_case_segment_output(
                cases=result.get('complete_cases', []),
                prev_context=None
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
            
            return result['complete_cases']
            
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
        def _ensure_case_schema(c: Dict[str, Any]) -> Dict[str, Any]:
            """补齐字段、规范类型，不改入参（在外层会 deepcopy）"""
            # Import from ChannelSegmenter constants
            REQUIRED_FIELDS_DEFAULTS = {
                "summary": "N/A",
                "status": "ongoing",
                "pending_party": "N/A",
                "last_update": "N/A",
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

            return c
        
        def _anchor_strength(case: Dict[str, Any]) -> int:
            # tracking(4) > order(3) > buyer(2) > topic(1)
            anc = case.get("anchors", {})
            if anc.get("tracking"): return 4
            if anc.get("order") or anc.get("order_ids"): return 3
            if anc.get("buyer") or anc.get("buyers"): return 2
            if anc.get("topic"): return 1
            return 0

        def _hits_active_hints(case: Dict[str, Any], prev_context: Optional[Dict[str, Any]]) -> bool:
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

        def _proximity_score(i: int, case: Dict[str, Any]) -> float:
            ml = case.get("msg_list", [])
            if not ml: return 0.0
            dist = min(abs(i - m) for m in ml)
            return 1.0 / (1 + dist)  # 1, 0.5, 0.33, ...

        def _choose_one_for_duplicate(i: int, cases: List[Dict[str, Any]], cids: List[int], prev_context: Optional[Dict[str, Any]]) -> int:
            # 规则：anchor_strength > 承接(prev_context) > confidence > proximity > 较小 case_id（稳定）
            scored = []
            for cid in cids:
                c = cases[cid]
                scored.append((
                    _anchor_strength(c),
                    1 if _hits_active_hints(c, prev_context) else 0,
                    float(c.get("confidence", 0.0)),
                    _proximity_score(i, c),
                    -cid,  # 反向用于最后的稳定 tie-break（越小优先）
                    cid
                ))
            scored.sort(reverse=True)
            return scored[0][-1]

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
                # 仅考虑 open/ongoing/blocked 的，跳过 resolved
                if c.get("status") == "resolved":
                    continue
                scored.append((
                    1 if _hits_active_hints(c, prev_context) else 0,
                    _anchor_strength(c),
                    _proximity_score(msg_idx, c),
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
            out[idx] = _ensure_case_schema(out[idx])

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
            winner = _choose_one_for_duplicate(i, out, cids, prev_context)
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
    
    def __init__(self, df_clean: pd.DataFrame, chunk_size: int = 80, overlap: int = 20, review_gap_threshold: float = 0.05):
        self.df_clean = df_clean
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
                messages=chunk_messages
            )
            
            msg_indices = chunk_messages['msg_ch_idx'].tolist()
            print(f"Generated chunk {i}: msg_indices [{min(msg_indices)}, {max(msg_indices)}], "
                  f"{len(chunk_messages)} messages, channel: {format_channel_for_display(channel_url)}")
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
        
        chunk_cases = []
        if chunks:
            chunk_cases = self.segment_all_chunks(chunks, llm_client)
        
        return chunk_cases
    

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
        print(f"    Status: {case.get('status', 'unknown')} | Messages: {len(case.get('msg_list', []))}")
    
    # Save results to JSON file
    import json
    output_file = os.path.join(output_dir, "test_case_segments.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(case_results, f, indent=2, ensure_ascii=False)
    print(f"Case segmentation test results saved to: {output_file}")
    
    return case_results




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
            print(f"Channel: {format_channel_for_display(channel_url)}")
            print(f"Messages: {len(channel_df)}")
            
            # Stage 2: Channel Segmentation for this channel
            segmenter = ChannelSegmenter(channel_df, args.chunk_size, args.overlap)
            chunks = segmenter.generate_chunks()
            
            print(f"Generated {len(chunks)} chunks with chunk_size={args.chunk_size}")
            
                # Process this channel with full pipeline and save results immediately
            if args.enable_review:
                channel_result = segmenter.segment_all_chunks_with_review(segmenter.chunks, llm_client)
            else:
                channel_result = segmenter.segment_all_chunks_simple(segmenter.chunks, llm_client)
                
                # Save this channel's results independently
            save_channel_results(channel_result, channel_url, channel_idx, channel_df, args.output_dir)
        
        
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