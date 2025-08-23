#!/usr/bin/env python3
"""
Chunk processing module for customer support message segmentation.

This module contains:
- Pydantic models for LLM structured output
- Chunk class for processing message chunks and case segmentation
"""

import pandas as pd # type: ignore
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime
import copy
from collections import defaultdict

# Pydantic imports (only for LLM-compatible classes)
from typing import Literal
from pydantic import BaseModel, Field  # type: ignore

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from llm_client import LLMClient


# ----------------------------
# Pydantic Models for Structured Output
# ----------------------------

@dataclass
class MetaInfo:
    """Meta information structure for each case"""
    tracking_numbers: List[str] = field(default_factory=list)
    order_numbers: List[str] = field(default_factory=list)
    user_names: List[str] = field(default_factory=list)

@dataclass
class CaseItem:
    """Individual case structure for case segmentation output"""
    case_id: Optional[int] = None  # Case ID (assigned during processing)
    msg_list: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = "N/A"
    status: str = "ongoing"  # open | ongoing | resolved | blocked
    pending_party: str = "N/A"  # seller|platform|N/A
    last_update: str = "N/A"  # ISO timestamp or N/A
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
            'msg_list': self.msg_list.to_dict('records') if not self.msg_list.empty else [],
            'summary': self.summary,
            'status': self.status,
            'pending_party': self.pending_party,
            'last_update': self.last_update,
            'confidence': self.confidence,
            'meta': {
                'tracking_numbers': self.meta.tracking_numbers,
                'order_numbers': self.meta.order_numbers,
                'user_names': self.meta.user_names
            } if self.meta else {}
        }

# LLM-Compatible Classes (for structured output generation)
class CaseItemForLLM(BaseModel):
    """LLM-compatible case structure using List[int] for msg_list"""
    model_config = {"extra": "forbid"}
    
    case_id: Optional[int] = None  # Case ID (assigned during processing)
    msg_list: List[int]  # List of message indices instead of DataFrame
    summary: str
    status: str  # open | ongoing | resolved | blocked
    pending_party: str  # seller|platform|N/A
    last_update: str  # ISO timestamp or N/A
    confidence: float
    meta: MetaInfo

class CasesSegmentationResponseForLLM(BaseModel):
    """LLM-compatible response structure for case segmentation"""
    model_config = {"extra": "forbid"}  # Ensures additionalProperties: false
    
    complete_cases: List[CaseItemForLLM]
    total_messages_analyzed: int
    llm_duration_seconds: Optional[float] = None


# Case Review Models (LLM-compatible)
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


# ----------------------------
# Chunk Class
# ----------------------------

@dataclass
class Chunk:
    """Data structure for a single chunk of messages"""
    chunk_id: int                    # Sequential chunk ID (0, 1, 2, ...)
    channel_url: str                 # Channel this chunk belongs to
    messages: pd.DataFrame           # DataFrame slice with messages in this chunk
    has_segmentation_result: bool = False                    # Whether segmentation has been completed
    cases: List[CaseItem] = field(default_factory=list)  # Cached segmentation results
    
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
            if llm_client.provider == "openai" and CasesSegmentationResponseForLLM:
                # Structured output with LLM-compatible schema (uses List[int] for msg_list)
                structured_response = llm_client.generate_structured(
                    final_prompt, 
                    CasesSegmentationResponseForLLM, 
                    call_label="case_segmentation"
                )
                
            # Convert LLM response to dict format that repair function expects
            raw_cases = []
            for case in structured_response.complete_cases:
                case_dict = {
                    'msg_list': case.msg_list,  # Keep as List[int] for repair function
                    'summary': case.summary,
                    'status': case.status,
                    'pending_party': case.pending_party,
                    'last_update': case.last_update,
                    'confidence': case.confidence,
                    'meta': {
                        'tracking_numbers': case.meta.tracking_numbers,
                        'order_numbers': case.meta.order_numbers,
                        'user_names': case.meta.user_names
                    }
                }
                raw_cases.append(case_dict)
                
            repair_result = self.repair_case_segment_output(
                cases=raw_cases,
                prev_context=None
            )
            
            # 转换repair结果为CaseItem对象
            case_items = []
            for idx, case_dict in enumerate(repair_result['cases_out']):
                # 确保meta字段格式正确
                meta_dict = case_dict.get('meta', {})
                meta_info = MetaInfo(
                    tracking_numbers=meta_dict.get('tracking_numbers', []),
                    order_numbers=meta_dict.get('order_numbers', []),
                    user_names=meta_dict.get('user_names', [])
                )
                
                # 将索引列表转换为DataFrame
                msg_indices = case_dict.get('msg_list', [])
                if msg_indices:
                    # 从chunk的messages DataFrame中提取对应的行
                    msg_dataframe = self.messages.iloc[msg_indices].copy().reset_index(drop=True)
                else:
                    # 如果没有消息，创建空的DataFrame，保持相同的列结构
                    msg_dataframe = self.messages.iloc[0:0].copy()
                
                case_item = CaseItem(
                    case_id=idx,  # Assign case_id based on index
                    msg_list=msg_dataframe,
                    summary=case_dict.get('summary', 'N/A'),
                    status=case_dict.get('status', 'ongoing'),
                    pending_party=case_dict.get('pending_party', 'N/A'),
                    last_update=case_dict.get('last_update', 'N/A'),
                    confidence=case_dict.get('confidence', 0.0),
                    meta=meta_info
                )
                case_items.append(case_item)
            
            # 缓存结果
            self.cases = case_items
            self.has_segmentation_result = True
            
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
            
            # 返回JSON格式以保持兼容性
            return repair_result['cases_out']
            
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
                "meta": {
                    "tracking_numbers": [],
                    "order_numbers": [],
                    "user_names": []
                }
            }
            # Legacy anchor keys to meta field mapping
            ANCHOR_TO_META_MAPPING = {
                "tracking": "tracking_numbers",
                "order": "order_numbers", 
                "order_ids": "order_numbers",
                "buyer": "user_names",
                "buyers": "user_names"
            }
            
            if "msg_list" not in c or not isinstance(c["msg_list"], list):
                c["msg_list"] = []
            # 统一整型 + 升序去重
            c["msg_list"] = sorted({int(x) for x in c["msg_list"]})

            for k, v in REQUIRED_FIELDS_DEFAULTS.items():
                c.setdefault(k, copy.deepcopy(v))

            # 处理legacy anchors字段，转换为meta结构
            if "anchors" in c and isinstance(c["anchors"], dict):
                # 迁移anchors到meta
                for anchor_key, meta_key in ANCHOR_TO_META_MAPPING.items():
                    anchor_value = c["anchors"].get(anchor_key)
                    if anchor_value is not None:
                        # 统一为 list[str]
                        if isinstance(anchor_value, (str, int)):
                            meta_list = [str(anchor_value)]
                        elif isinstance(anchor_value, list):
                            meta_list = [str(x) for x in anchor_value if x is not None]
                        else:
                            meta_list = [str(anchor_value)]
                        
                        # 合并到meta字段，避免重复
                        existing = set(c["meta"][meta_key])
                        c["meta"][meta_key] = list(existing.union(meta_list))
                
                # 删除legacy anchors字段
                del c["anchors"]

            # 确保meta字段结构正确
            if not isinstance(c["meta"], dict):
                c["meta"] = copy.deepcopy(REQUIRED_FIELDS_DEFAULTS["meta"])
            
            # 确保meta中的所有必需字段存在
            for field in ["tracking_numbers", "order_numbers", "user_names"]:
                if field not in c["meta"]:
                    c["meta"][field] = []
                elif not isinstance(c["meta"][field], list):
                    c["meta"][field] = []

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
            # tracking(4) > order(3) > buyer(2)
            meta = case.get("meta", {})
            if meta.get("tracking_numbers"): return 4
            if meta.get("order_numbers"): return 3
            if meta.get("user_names"): return 2
            return 0

        def _hits_active_hints(case: Dict[str, Any], prev_context: Optional[Dict[str, Any]]) -> bool:
            if not prev_context: return False
            hints = prev_context.get("ACTIVE_CASE_HINTS", [])
            if not hints: return False
            meta = case.get("meta", {})
            
            # Check meta fields against hints
            meta_fields = ["tracking_numbers", "order_numbers", "user_names"]
            for h in hints:
                h_meta = h.get("meta", {})
                for field in meta_fields:
                    case_values = set(meta.get(field, []))
                    hint_values = set(h_meta.get(field, []))
                    if case_values & hint_values:  # Intersection
                        return True
            return False

        def _proximity_score(i: int, case: Dict[str, Any]) -> float:
            ml = case.get("msg_list", [])
            if not ml: return 0.0
            # msg_list is still indices at repair stage
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
    
    def convert_llm_response_to_internal(self, llm_response: 'CasesSegmentationResponseForLLM') -> Dict[str, Any]:
        """Convert LLM-compatible response to internal format with DataFrames"""
        converted_cases = []
        
        for idx, case_llm in enumerate(llm_response.complete_cases):
            # Convert List[int] back to DataFrame
            msg_indices = case_llm.msg_list
            if msg_indices:
                # Extract corresponding rows from chunk's messages DataFrame
                msg_dataframe = self.messages.iloc[msg_indices].copy().reset_index(drop=True)
            else:
                # Empty DataFrame with same columns
                msg_dataframe = self.messages.iloc[0:0].copy()
            
            # Convert MetaInfo from Pydantic to dataclass
            meta_internal = MetaInfo(
                tracking_numbers=case_llm.meta.tracking_numbers,
                order_numbers=case_llm.meta.order_numbers,
                user_names=case_llm.meta.user_names
            )
            
            # Create internal CaseItem with DataFrame
            case_internal = CaseItem(
                case_id=case_llm.case_id if case_llm.case_id is not None else idx,
                msg_list=msg_dataframe,
                summary=case_llm.summary,
                status=case_llm.status,
                pending_party=case_llm.pending_party,
                last_update=case_llm.last_update,
                confidence=case_llm.confidence,
                meta=meta_internal
            )
            converted_cases.append(case_internal)
        
        return {
            'complete_cases': [case.to_dict() for case in converted_cases],
            'total_messages_analyzed': llm_response.total_messages_analyzed,
            'llm_duration_seconds': llm_response.llm_duration_seconds
        }