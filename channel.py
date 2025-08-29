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
from typing import List, Dict, Any, TYPE_CHECKING, Optional
from case import Case, MetaInfo, CasesSegmentationListLLMRes
from utils import Utils
import copy
from collections import defaultdict

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
        self.cases: List[Case] = []
        
        self.validate_parameters()
    
    def validate_parameters(self) -> None:
        """Validate chunk_size and overlap parameters"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size / 4:
            raise ValueError(f"overlap ({self.overlap}) must be < chunk_size/4 ({self.chunk_size/4:.1f})")
    
    
    
    def segment_all_chunks(self, llm_client: 'LLMClient') -> List[Dict[str, Any]]:
        """
        生成chunks并进行LLM case segmentation，返回修复后的cases列表
        
        Args:
            llm_client: LLM客户端
            
        Returns:
            修复后的case字典列表
        """
        # Generate chunks internally
        total_messages = len(self.df_clean)
        
        # Calculate number of chunks needed
        import math
        num_chunks = math.ceil(total_messages / self.chunk_size)
        
        print(f"        📦 Processing {num_chunks} chunks for LLM segmentation")
        
        raw_cases = []
        for i in range(num_chunks):
            # Calculate chunk boundaries using half-open intervals
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, total_messages)
            
            # Create chunk DataFrame slice
            chunk_messages = self.df_clean.iloc[start_idx:end_idx].copy()
            
            print(f"            Chunk {i+1}/{num_chunks}: Processing chunk {i}")
            
            # Format messages using local helper method
            current_messages = self._format_messages_for_prompt(chunk_messages)
            
            # Generate case segments using LLM for current chunk
            try:
                # Load the segmentation prompt template
                prompt_template = llm_client.load_prompt("segmentation_prompt.md")
            except FileNotFoundError as e:
                raise RuntimeError(f"Cannot load segmentation prompt: {e}")
            
            final_prompt = prompt_template.replace(
                "<<<INSERT_CHUNK_BLOCK_HERE>>>", 
                current_messages
            )
            
            # Generate case segments using LLM
            try:
                # Generate contextual call label with timestamp                
                structured_response = llm_client.generate_structured(
                    final_prompt, 
                    CasesSegmentationListLLMRes, 
                    call_label=f"case_segmentation_{Utils.format_channel_for_display(self.channel_url)}_chunk_{i}"
                )
                    
                # Convert LLM response to dict format and add to raw cases
                chunk_raw_cases = [case.model_dump() for case in structured_response.complete_cases]
                raw_cases.extend(chunk_raw_cases)
                
            except Exception as e:
                raise RuntimeError(f"Failed to generate case segments for chunk {i}: {e}")
        
        print(f"        ✅ LLM segmentation complete ({len(raw_cases)} raw cases collected)")
        
        # Repair all cases with unified logic (repair method includes internal reporting)
        print(f"        🔧 Running unified repair on {len(raw_cases)} raw cases")
        repair_result = self.repair_case_segment_output(
            cases=raw_cases,
            chunk_df=self.df_clean,  # Use full dataframe for repair
            prev_context=None
        )
        
        repaired_cases = repair_result['cases_out']
        print(f"    ✅ Segmentation and repair complete ({len(repaired_cases)} repaired cases)")
        return repaired_cases
    
    
    def build_cases_simple(self, llm_client: 'LLMClient') -> List[Case]:
        """
        构建cases：直接对channel messages进行分割，创建Case对象并分类
        
        Args:
            llm_client: LLM客户端
            
        Returns:
            Case对象列表，包含分类和性能指标
        """
        print(f"    🔄 Segmenting channel messages directly")
        
        # 1. 直接对整个channel的消息进行分割
        repaired_case_dicts = self.segment_all_chunks(llm_client)
        
        print(f"    🏗️  Creating Case objects with classification and metrics")
        
        # 2. 将字典转换为Case对象，并添加分类和指标
        case_objects = []
        for idx, case_dict in enumerate(repaired_case_dicts):
            # Extract messages first using msg_index_list from dictionary
            msg_index_list = case_dict['msg_index_list']
            case_messages = self.df_clean[self.df_clean['msg_ch_idx'].isin(msg_index_list)].copy()
            
            # Create Case object from dictionary
            case_obj = Case(
                case_id=f'case_{idx:03d}',
                msg_index_list=msg_index_list,
                messages=case_messages,
                summary=case_dict['summary'],
                status=case_dict['status'],
                pending_party=case_dict['pending_party'],
                confidence=case_dict['confidence'],
                meta=MetaInfo(
                    tracking_numbers=case_dict.get('meta', {}).get('tracking_numbers', []),
                    order_numbers=case_dict.get('meta', {}).get('order_numbers', []),
                    user_names=case_dict.get('meta', {}).get('user_names', [])
                )
            )
                        
            # Perform classification using LLM
            print(f"        📊 Classifying case {case_obj.case_id}")
            try:
                case_obj.classify_case(llm_client)
            except Exception as e:
                print(f"        ⚠️  Classification failed for {case_obj.case_id}: {e}")
            
            # Calculate performance metrics
            case_obj.calculate_metrics()
            
            case_objects.append(case_obj)
        
        self.cases = case_objects
        
        print(f"    ✅ Cases built successfully ({len(self.cases)} Case objects)")
        return self.cases
    
    def build_cases_via_file(self, output_dir: str) -> List[Case]:
        """
        从JSON文件加载现有结果并构建Case对象，确保与LLM处理路径的self.cases结构完全一致
        
        Args:
            output_dir: 输出目录路径
            
        Returns:
            Case对象列表，包含所有分类和性能指标数据
        """
        import json
        
        # 构建文件路径（与save_results_to_json相同的逻辑）
        session_folder = os.path.join(output_dir, f"session_{self.session}")
        channel_name = Utils.format_channel_for_display(self.channel_url)
        channel_cases_file = os.path.join(session_folder, f"cases_{channel_name}.json")
        
        if not os.path.exists(channel_cases_file):
            raise FileNotFoundError(f"JSON file not found: {channel_cases_file}")
        
        # 加载JSON数据
        try:
            with open(channel_cases_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON file: {e}")
        
        global_cases_data = saved_data.get('global_cases', [])
        
        # 将字典数据转换为Case对象（与build_global_cases完全相同的逻辑）
        case_objects = []
        for case_dict in global_cases_data:
            # 创建Case对象，使用文件中的所有数据
            case_obj = Case(
                case_id=case_dict.get('case_id'),
                msg_index_list=case_dict.get('msg_index_list', []),
                summary=case_dict.get('summary', 'N/A'),
                status=case_dict.get('status', 'ongoing'),
                pending_party=case_dict.get('pending_party', 'N/A'),
                confidence=case_dict.get('confidence', 0.0),
                # 加载分类结果
                main_category=case_dict.get('main_category', 'unknown'),
                sub_category=case_dict.get('sub_category', 'unknown'),
                classification_reasoning=case_dict.get('classification_reasoning', 'N/A'),
                classification_confidence=case_dict.get('classification_confidence', 0.0),
                classification_indicators=case_dict.get('classification_indicators', []),
                # 加载性能指标
                first_res_time=case_dict.get('first_res_time', -1),
                handle_time=case_dict.get('handle_time', -1),
                first_contact_resolution=case_dict.get('first_contact_resolution', -1),
                usr_msg_num=case_dict.get('usr_msg_num', -1),
                total_msg_num=case_dict.get('total_msg_num', -1),
                # 加载meta信息
                meta=MetaInfo(
                    tracking_numbers=case_dict.get('meta', {}).get('tracking_numbers', []),
                    order_numbers=case_dict.get('meta', {}).get('order_numbers', []),
                    user_names=case_dict.get('meta', {}).get('user_names', [])
                )
            )
            
            # 提取消息DataFrame（与build_global_cases相同的逻辑）
            case_messages = self.df_clean[self.df_clean['msg_ch_idx'].isin(case_obj.msg_index_list)].copy()
            case_obj.messages = case_messages
            
            case_objects.append(case_obj)
        
        self.cases = case_objects
        
        print(f"        ✅ Cases loaded from file successfully ({len(self.cases)} Case objects)")
        return self.cases

    def _format_messages_for_prompt(self, chunk_df: pd.DataFrame) -> str:
        """Format DataFrame messages for LLM prompt: message_index | sender id | role | timestamp | text"""
        formatted_lines = []
        for _, row in chunk_df.iterrows():
            # Handle NaN messages and replace newlines with spaces to keep one line per message
            message_text = str(row['Message']).replace('\n', ' ').replace('\r', ' ')
            if message_text == 'nan':
                message_text = ''
            
            formatted_line = f"{row['msg_ch_idx']} | {row['Sender ID']} | {row['role']} | {row['Created Time']} | {message_text}"
            formatted_lines.append(formatted_line)
        return '\n'.join(formatted_lines)

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
            "global_cases": [case.to_dict() for case in self.cases],
            "total_messages": len(self.df_clean),
        }
        
        try:
            with open(channel_cases_file, 'w', encoding='utf-8') as f:
                json.dump(save_result, f, indent=2, ensure_ascii=False)
            print(f"            Channel cases saved to: {channel_cases_file}")
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
        for case_obj in self.cases:
            case_id = case_obj.case_id or "unknown"
            main_category = case_obj.main_category
            sub_category = case_obj.sub_category
            
            for msg_ch_idx in case_obj.msg_index_list:
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
            print(f"            Channel annotated CSV saved to: {channel_segmented_file}")
        except IOError as e:
            print(f"                ❌ Error saving CSV file: {e}")
            raise

    def repair_case_segment_output(self, cases: List[Dict[str, Any]], 
                                 chunk_df: pd.DataFrame,
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
            
            if "msg_index_list" not in c or not isinstance(c["msg_index_list"], list):
                c["msg_index_list"] = []
            # 统一整型 + 升序去重
            c["msg_index_list"] = sorted({int(x) for x in c["msg_index_list"]})

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
            ml = case.get("msg_index_list", [])
            if not ml: return 0.0
            # msg_index_list is still indices at repair stage
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
            if msg_idx not in chunk_df['msg_ch_idx'].values:
                return True
            message = chunk_df[chunk_df['msg_ch_idx'] == msg_idx].iloc[0]
            text = str(message.get('Message', '')).strip()  # Use 'Message' column as seen in Utils.format_one_msg_for_prompt
            return len(text) == 0
        
        def _find_nearest_same_sender_case(msg_idx: int, cases: List[Dict]) -> Optional[int]:
            """查找包含最近的相同sender_id消息的case"""
            if msg_idx not in chunk_df['msg_ch_idx'].values:
                return None
            
            target_sender = chunk_df[chunk_df['msg_ch_idx'] == msg_idx].iloc[0].get('Sender ID', '')
            if not target_sender:
                return None
            
            # 构建消息到case的映射
            msg_to_case = {}
            for case_idx, case in enumerate(cases):
                for msg_id in case.get('msg_index_list', []):
                    msg_to_case[msg_id] = case_idx
            
            # 寻找最近的相同sender消息
            best_distance = float('inf')
            best_case_id = None
            
            for check_msg_idx, case_idx in msg_to_case.items():
                if check_msg_idx in chunk_df['msg_ch_idx'].values:
                    check_sender = chunk_df[chunk_df['msg_ch_idx'] == check_msg_idx].iloc[0].get('Sender ID', '')
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
                for msg_id in case.get('msg_index_list', []):
                    distance = abs(msg_id - msg_idx)
                    if distance < best_distance:
                        best_distance = distance
                        best_case_id = case_idx
            
            return best_case_id
        
        def _attach_to_case(msg_idx: int, case_id: int, cases: List[Dict], provisionals: List[Dict], reason: str):
            """执行挂靠操作"""
            if case_id < len(cases):
                cases[case_id]["msg_index_list"].append(msg_idx)
                cases[case_id]["msg_index_list"] = sorted(set(cases[case_id]["msg_index_list"]))
                provisionals.append({
                    "type": "auto_attach",
                    "msg_idx": msg_idx,
                    "attached_to": case_id,
                    "reason": reason
                })
        
        # 使用自己的消息ID列表
        chunk_msg_ids = chunk_df['msg_ch_idx'].tolist()
            
        out = copy.deepcopy(cases)

        # 1) 规范化 & 补齐字段
        for idx in range(len(out)):
            out[idx] = _ensure_case_schema(out[idx])

        # 2) case 内排序稳定 + 去空 case
        out = [c for c in out if c["msg_index_list"]]
        out.sort(key=lambda c: (c["msg_index_list"][0], c.get("confidence", 0.0) * -1))

        # 3) 建立 msg -> cases 反查
        msg_to_cases = defaultdict(list)
        for cid, c in enumerate(out):
            for i in c["msg_index_list"]:
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
                ml = out[cid]["msg_index_list"]
                if i in ml:
                    out[cid]["msg_index_list"] = [x for x in ml if x != i]
            provisionals.append({
                "type": "duplicate_resolution",
                "msg_idx": i,
                "chosen_case": winner,
                "rejected_cases": losers,
                "reason": "anchor > continuation > confidence > proximity > case_id"
            })

        # 5) 再次清理空 case
        out = [c for c in out if c["msg_index_list"]]
        # 重新构建 msg 索引
        msg_to_cases.clear()
        for cid, c in enumerate(out):
            for i in c["msg_index_list"]:
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
        out.sort(key=lambda c: c["msg_index_list"][0])

        # 9) 自检报告
        # 9.1 统计重复与覆盖率
        final_msg_to_cases = defaultdict(int)
        for c in out:
            for i in c["msg_index_list"]:
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

        # 打印修复情况报告
        if provisionals:
            print(f"        🔧 Applied {len(provisionals)} repair actions:")
            for prov in provisionals:
                if prov['type'] == 'duplicate_resolution':
                    print(f"            ➜ Resolved duplicate msg {prov['msg_idx']}: kept in case {prov['chosen_case']}")
                elif prov['type'] == 'auto_attach':
                    print(f"            ➕ Auto-attached msg {prov['msg_idx']} to case {prov['attached_to']}")
                elif prov['type'] == 'misc_bucket':
                    print(f"            📦 Created misc case for {len(prov['msg_idxs'])} unassigned messages")
        
        # 打印最终验证结果
        if report['missing_msgs'] == 0 and report['duplicates_after'] == 0:
            print(f"        ✅ Repair completed: 100% coverage achieved - Final: {report['covered_msgs']}/{report['total_msgs']} messages in {report['total_cases_out']} cases")
        else:
            print(f"        ⚠️ Repair incomplete: Missing: {report['missing_msgs']}, Duplicates: {report['duplicates_after']}")

        return {
            "cases_out": out,
            "provisionals": provisionals,
            "report": report
        }