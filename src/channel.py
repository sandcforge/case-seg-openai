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
import copy
from collections import defaultdict

# Local imports - compatible with both direct execution and module execution
try:
    from .case import Case, MetaInfo, CasesSegmentationListLLMRes
    from .utils import Utils
    from .vision_processor import VisionProcessor
except ImportError:
    from case import Case, MetaInfo, CasesSegmentationListLLMRes
    from utils import Utils
    from vision_processor import VisionProcessor

if TYPE_CHECKING:
    try:
        from .llm_client import LLMClient
    except ImportError:
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
        "segmentation_confidence": 0.0,
        "anchors": {}
    }
    
    ANCHOR_KEYS_STRICT = ("tracking", "order", "buyer", "topic")
    ANCHOR_KEYS_LAX = ("tracking", "order", "order_ids", "buyer", "buyers", "topic")
    
    def __init__(self, df_clean: pd.DataFrame, channel_url: str, session: str, chunk_size: int = 80, overlap: int = 20, enable_classification: bool = True, enable_vision_processing: bool = True, enable_find_sop: bool = True):
        self.df_clean = df_clean.copy()  # Make a copy to avoid modifying original
        self.channel_url = channel_url
        self.session = session
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enable_classification = enable_classification
        self.enable_vision_processing = enable_vision_processing
        self.enable_find_sop = enable_find_sop
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

            # Format messages using Utils helper method
            current_messages = Utils.format_messages_for_prompt2(chunk_messages)
            
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
                    
                # structured_response is already a dict, extract complete_cases
                chunk_raw_cases = structured_response['complete_cases']
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
    
    
    def build_cases_via_llm(self, llm_client: 'LLMClient') -> List[Case]:
        """
        构建cases：直接对channel messages进行分割，创建Case对象并分类
        
        Args:
            llm_client: LLM客户端
            
        Returns:
            Case对象列表，包含分类和性能指标
        """
        print(f"    🔄 Segmenting channel messages directly")
        
        # 0. Process vision analysis if enabled
        if self.enable_vision_processing:
            try:
                self.process_file_type_messages(llm_client)
            except Exception as e:
                print(f"        ⚠️  Vision processing failed: {e}")
                print(f"            Continuing without vision processing...")
        
        # 1. 直接对整个channel的消息进行分割
        repaired_case_dicts = self.segment_all_chunks(llm_client)
        
        print(f"    🏗️  Creating Case objects with classification and metrics")
        
        # 2. 将字典转换为Case对象，并添加分类和指标
        case_objects = []
        for idx, case_dict in enumerate(repaired_case_dicts):
            # Extract messages first using message_id_list from dictionary
            message_id_list = case_dict['message_id_list']
            # message_id_list now contains actual Message IDs
            case_messages = self.df_clean[self.df_clean['Message ID'].isin(message_id_list)].copy()
            
            # Create Case object from dictionary
            case_obj = Case(
                case_id=Utils.generate_short_case_id(),
                channel_url=self.channel_url,
                message_id_list=message_id_list,
                messages=case_messages,
                summary=case_dict['summary'],
                status=case_dict['status'],
                pending_party=case_dict['pending_party'],
                segmentation_confidence=case_dict['segmentation_confidence'],
                meta=MetaInfo(
                    tracking_numbers=case_dict.get('meta', {}).get('tracking_numbers', []),
                    order_numbers=case_dict.get('meta', {}).get('order_numbers', []),
                    user_names=case_dict.get('meta', {}).get('user_names', [])
                )
            )
                        
            # Perform classification using LLM
            try:
                if self.enable_classification == True:
                    print(f"        📊 Classifying case {case_obj.case_id}")
                    case_obj.classify_case(llm_client)
            except Exception as e:
                print(f"        ⚠️  Classification failed for {case_obj.case_id}: {e}")

            # Perform SOP finding
            try:
                if self.enable_find_sop == True:
                    print(f"        🔍 Finding SOP for case {case_obj.case_id}")
                    case_obj.find_sop()
            except Exception as e:
                print(f"        ⚠️  SOP finding failed for {case_obj.case_id}: {e}")

            case_objects.append(case_obj)
        
        self.cases = case_objects
        
        print(f"    ✅ Cases built successfully ({len(self.cases)} Case objects)")
        return self.cases
    
    def build_cases_via_file(self, output_dir: str, llm_client: 'LLMClient') -> List[Case]:
        """
        从JSON文件加载现有结果并构建Case对象，可选进行LLM分类和SOP查找

        Args:
            output_dir: 输出目录路径
            llm_client: LLM客户端，用于分类和SOP查找

        Returns:
            Case对象列表，包含从文件加载的所有数据
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
                global_cases_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON file: {e}")
        
        # 将字典数据转换为Case对象
        case_objects = []
        for case_dict in global_cases_data:
            # 创建Case对象，使用文件中的所有数据
            message_id_list = case_dict.get('message_id_list', [])


            # 加载Classification相关字段
            has_classification = case_dict.get('has_classification', False)
            main_category = case_dict.get('main_category', 'unknown')
            sub_category = case_dict.get('sub_category', 'unknown')
            classification_reasoning = case_dict.get('classification_reasoning', 'N/A')
            classification_confidence = case_dict.get('classification_confidence', 0.0)
            classification_indicators = case_dict.get('classification_indicators', [])                

            # 加载SOP相关字段
            has_sop = case_dict.get('has_sop', False)
            sop_content = case_dict.get('sop_content', 'N/A')
            sop_url = case_dict.get('sop_url', 'N/A')
            sop_score = case_dict.get('sop_score', 0.0)

            case_obj = Case(
                case_id=case_dict.get('case_id'),
                message_id_list=message_id_list,
                messages=pd.DataFrame(case_dict.get('messages', [])),
                summary=case_dict.get('summary', 'N/A'),
                status=case_dict.get('status', 'ongoing'),
                pending_party=case_dict.get('pending_party', 'N/A'),
                segmentation_confidence=case_dict.get('segmentation_confidence', 0.0),
                channel_url=case_dict.get('channel_url', self.channel_url),
                # 加载分类结果（基于has_classification决定）
                main_category=main_category,
                sub_category=sub_category,
                classification_reasoning=classification_reasoning,
                classification_confidence=classification_confidence,
                classification_indicators=classification_indicators,
                has_classification=has_classification,
                # 加载SOP信息
                has_sop=has_sop,
                sop_content=sop_content,
                sop_url=sop_url,
                sop_score=sop_score,
                # 加载meta信息
                meta=MetaInfo(
                    tracking_numbers=case_dict.get('meta', {}).get('tracking_numbers', []),
                    order_numbers=case_dict.get('meta', {}).get('order_numbers', []),
                    user_names=case_dict.get('meta', {}).get('user_names', [])
                )
            )

            if self.enable_classification == True and case_obj.has_classification == False:
                try:
                    print(f"        📊 Classifying case {case_obj.case_id}")
                    case_obj.classify_case(llm_client)
                except Exception as e:
                    print(f"        ⚠️  Classification failed for {case_obj.case_id}: {e}")

            if self.enable_find_sop == True and case_obj.has_sop == False:
                try:
                    print(f"        🔍 Finding SOP for case {case_obj.case_id}")
                    case_obj.find_sop()
                except Exception as e:
                    print(f"        ⚠️  SOP finding failed for {case_obj.case_id}: {e}")

            case_objects.append(case_obj)

        self.cases = case_objects
        
        print(f"        ✅ Cases loaded from file successfully ({len(self.cases)} Case objects)")
        return self.cases

    def classify_all_cases_via_llm(self, llm_client: 'LLMClient') -> Dict[str, Any]:
        """
        对所有已加载的cases进行LLM分类
        
        Args:
            llm_client: LLM客户端
            
        Returns:
            分类结果统计字典，包含成功/失败数量和详细信息
        """
        if not self.cases:
            raise ValueError("No cases loaded. Call build_cases_via_llm or build_cases_via_file first.")
        
        print(f"    📊 Classifying {len(self.cases)} cases via LLM...")
        
        results = {
            "total_cases": len(self.cases),
            "successful_classifications": 0,
            "failed_classifications": 0,
            "failures": []
        }
        
        for case_obj in self.cases:
            print(f"        📊 Classifying case {case_obj.case_id}")
            try:
                case_obj.classify_case(llm_client)
                results["successful_classifications"] += 1
            except Exception as e:
                print(f"        ⚠️  Classification failed for {case_obj.case_id}: {e}")
                results["failed_classifications"] += 1
                results["failures"].append({
                    "case_id": case_obj.case_id,
                    "error": str(e)
                })
        
        success_rate = (results["successful_classifications"] / results["total_cases"]) * 100
        print(f"    ✅ Classification complete: {results['successful_classifications']}/{results['total_cases']} successful ({success_rate:.1f}%)")
        
        return results

    def classify_all_cases_via_file(self, output_dir: str) -> Dict[str, Any]:
        """
        从JSON文件更新所有cases的分类信息，不进行LLM调用
        
        Args:
            output_dir: 输出目录路径
            
        Returns:
            更新结果统计字典，包含成功/失败数量和详细信息
        """
        if not self.cases:
            raise ValueError("No cases loaded. Call build_cases_via_llm or build_cases_via_file first.")
        
        import json
        
        # 构建文件路径
        session_folder = os.path.join(output_dir, f"session_{self.session}")
        channel_name = Utils.format_channel_for_display(self.channel_url)
        channel_cases_file = os.path.join(session_folder, f"cases_{channel_name}.json")
        
        if not os.path.exists(channel_cases_file):
            raise FileNotFoundError(f"JSON file not found: {channel_cases_file}")
        
        print(f"    📄 Updating {len(self.cases)} cases classification from file...")
        
        # 加载JSON数据
        try:
            with open(channel_cases_file, 'r', encoding='utf-8') as f:
                file_cases_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON file: {e}")
        
        # 创建case_id到文件数据的映射
        file_cases_by_id = {case_data.get('case_id'): case_data for case_data in file_cases_data}
        
        results = {
            "total_cases": len(self.cases),
            "updated_cases": 0,
            "not_found_in_file": 0,
            "not_found_cases": []
        }
        
        for case_obj in self.cases:
            case_id = case_obj.case_id
            if case_id in file_cases_by_id:
                file_case_data = file_cases_by_id[case_id]
                
                # 更新分类相关字段
                case_obj.main_category = file_case_data.get('main_category', 'unknown')
                case_obj.sub_category = file_case_data.get('sub_category', 'unknown')
                case_obj.classification_reasoning = file_case_data.get('classification_reasoning', 'N/A')
                case_obj.classification_confidence = file_case_data.get('classification_confidence', 0.0)
                case_obj.classification_indicators = file_case_data.get('classification_indicators', [])
                
                results["updated_cases"] += 1
                print(f"        📄 Updated classification for case {case_id}")
            else:
                results["not_found_in_file"] += 1
                results["not_found_cases"].append(case_id)
                print(f"        ⚠️  Case {case_id} not found in file data")
        
        update_rate = (results["updated_cases"] / results["total_cases"]) * 100
        print(f"    ✅ Classification update complete: {results['updated_cases']}/{results['total_cases']} updated ({update_rate:.1f}%)")
        
        return results

    def save_results_to_json(self, output_dir: str) -> None:
        """Save channel cases to JSON file"""
        import json
        
        # Create session folder for organized output
        session_folder = os.path.join(output_dir, f"session_{self.session}")
        os.makedirs(session_folder, exist_ok=True)
        
        # Save channel cases to JSON in session folder
        channel_name = Utils.format_channel_for_display(self.channel_url)
        channel_cases_file = os.path.join(session_folder, f"cases_{channel_name}.json")
        save_result = [case.to_dict() for case in self.cases]
        
        try:
            with open(channel_cases_file, 'w', encoding='utf-8') as f:
                json.dump(save_result, f, indent=2, ensure_ascii=False)
            print(f"            Channel cases saved to: {channel_cases_file}")
        except IOError as e:
            print(f"                ❌ Error saving JSON file: {e}")
            raise
    
    def save_results_to_csv(self, output_dir: str) -> None:
        """Save annotated messages to CSV file"""
        # Note: File Summary is restored from case.messages during CSV export
        # (case.messages contains the vision analysis data even if df_clean lost it)

        # Generate annotated CSV for this channel
        df_annotated = self.df_clean.copy()
        df_annotated['case_id'] = "unassigned"  # Default: unassigned (string type)
        # Add classification columns (only main_category and sub_category)
        df_annotated['main_category'] = "unknown"
        df_annotated['sub_category'] = "unknown"
        df_annotated['sop_url'] = "N/A"
        df_annotated['sop_score'] = 0.0

        # Map case assignments and classification data using Message ID
        # Also restore File Summary from case.messages (in case df_clean lost it)
        for case_obj in self.cases:
            case_id = case_obj.case_id or "unknown"
            main_category = case_obj.main_category
            sub_category = case_obj.sub_category
            sop_url = case_obj.sop_url
            sop_score = case_obj.sop_score

            # Build a mapping from Message ID to File Summary from case.messages
            if case_obj.messages is not None and not case_obj.messages.empty:
                if 'File Summary' in case_obj.messages.columns:
                    file_summary_map = dict(zip(
                        case_obj.messages['Message ID'],
                        case_obj.messages['File Summary']
                    ))
                else:
                    file_summary_map = {}
            else:
                file_summary_map = {}

            for message_id in case_obj.message_id_list:
                mask = df_annotated['Message ID'] == message_id
                df_annotated.loc[mask, 'case_id'] = case_id
                df_annotated.loc[mask, 'main_category'] = main_category
                df_annotated.loc[mask, 'sub_category'] = sub_category
                df_annotated.loc[mask, 'sop_url'] = sop_url
                df_annotated.loc[mask, 'sop_score'] = sop_score

                # Restore File Summary from case.messages if available
                if message_id in file_summary_map:
                    file_summary_value = file_summary_map[message_id]
                    # Only update if it's not empty/NaN
                    if pd.notna(file_summary_value) and str(file_summary_value).strip():
                        # Ensure File Summary column is object dtype before assigning string
                        if df_annotated['File Summary'].dtype != 'object':
                            df_annotated['File Summary'] = df_annotated['File Summary'].astype('object')
                        df_annotated.loc[mask, 'File Summary'] = str(file_summary_value)
        
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

    def save_results_to_bigquery(self) -> int:
        """
        Save all cases to BigQuery

        遍历所有 cases，调用每个 case 的 save_to_bigquery() 方法

        Returns:
            成功保存的 case 数量
        """
        if not self.cases:
            print("⚠️  No cases to save")
            return 0

        print(f"💾 Saving {len(self.cases)} cases to BigQuery...")
        saved_count = 0

        for i, case in enumerate(self.cases, 1):
            try:
                case.save_to_bigquery()
                saved_count += 1
                print(f"   ✅ [{i}/{len(self.cases)}] Saved case: {case.case_id}")
            except Exception as e:
                print(f"   ❌ [{i}/{len(self.cases)}] Failed to save case {case.case_id}: {e}")

        print(f"✅ Successfully saved {saved_count}/{len(self.cases)} cases to BigQuery")
        return saved_count

    def repair_case_segment_output(self, cases: List[Dict[str, Any]],
                                 chunk_df: pd.DataFrame,
                                 prev_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        对单个 chunk 的 LLM 分段结果进行修复 & 校验（不修改入参）。
        - 去重：同一 msg 出现在多个 case，只保留一个（可解释的择一规则）
        - 未分配：必须挂靠到合理的 case（空消息优先挂靠到相同sender的最近消息）
        - 补齐字段、排序稳定、自检报告

        Note: message_id_list contains actual Message IDs (nanoid strings like "ZH8P72QBVKYDxvyXsmbHN").
              chunk_df must have 'Message ID' column.

        Args
        ----
        cases : List[Dict]      # LLM 输出的 cases (message_id_list contains nanoid strings)
        chunk_df : pd.DataFrame # DataFrame containing messages with 'Message ID' column
        prev_context : Optional[Dict]  # 上一块尾部摘要，用于承接判断

        Returns
        -------
        {
          "cases_out": List[Dict],
          "provisionals": List[Dict],   # 去重/自动挂靠的记录，便于后续复核
          "report": { ... }             # 自检统计
        }
        """
        # 构建 Message ID → 索引位置 的映射，用于排序和距离计算
        msg_id_to_index = {str(mid): idx for idx, mid in enumerate(chunk_df['Message ID'].tolist())}

        # 内部helper函数
        def _ensure_case_schema(c: Dict[str, Any]) -> Dict[str, Any]:
            """补齐字段、规范类型，不改入参（在外层会 deepcopy）"""
            # Import from ChannelSegmenter constants
            REQUIRED_FIELDS_DEFAULTS = {
                "summary": "N/A",
                "status": "ongoing",
                "pending_party": "N/A",
                "segmentation_confidence": 0.0,
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
            
            if "message_id_list" not in c or not isinstance(c["message_id_list"], list):
                c["message_id_list"] = []
            # 统一字符串 + 按 chunk_df 索引排序去重
            c["message_id_list"] = sorted({str(x) for x in c["message_id_list"]}, key=lambda x: msg_id_to_index.get(x, 0))

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
                c["segmentation_confidence"] = float(c.get("segmentation_confidence", 0.0))
            except Exception:
                c["segmentation_confidence"] = 0.0
            c["segmentation_confidence"] = max(0.0, min(1.0, c["segmentation_confidence"]))

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

        def _proximity_score(i: str, case: Dict[str, Any]) -> float:
            ml = case.get("message_id_list", [])
            if not ml: return 0.0
            i_idx = msg_id_to_index.get(i, 0)
            dist = min(abs(i_idx - msg_id_to_index.get(m, 0)) for m in ml)
            return 1.0 / (1 + dist)  # 1, 0.5, 0.33, ...

        def _choose_one_for_duplicate(i: str, cases: List[Dict[str, Any]], cids: List[int], prev_context: Optional[Dict[str, Any]]) -> int:
            # 规则：anchor_strength > 承接(prev_context) > segmentation_confidence > proximity > 较小 case_id（稳定）
            scored = []
            for cid in cids:
                c = cases[cid]
                scored.append((
                    _anchor_strength(c),
                    1 if _hits_active_hints(c, prev_context) else 0,
                    float(c.get("segmentation_confidence", 0.0)),
                    _proximity_score(i, c),
                    -cid,  # 反向用于最后的稳定 tie-break（越小优先）
                    cid
                ))
            scored.sort(reverse=True)
            return scored[0][-1]

        def _is_empty_message(msg_id: str) -> bool:
            """检查消息内容是否为空或空白"""
            if msg_id not in chunk_df['Message ID'].values:
                return True
            message = chunk_df[chunk_df['Message ID'] == msg_id].iloc[0]
            text = str(message.get('Message', '')).strip()  # Use 'Message' column as seen in Utils.format_messages_for_prompt
            return len(text) == 0
        
        def _find_nearest_same_sender_case(msg_id: str, cases: List[Dict]) -> Optional[int]:
            """查找包含最近的相同sender_id消息的case"""
            if msg_id not in chunk_df['Message ID'].values:
                return None

            target_sender = chunk_df[chunk_df['Message ID'] == msg_id].iloc[0].get('Sender ID', '')
            if not target_sender:
                return None

            # 构建消息到case的映射
            msg_to_case = {}
            for case_idx, case in enumerate(cases):
                for message_id in case.get('message_id_list', []):
                    msg_to_case[message_id] = case_idx

            # 寻找最近的相同sender消息（基于索引距离）
            msg_idx = msg_id_to_index.get(msg_id, 0)
            best_distance = float('inf')
            best_case_id = None

            for check_msg_id, case_idx in msg_to_case.items():
                if check_msg_id in chunk_df['Message ID'].values:
                    check_sender = chunk_df[chunk_df['Message ID'] == check_msg_id].iloc[0].get('Sender ID', '')
                    if check_sender == target_sender:
                        distance = abs(msg_id_to_index.get(check_msg_id, 0) - msg_idx)
                        if distance < best_distance:
                            best_distance = distance
                            best_case_id = case_idx

            return best_case_id
        
        def _attach_unassigned_smart(msg_id: str, cases: List[Dict]) -> Optional[int]:
            """智能挂靠逻辑（基于原_attach_unassigned_simple）"""
            scored = []
            for cid, c in enumerate(cases):
                # 仅考虑 open/ongoing/blocked 的，跳过 resolved
                if c.get("status") == "resolved":
                    continue
                scored.append((
                    1 if _hits_active_hints(c, prev_context) else 0,
                    _anchor_strength(c),
                    _proximity_score(msg_id, c),
                    float(c.get("segmentation_confidence", 0.0)),
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
        
        def _attach_to_any_nearest_case(msg_id: str, cases: List[Dict]) -> int:
            """终极兜底：挂靠到任何最近的case"""
            if not cases:
                return 0  # 如果没有cases，返回第一个（这种情况理论上不应该发生）

            # 找到包含最近消息的case（基于索引距离）
            msg_idx = msg_id_to_index.get(msg_id, 0)
            best_distance = float('inf')
            best_case_id = 0

            for case_idx, case in enumerate(cases):
                for message_id in case.get('message_id_list', []):
                    distance = abs(msg_id_to_index.get(message_id, 0) - msg_idx)
                    if distance < best_distance:
                        best_distance = distance
                        best_case_id = case_idx

            return best_case_id
        
        def _attach_to_case(msg_id: str, case_id: int, cases: List[Dict], provisionals: List[Dict], reason: str):
            """执行挂靠操作"""
            if case_id < len(cases):
                cases[case_id]["message_id_list"].append(msg_id)
                cases[case_id]["message_id_list"] = sorted(set(cases[case_id]["message_id_list"]), key=lambda x: msg_id_to_index.get(x, 0))
                provisionals.append({
                    "type": "auto_attach",
                    "msg_idx": msg_id,
                    "attached_to": case_id,
                    "reason": reason
                })
        
        # 使用实际的Message ID列表（字符串）
        chunk_msg_ids = [str(x) for x in chunk_df['Message ID'].tolist()]
            
        out = copy.deepcopy(cases)

        # 1) 规范化 & 补齐字段
        for idx in range(len(out)):
            out[idx] = _ensure_case_schema(out[idx])

        # 2) case 内排序稳定 + 去空 case
        out = [c for c in out if c["message_id_list"]]
        out.sort(key=lambda c: (msg_id_to_index.get(c["message_id_list"][0], 0), c.get("segmentation_confidence", 0.0) * -1))

        # 3) 建立 msg -> cases 反查
        msg_to_cases = defaultdict(list)
        for cid, c in enumerate(out):
            for i in c["message_id_list"]:
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
                ml = out[cid]["message_id_list"]
                if i in ml:
                    out[cid]["message_id_list"] = [x for x in ml if x != i]
            provisionals.append({
                "type": "duplicate_resolution",
                "msg_idx": i,
                "chosen_case": winner,
                "rejected_cases": losers,
                "reason": "anchor > continuation > segmentation_confidence > proximity > case_id"
            })

        # 5) 再次清理空 case
        out = [c for c in out if c["message_id_list"]]
        # 重新构建 msg 索引
        msg_to_cases.clear()
        for cid, c in enumerate(out):
            for i in c["message_id_list"]:
                msg_to_cases[i].append(cid)

        # 6) 识别未分配
        chunk_set = set(str(x) for x in chunk_msg_ids)
        assigned = set(msg_to_cases.keys())
        unassigned = sorted(list(chunk_set - assigned), key=lambda x: msg_id_to_index.get(x, 0))

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

        # 8) 最终排序稳定（按最小 msg 索引升序）
        out.sort(key=lambda c: msg_id_to_index.get(c["message_id_list"][0], 0))

        # 9) 自检报告
        # 9.1 统计重复与覆盖率
        final_msg_to_cases = defaultdict(int)
        for c in out:
            for i in c["message_id_list"]:
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
    
    def process_file_type_messages(self, llm_client: 'LLMClient') -> None:
        """Process FILE messages with vision analysis and populate File Summary column."""
        print("        🔍 Processing FILE messages with vision analysis...")
        
        # Find FILE messages with image URLs in this channel
        file_messages = self.df_clean[
            (self.df_clean['Type'] == 'FILE') & 
            (self.df_clean['File URL'].notna())
        ].copy()
        
        if len(file_messages) == 0:
            print("            No image FILE messages found for vision processing")
            return
            
        print(f"            Found {len(file_messages)} image FILE messages to process")
            
        # Process each image
        processed_count = 0
        for idx, (df_idx, file_msg) in enumerate(file_messages.iterrows()):
            try:
                image_url = file_msg['File URL']
                message_id = file_msg['Message ID']

                print(f"            Processing image {idx + 1}/{len(file_messages)}: {image_url.split('/')[-1]}")

                # Get context for this image using static method
                context_df = VisionProcessor.get_context_for_image(
                    channel_df=self.df_clean,
                    image_message_id=message_id,
                    context_size=3  # Smaller context for batch processing
                )
                
                # Analyze image with context using class method
                analysis_result = VisionProcessor.analyze_image_with_context(
                    context_df=context_df,
                    image_url=image_url,
                    llm_client=llm_client
                )
                
                # Generate synthesized text summary using static method
                summary_text = VisionProcessor.synthesize_visual_text(analysis_result)

                # Update File Summary in df_clean
                self.df_clean.loc[df_idx, 'File Summary'] = summary_text

                processed_count += 1
                
            except Exception as e:
                print(f"            ⚠️  Error processing image {image_url}: {e}")
                continue
                
        print(f"            ✅ Successfully processed {processed_count}/{len(file_messages)} images")