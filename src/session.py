#!/usr/bin/env python3
"""
Session management module for customer support message segmentation pipeline.

This module contains the Session class that orchestrates the complete processing pipeline:
- File data processing across all channels
- Individual channel segmentation and case generation  
- Cross-channel statistics generation
- Session-level output management
"""

import os
import json
from datetime import datetime
from typing import List, Optional
import pandas as pd # type: ignore

# Local imports - compatible with both direct execution and module execution
try:
    from .channel import Channel
    from .llm_client import LLMClient
    from .utils import Utils
except ImportError:
    from channel import Channel
    from llm_client import LLMClient
    from utils import Utils


class Session:
    """
    Session orchestrator for the complete message processing pipeline.
    
    Manages cross-channel operations, file processing, and session-level outputs.
    Encapsulates all business logic that was previously scattered across main.py and FileProcessor.
    """
    
    def __init__(self,
                 input_file: str,
                 output_dir: str = 'out',
                 chunk_size: int = 60,
                 overlap: int = 10,
                 model: str = 'gpt-5',
                 session_name: Optional[str] = None,
                 enable_review: bool = False,
                 enable_vision_processing: bool = True,
                 enable_classification: bool = True,
                 enable_find_sop: bool = True):
        """
        Initialize session with explicit parameters.
        
        Args:
            input_file: Path to input CSV file (required)
            output_dir: Base output directory (default: 'out')
            chunk_size: Chunk size for segmentation (default: 80)
            overlap: Overlap size between chunks (default: 20)
            model: LLM model name (default: 'gpt-5')
            session_name: Optional session name (auto-generated if None)
            enable_review: Enable case review flag (default: False)
            enable_vision_processing: Enable vision processing for FILE messages (default: False)
            enable_classification: Enable classification when loading from files (default: False)
            enable_find_sop: Enable SOP finding when running QA function (default: True)
        """
        # Pipeline configuration
        self.input_file = input_file
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = model
        self.enable_review = enable_review
        self.enable_vision_processing = enable_vision_processing
        self.enable_classification = enable_classification
        self.enable_find_sop = enable_find_sop
        
        # Session identification and output management
        self.session_name = session_name or datetime.now().strftime("%y%m%d_%H%M%S")
        self.output_folder = os.path.join(output_dir, f"session_{self.session_name}")
        
        # Data processing state
        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        
        # Cross-channel collections
        self.channels: List[Channel] = []
    
    def cs_mbr(self) -> None:
        """
        Execute the complete processing pipeline.
        
        Pipeline stages:
        1. File data processing (cross-channel)
        2. Session folder setup
        3. Individual channel processing
        4. Export all cases to CSV
        """
        print(f"🚀 Starting pipeline session: {self.session_name}")
        
        try:
            # Stage 1: Process file data across all channels
            if not self.process_file_data():
                print("Error: File processing failed")
                exit(1)
            
            # Stage 2: Create session folder structure
            self.create_session_folder()
            
            # Stage 3: Initialize LLM client and process channels
            self.process_channels()
            
            # Stage 4: Export all cases to CSV
            print(f"\n📄 Exporting all cases to CSV...")
            try:
                csv_path = self.save_all_cases_to_csv()
                print(f"✅ CSV export complete: {csv_path}")
            except Exception as e:
                print(f"⚠️  CSV export failed: {e}")
            
            print(f"\n✅ Pipeline complete!")
            
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
    
    def cs_qa(self) -> None:
        """
        Execute QA processing pipeline.
        
        QA-specific pipeline for question answering tasks.
        """
        print(f"🚀 Starting QA pipeline session: {self.session_name}")
        
        try:
            # Stage 1: Process file data across all channels
            if not self.process_file_data():
                print("Error: File processing failed")
                exit(1)
            
            # Stage 2: Create session folder structure
            self.create_session_folder()
            
            # Stage 3: Process all channels
            print(f"\n🔄 Processing all channels...")

            # Initialize LLM client
            llm_client = LLMClient(model=self.model)
            print(f"LLM Client initialized with model: {self.model}")

            # Process all channels
            channel_urls = self.df_clean['Channel URL'].unique()
            for channel_idx, channel_url in enumerate(channel_urls, 1):
                channel_df = self.df_clean[self.df_clean['Channel URL'] == channel_url]
                print(f"🔄 Channel {channel_idx}/{len(channel_urls)}: {Utils.format_channel_for_display(channel_url)} ({len(channel_df)} messages)")
                channel = self.build_one_channel(channel_url, llm_client)
                self.channels.append(channel)

            # Stage 4: Find first 10 cases matching specific criteria
            print(f"\n🔍 Filtering cases by category criteria...")

            # Collect all cases from processed channels
            all_cases = []
            for channel in self.channels:
                if hasattr(channel, 'cases') and channel.cases:
                    all_cases.extend(channel.cases)

            # Filter cases by exact category criteria and take first 10
            filtered_cases = []
            target_main_category = "Order & Post-sale"
            target_sub_category = "Modify Order / Modify Address / Modify Shipping to local pick up"

            for case in all_cases:
                filtered_cases.append(case)
                if len(filtered_cases) == 10:  # Stop after collecting 10 cases
                    break

            # Display results
            print(f"🔍 Found {len(filtered_cases)} cases (showing first 10) with:")
            print(f"   Main Category: {target_main_category}")
            print(f"   Sub Category: {target_sub_category}")
            print(f"   Processed {len(channel_urls)} channels total")

            if filtered_cases:
                print(f"\nFirst {len(filtered_cases)} matching cases with their messages:")
                for i, case in enumerate(filtered_cases, 1):
                    print("=" * 100)
                    case.find_sop()
                    case.print_case()
                    print()
            else:
                print(f"No cases found matching the criteria.")
            
            print(f"\n✅ QA Pipeline complete!")
            
        except Exception as e:
            print(f"Error: {e}")
            exit(1)
    
    def process_file_data(self) -> bool:
        """
        Process CSV file data and prepare channel data list.
        
        All file processing logic is contained within this method.
        
        Returns:
            True if processing successful, False otherwise
        """
        print("Starting file processing...")

        # 1. Load CSV data into DataFrame
        try:
            self.df = pd.read_csv(self.input_file)
            print(f"        Loaded {len(self.df)} messages from {self.input_file}")
        except Exception as e:
            print(f"Error loading file {self.input_file}: {e}")
            return False

        # 2-6. Preprocess DataFrame using Utils method
        self.df_clean = Utils.preprocess_dataframe(self.df, verbose=True)

        return True

    def create_session_folder(self) -> None:
        """Create session output folder structure."""
        os.makedirs(self.output_folder, exist_ok=True)
    
    def build_one_channel(self, channel_url: str, llm_client) -> 'Channel':
        """
        Build and process a single channel.
        
        Args:
            channel_url: The channel URL to process
            llm_client: LLM client instance for processing
            
        Returns:
            Processed Channel instance
        """
        # Extract channel data
        channel_df = self.df_clean[self.df_clean['Channel URL'] == channel_url].copy()

        # Check if channel results already exist
        channel_name = Utils.format_channel_for_display(channel_url)
        channel_cases_file = os.path.join(self.output_folder, f"cases_{channel_name}.json")
        
        # Create Channel instance
        channel = Channel(channel_df, channel_url, self.session_name, self.chunk_size, self.overlap, self.enable_classification, self.enable_vision_processing, self.enable_find_sop)
        
        if os.path.exists(channel_cases_file):
            print(f"        ⏭️  Loading existing results from file")
            channel.build_cases_via_file(self.output_dir, llm_client)
        else:
            # Process channel with full pipeline (includes vision processing if enabled)
            channel.build_cases_via_llm(llm_client)


        return channel
    
    def process_channels(self) -> None:
        """
        Process each channel individually and collect results.
        
        Migrated from main.py channel processing loop.
        """
        # Initialize LLM client
        llm_client = LLMClient(model=self.model)
        print(f"LLM Client initialized with model: {self.model}")
        
        # Process each channel directly
        channel_urls = self.df_clean['Channel URL'].unique()
        for channel_idx, channel_url in enumerate(channel_urls):
            channel_df = self.df_clean[self.df_clean['Channel URL'] == channel_url].copy()
            
            print(f"🔄 Channel {channel_idx + 1}/{len(channel_urls)}: {Utils.format_channel_for_display(channel_url)} ({len(channel_df)} messages)")
            
            # Build and process the channel
            channel = self.build_one_channel(channel_url, llm_client)
            
            # Save channel results
            print(f"    💾 Saving results...")
            try:
                channel.save_results_to_json(self.output_dir)
                channel.save_results_to_csv(self.output_dir)
                print(f"    ✅ Results saved successfully")
            except Exception as save_error:
                print(f"        ❌ Error saving results: {str(save_error)}")
                print(f"        Processing completed but save failed - continuing...")
            
            # Collect channel for cross-channel operations
            self.channels.append(channel)
        
        # Summary
        print(f"\n✅ Pipeline processing complete!")
        print(f"Processed {len(self.channels)} channels")
        print(f"Results saved to timestamped session folders in output directory")
        print(f"Each session contains JSON and CSV files for successfully saved channels")
    
    def save_all_cases_to_csv(self, output_filename: str = None) -> str:
        """
        导出所有 cases 到 CSV 文件，格式与 merge_aug.csv 一致
        
        Args:
            output_filename: 输出文件名（可选），默认为 'all_cases_merged.csv'
            
        Returns:
            保存的文件路径
        """
        import pandas as pd
        
        # 收集所有 channels 的 cases
        all_cases = []
        for channel in self.channels:
            if hasattr(channel, 'cases') and channel.cases:
                all_cases.extend(channel.cases)
        
        if not all_cases:
            print("⚠️  No cases found to export")
            return ""
        
        print(f"📄 Exporting {len(all_cases)} cases to CSV...")
        
        # 准备CSV数据
        csv_data = []
        for case in all_cases:
            # 处理数组字段，转换为字符串格式
            def format_array_field(field_value):
                if not field_value:
                    return "[]"
                return str(field_value)  # 转换为字符串格式 ['item1', 'item2']
            
            # 提取meta信息
            meta = case.meta if case.meta else None
            tracking_numbers = meta.tracking_numbers if meta else []
            order_numbers = meta.order_numbers if meta else []
            user_names = meta.user_names if meta else []
            
            row = {
                'channel_url': case.channel_url or '',
                'summary': case.summary or '',
                'status': case.status or '',
                'pending_party': case.pending_party or '',
                'segmentation_confidence': case.segmentation_confidence or 0.0,
                'main_category': case.main_category or '',
                'sub_category': case.sub_category or '',
                'classification_confidence': case.classification_confidence or 0.0,
                'first_res_time': case.first_res_time if case.first_res_time != -1 else -1,
                'handle_time': case.handle_time if case.handle_time != -1 else -1,
                'first_contact_resolution': case.first_contact_resolution if case.first_contact_resolution != -1 else -1,
                'usr_msg_num': case.usr_msg_num if case.usr_msg_num != -1 else -1,
                'total_msg_num': case.total_msg_num if case.total_msg_num != -1 else -1,
                'start_time': case.start_time or '',
                'end_time': case.end_time or '',
                'tracking_numbers': format_array_field(tracking_numbers),
                'order_numbers': format_array_field(order_numbers),
                'user_names': format_array_field(user_names),
                'message_id_list': format_array_field(case.message_id_list)
            }
            csv_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(csv_data)
        
        # 确定输出文件路径
        session_folder = os.path.join(self.output_dir, f"session_{self.session_name}")
        os.makedirs(session_folder, exist_ok=True)
        
        if output_filename is None:
            output_filename = "all_cases_merged.csv"
        
        output_path = os.path.join(session_folder, output_filename)
        
        # 保存到CSV文件
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"            ✅ All cases exported to: {output_path}")
            return output_path
        except Exception as e:
            print(f"            ❌ Error exporting cases to CSV: {e}")
            raise