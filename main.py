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
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv # type: ignore
import anthropic # type: ignore
from datetime import datetime

# Load environment variables
load_dotenv()


@dataclass
class Chunk:
    """Data structure for a single chunk of messages"""
    chunk_id: int                    # Sequential chunk ID (0, 1, 2, ...)
    channel_url: str                 # Channel this chunk belongs to
    start_idx: int                   # Start index in the channel (inclusive)
    end_idx: int                     # End index in the channel (exclusive) - half-open interval [start, end)
    messages: pd.DataFrame           # DataFrame slice with messages in this chunk
    total_messages: int              # Number of messages in this chunk
    has_overlap_with_previous: bool  # Whether this chunk overlaps with previous chunk
    overlap_size: int                # Number of overlapping messages with previous chunk
    tail_summary: Optional[str] = None  # Generated tail summary for next chunk

    def get_message_indices(self) -> List[int]:
        """Get list of msg_ch_idx values for messages in this chunk"""
        return self.messages['msg_ch_idx'].tolist()
    
    def format_for_prompt(self) -> str:
        """Format chunk messages as: message_index | sender id | role | timestamp | text"""
        formatted_lines = []
        for _, row in self.messages.iterrows():
            # Handle NaN messages and replace newlines with spaces to keep one line per message
            message_text = str(row['Message']).replace('\n', ' ').replace('\r', ' ')
            if message_text == 'nan':
                message_text = ''
            
            formatted_lines.append(
                f"{row['msg_ch_idx']} | {row['Sender ID']} | {row['role']} | {row['Created Time']} | {message_text}"
            )
        return '\n'.join(formatted_lines)
    
    def get_tail_messages(self, overlap_size: int) -> List[str]:
        """Get the last N messages from this chunk for tail summary"""
        if len(self.messages) == 0:
            return []
        
        # Get the last 'overlap_size' messages, but at least 5 if available
        num_messages = max(min(overlap_size, len(self.messages)), min(5, len(self.messages)))
        tail_messages = self.messages.tail(num_messages)
        
        formatted_messages = []
        for _, row in tail_messages.iterrows():
            # Format: msg_ch_idx | sender id=sender_id | role=role | ISO timestamp | text=truncated text
            message_text = str(row['Message']).replace('\n', ' ').replace('\r', ' ')
            if message_text == 'nan':
                message_text = ''
            
            # Truncate long messages
            if len(message_text) > 150:
                message_text = message_text[:150] + '...'
            
            formatted_messages.append(
                f"- {row['msg_ch_idx']} | sender id={row['Sender ID']} | role={row['role']} | {row['Created Time']} | text={message_text}"
            )
        
        return formatted_messages
    
    def extract_case_hints(self, complete_cases: List[Dict[str, Any]]) -> List[str]:
        """Extract active case hints from complete_cases for tail summary"""
        if not complete_cases:
            return ["None"]
        
        hints = []
        for case in complete_cases[:5]:  # Max 5 hints as per prompt
            status = case.get('status', 'open').lower()
            
            # Only include unresolved cases
            if status in ['open', 'ongoing', 'blocked']:
                msg_list = case.get('msg_list', [])
                summary = case.get('summary', 'Case summary not available')
                
                # Extract key information from summary (simplified version)
                hint = f"- topic: \"{summary[:50]}{'...' if len(summary) > 50 else ''}\"\n"
                hint += f"  status: \"{status}\"\n"
                hint += f"  evidence_msg_ch_idx: {msg_list}"
                
                hints.append(hint)
        
        return hints if hints else ["None"]
    
    def generate_tail_summary(self, complete_cases: List[Dict[str, Any]], overlap_size: int, 
                            llm_client: 'LLMClient', previous_context: str = "") -> str:
        """Generate tail summary using LLM for the next chunk"""
        # Load the prompt template
        try:
            prompt_template = load_prompt("tail_summary_prompt.md")
        except FileNotFoundError as e:
            raise RuntimeError(f"Cannot load tail summary prompt: {e}")
        
        # Get recent messages from this chunk
        recent_messages = self.get_tail_messages(overlap_size)
        recent_messages_block = '\n'.join(recent_messages)
        
        # Get current chunk messages for analysis
        chunk_messages_block = self.format_for_prompt()
        
        # Extract case hints
        case_hints = self.extract_case_hints(complete_cases)
        case_hints_block = '\n'.join(case_hints)
        
        # Get time window
        if len(self.messages) > 0:
            start_time = self.messages.iloc[0]['Created Time']
            end_time = self.messages.iloc[-1]['Created Time']
            time_window = f'["{start_time}", "{end_time}"]'
        else:
            time_window = '["N/A", "N/A"]'
        
        # Build previous context block
        if previous_context:
            context_block = previous_context
        else:
            # Build context block for this chunk
            context_block = f"""ACTIVE_CASE_HINTS:
{case_hints_block}

RECENT_MESSAGES:
{recent_messages_block}

META (optional):
- overlap: {overlap_size}
- channel: {self.channel_url[:50]}{'...' if len(self.channel_url) > 50 else ''}
- time_window: {time_window}"""
        
        # Replace placeholders in prompt
        final_prompt = prompt_template.replace(
            "<<<INSERT_PREVIOUS_CONTEXT_SUMMARY_BLOCK_HERE>>>", 
            context_block
        ).replace(
            "<<<INSERT_CHUNK_BLOCK_HERE>>>", 
            chunk_messages_block
        )
        
        # Generate tail summary using LLM
        try:
            self.tail_summary = llm_client.generate(final_prompt, call_label="tail_summary")
            return self.tail_summary
        except Exception as e:
            raise RuntimeError(f"Failed to generate tail summary for chunk {self.chunk_id}: {e}")
    
    def generate_case_segments(self, 
                             current_messages: str, 
                             previous_tail_summary: Optional[str], 
                             llm_client: 'LLMClient') -> Dict[str, Any]:
        """Generate case segments using LLM for current chunk messages"""
        # Load the segmentation prompt template
        try:
            prompt_template = load_prompt("segmentation_prompt.md")
        except FileNotFoundError as e:
            raise RuntimeError(f"Cannot load segmentation prompt: {e}")
        
        # Handle previous context
        if previous_tail_summary is None:
            context_block = "No previous context"
        else:
            context_block = previous_tail_summary
        
        # Replace placeholders in prompt template
        final_prompt = prompt_template.replace(
            "<<<INSERT_PREVIOUS_CONTEXT_SUMMARY_BLOCK_HERE>>>", 
            context_block
        ).replace(
            "<<<INSERT_CHUNK_BLOCK_HERE>>>", 
            current_messages
        )
        
        # Generate case segments using LLM
        try:
            response = llm_client.generate(final_prompt, call_label="case_segmentation")
            
            # Parse JSON response
            import json
            try:
                # Try direct JSON parsing first
                result = json.loads(response)
                
                # Validate and fix case segmentation results
                result = self._validate_and_fix_result(result)
                
            except json.JSONDecodeError:
                # Fallback: extract JSON from response using regex
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Validate and fix case segmentation results for fallback parsing too
                    result = self._validate_and_fix_result(result)
                else:
                    raise ValueError("No valid JSON found in LLM response")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate case segments for chunk {self.chunk_id}: {e}")
    
    def _validate_case_segmentation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate case segmentation results for coverage and uniqueness"""
        complete_cases = result.get('complete_cases', [])
        
        # Extract all assigned messages and track assignments
        all_assigned_messages = []
        message_to_cases = {}
        
        for case_idx, case in enumerate(complete_cases):
            msg_list = case.get('msg_list', [])
            case_id = case_idx + 1
            
            for msg_idx in msg_list:
                all_assigned_messages.append(msg_idx)
                
                if msg_idx not in message_to_cases:
                    message_to_cases[msg_idx] = []
                message_to_cases[msg_idx].append(case_id)
        
        # Find issues
        expected_messages = set(range(self.total_messages))
        assigned_messages = set(all_assigned_messages)
        missing_messages = sorted(expected_messages - assigned_messages)
        duplicate_assignments = {msg: cases for msg, cases in message_to_cases.items() if len(cases) > 1}
        
        # Calculate coverage
        coverage_percentage = len(assigned_messages) / self.total_messages * 100
        
        # Generate warnings
        warnings = []
        if missing_messages:
            warnings.append(f"Missing {len(missing_messages)} messages: {missing_messages}")
        if duplicate_assignments:
            warnings.append(f"Duplicate assignments for {len(duplicate_assignments)} messages:")
            for msg_idx, cases in sorted(duplicate_assignments.items()):
                warnings.append(f"  Message {msg_idx} in cases: {cases}")
        if coverage_percentage < 100:
            warnings.append(f"Coverage: {coverage_percentage:.1f}% ({len(assigned_messages)}/{self.total_messages})")
        
        return {
            'warnings': warnings,
            'coverage': coverage_percentage,
            'missing': missing_messages,
            'duplicates': duplicate_assignments,
            'total_cases': len(complete_cases),
            'unique_assigned': len(assigned_messages)
        }
    
    def _fix_assignment_issues(self, result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply simple policies to fix missing and duplicate message assignments"""
        complete_cases = result.get('complete_cases', [])
        actions_taken = []
        
        # Fix missing messages
        for missing_msg in validation_result.get('missing', []):
            best_case_idx = self._find_closest_case(missing_msg, complete_cases)
            case_summary = complete_cases[best_case_idx]['summary'][:50] + "..."
            complete_cases[best_case_idx]['msg_list'].append(missing_msg)
            complete_cases[best_case_idx]['msg_list'].sort()
            actions_taken.append(f"  ➕ Added missing msg {missing_msg} to case {best_case_idx + 1}: \"{case_summary}\"")
        
        # Fix duplicate assignments (confidence-based)
        for msg_idx, case_list in validation_result.get('duplicates', {}).items():
            # Find case with highest confidence, or first case if tied
            best_case_id = self._select_best_case_for_message(case_list, complete_cases)
            best_case_summary = complete_cases[best_case_id - 1]['summary'][:50] + "..."
            best_confidence = complete_cases[best_case_id - 1].get('confidence', 0)
            
            removed_from = []
            # Remove message from all cases except the best one
            for case_idx, case in enumerate(complete_cases):
                case_id = case_idx + 1
                if msg_idx in case['msg_list'] and case_id != best_case_id:
                    case['msg_list'].remove(msg_idx)
                    removed_from.append(str(case_id))
            
            if removed_from:
                actions_taken.append(f"  🎯 Kept msg {msg_idx} in case {best_case_id} (confidence: {best_confidence}): \"{best_case_summary}\"")
                actions_taken.append(f"     Removed from cases: {', '.join(removed_from)}")
        
        # Log all actions taken
        if actions_taken:
            print("🔧 Policy actions taken:")
            for action in actions_taken:
                print(action)
        
        return result
    
    def _find_closest_case(self, missing_msg: int, complete_cases: List[Dict[str, Any]]) -> int:
        """Find case index with messages closest to missing_msg"""
        min_distance = float('inf')
        best_case_idx = 0
        
        for case_idx, case in enumerate(complete_cases):
            msg_list = case.get('msg_list', [])
            if not msg_list:
                continue
                
            # Calculate minimum distance to any message in this case
            distance = min(abs(missing_msg - msg) for msg in msg_list)
            
            # Prefer smaller cases if distance is equal
            if distance < min_distance or (distance == min_distance and len(msg_list) < len(complete_cases[best_case_idx]['msg_list'])):
                min_distance = distance
                best_case_idx = case_idx
        
        return best_case_idx
    
    def _select_best_case_for_message(self, case_list: List[int], complete_cases: List[Dict[str, Any]]) -> int:
        """Select best case for duplicated message based on confidence, then order"""
        best_case_id = case_list[0]  # Default to first case
        best_confidence = complete_cases[best_case_id - 1].get('confidence', 0)
        
        for case_id in case_list:
            case_confidence = complete_cases[case_id - 1].get('confidence', 0)
            if case_confidence > best_confidence:
                best_confidence = case_confidence
                best_case_id = case_id
        
        return best_case_id
    
    def _validate_and_fix_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate case segmentation result and apply fixes if needed"""
        validation_result = self._validate_case_segmentation(result)
        
        if validation_result['warnings']:
            print(f"⚠️ Validation warnings for chunk {self.chunk_id}:")
            for warning in validation_result['warnings']:
                print(f"  {warning}")
            
            # Apply fix policy
            result = self._fix_assignment_issues(result, validation_result)
            print("🔧 Applied assignment fix policy")
            
            # Re-validate to confirm fixes
            final_validation = self._validate_case_segmentation(result)
            if not final_validation['warnings']:
                print("✅ All assignment issues resolved - 100% coverage achieved")
                print(f"   Final: {final_validation['coverage']:.1f}% ({final_validation['total_cases']} cases)")
            else:
                print("⚠️ Some issues remain after policy application")
                for warning in final_validation['warnings']:
                    print(f"  {warning}")
        else:
            print(f"✅ Case segmentation validation passed for chunk {self.chunk_id}")
            print(f"   Coverage: {validation_result['coverage']:.1f}% ({validation_result['total_cases']} cases)")
        
        return result


def load_prompt(filename: str) -> str:
    """Load prompt template from prompts directory"""
    prompt_path = os.path.join("prompts", filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


class LLMClient:
    """Simple LLM client for Claude API calls"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found. Please set it in .env file or pass as argument")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate(self, prompt: str, call_label: str = "unknown", max_tokens: int = 4000) -> str:
        """Generate response using Claude API with debug logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create debug output directory if it doesn't exist
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Generate debug log filename
        debug_file = os.path.join(debug_dir, f"{call_label}_{timestamp}.log")
        
        try:
            # Log the request
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== LLM CALL DEBUG LOG ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Call Label: {call_label}\n")
                f.write(f"Model: {self.model}\n")
                f.write(f"Max Tokens: {max_tokens}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write("\n=== PROMPT ===\n")
                f.write(prompt)
                f.write("\n\n")
            
            # Make the API call
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response = message.content[0].text
            
            # Log the successful response
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=== RESPONSE ===\n")
                f.write(response)
                f.write(f"\n\nResponse Length: {len(response)} characters\n")
                f.write("\n=== STATUS ===\n")
                f.write("Success: LLM call completed successfully\n")
            
            print(f"Debug log saved: {debug_file}")
            return response
            
        except Exception as e:
            # Log the error
            try:
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write("=== ERROR ===\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Error Type: {type(e).__name__}\n")
                    f.write("\n=== STATUS ===\n")
                    f.write(f"Failed: LLM call failed - {str(e)}\n")
            except:
                pass  # If debug logging fails, don't break the main functionality
            
            raise RuntimeError(f"LLM generation failed ({call_label}): {e}")


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
    
    def process(self) -> Optional[pd.DataFrame]:
        """Execute the complete processing pipeline"""
        print("Starting file processing...")
        
        if not self.load_data():
            return None
            
        # Execute processing pipeline
        self.add_role_column()
        self.process_time_columns()
        self.sort_and_group_data()
        self.add_message_index()
        self.create_clean_dataframe()
        
        # Save output
        output_path = self.save_output()
        
        print(f"Processing complete! Output saved to {output_path}")
        print(f"Processed {len(self.df_clean)} messages across {self.df_clean['Channel URL'].nunique()} channels")
        
        return self.df_clean


class ChannelSegmenter:
    """
    Segments processed messages into overlapping chunks for LLM analysis.
    
    Features:
    - Channel separation: Each channel is processed independently
    - Half-open intervals: Uses [start, end) to avoid boundary duplication
    - Overlap validation: Ensures overlap < chunk_size/3 for optimal coverage
    - Chunk tracking: Maintains overlap metadata for context continuity
    """
    
    def __init__(self, df_clean: pd.DataFrame, chunk_size: int = 80, overlap: int = 20):
        self.df_clean = df_clean
        self.chunk_size = chunk_size
        self.overlap = overlap
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
        """Generate chunks for all channels"""
        self.chunks = []
        chunk_id_counter = 0
        
        # Process each channel separately
        for channel_url in self.df_clean['Channel URL'].unique():
            channel_df = self.df_clean[self.df_clean['Channel URL'] == channel_url].copy()
            
            # Generate chunks for this channel
            channel_chunks = self.get_channel_chunks(channel_df, channel_url, chunk_id_counter)
            self.chunks.extend(channel_chunks)
            chunk_id_counter += len(channel_chunks)
        
        print(f"Generated {len(self.chunks)} chunks across {self.df_clean['Channel URL'].nunique()} channels")
        return self.chunks
    
    def get_channel_chunks(self, channel_df: pd.DataFrame, channel_url: str, start_chunk_id: int) -> List[Chunk]:
        """Generate chunks for a single channel using half-open intervals"""
        channel_chunks = []
        total_messages = len(channel_df)
        
        if total_messages == 0:
            return channel_chunks
        
        # Reset index to ensure continuous indexing within channel
        channel_df = channel_df.reset_index(drop=True)
        
        chunk_id = start_chunk_id
        i = 0
        
        while True:
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
            
            # Break if we've reached the end
            if start_idx >= total_messages:
                break
            
            # Create chunk with DataFrame slice
            chunk_messages = channel_df.iloc[start_idx:end_idx].copy()
            
            chunk = Chunk(
                chunk_id=chunk_id,
                channel_url=channel_url,
                start_idx=start_idx,
                end_idx=end_idx,
                messages=chunk_messages,
                total_messages=len(chunk_messages),
                has_overlap_with_previous=has_overlap_with_previous,
                overlap_size=overlap_size
            )
            
            print(f"Generated chunk {chunk_id}: [{start_idx}, {end_idx}), "
                  f"{len(chunk_messages)} messages, channel: {channel_url[:30]}...")
            channel_chunks.append(chunk)
            chunk_id += 1
            i += 1
            
            # Break if this chunk reaches the end
            if end_idx >= total_messages:
                break
        
        return channel_chunks


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
        default='claude-sonnet-4-20250514',
        help='LLM model to use for tail summary generation (default: claude-sonnet-4-20250514)'
    )
    parser.add_argument(
        '--api-key',
        help='API key for LLM provider (default: from ANTHROPIC_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    try:
        # Stage 1: File Processing
        processor = FileProcessor(args.input, args.output_dir)
        df_clean = processor.process()
        
        if df_clean is None:
            print("Error: File processing failed")
            exit(1)
        
        # Stage 2: Channel Segmentation
        segmenter = ChannelSegmenter(df_clean, args.chunk_size, args.overlap)
        chunks = segmenter.generate_chunks()
        
        print(f"Generated {len(chunks)} chunks with chunk_size={args.chunk_size}, overlap={args.overlap}")
        
        # Stage 3: Initialize LLM Client for case segmentation
        llm_client = LLMClient(model=args.model, api_key=args.api_key)
        print(f"LLM Client initialized with model: {args.model}")
        
        # Stage 4: Case Segmentation on First Chunk
        if chunks:
            print(f"\n--- Stage 4: Case Segmentation on First Chunk ---")
            first_chunk = chunks[0]
            print(f"Processing chunk {first_chunk.chunk_id} with {first_chunk.total_messages} messages...")
            
            # Format messages for case segmentation
            current_messages = first_chunk.format_for_prompt()
            
            # Generate case segments (no previous context for first chunk)
            print("Generating case segments using LLM...")
            case_results = first_chunk.generate_case_segments(
                current_messages=current_messages,
                previous_tail_summary=None,
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
            output_file = os.path.join(args.output_dir, f"first_chunk_cases.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(case_results, f, indent=2, ensure_ascii=False)
            print(f"Case results saved to: {output_file}")
        else:
            print("No chunks available for case segmentation")
        
        print(f"\n✅ Pipeline complete!")
        
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == '__main__':
    main()