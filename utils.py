#!/usr/bin/env python3
"""
Utility functions and classes for the customer support message segmentation system.

This module contains common utility functions that are used across different modules.
"""


class Utils:
    """Utility class containing static helper methods"""
    
    @staticmethod
    def format_channel_for_display(channel_url: str) -> str:
        """
        Format channel URL for display: extract hash part after last underscore
        Example: sendbird_group_channel_215482988_b374305ff3e440674e786d63916f1d5aacda8249 -> b374305ff3e440674e786d63916f1d5aacda8249
        """
        if '_' not in channel_url:
            return channel_url
        return channel_url.split('_')[-1]
    
    @staticmethod
    def format_messages_for_prompt(chunk_df) -> str:
        """
        Format DataFrame messages for LLM prompt: message_index | sender id | role | timestamp | text
        Supports both single rows and DataFrames via pandas iterrows() method
        """
        formatted_lines = []
        for _, row in chunk_df.iterrows():
            # Handle NaN messages and replace newlines with spaces to keep one line per message
            message_text = str(row['Message']).replace('\n', ' ').replace('\r', ' ')
            if message_text == 'nan':
                message_text = ''
            
            formatted_line = f"{row['msg_ch_idx']} | {row['Sender ID']} | {row['role']} | {row['Created Time']} | {message_text}"
            formatted_lines.append(formatted_line)
        return '\n'.join(formatted_lines)