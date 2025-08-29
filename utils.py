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
    def format_one_msg_for_prompt(row) -> str:
        """
        Format single message row for LLM prompt: message_index | sender id | role | timestamp | text
        Used for case classification in case.py
        """
        message_text = str(row['Message']).replace('\n', ' ').replace('\r', ' ')
        if message_text == 'nan':
            message_text = ''
        return f"{row['msg_ch_idx']} | {row['Sender ID']} | {row['role']} | {row['Created Time']} | {message_text}"
