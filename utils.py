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
        Format channel URL for display: show channel_ + last 5 characters
        Example: sendbird_group_channel_215482988_b374305ff3e440674e786d63916f1d5aacda8249 -> channel_da8249
        """
        if len(channel_url) <= 5:
            return channel_url
        return f"channel_{channel_url[-5:]}"