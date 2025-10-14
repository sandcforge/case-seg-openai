#!/usr/bin/env python3
"""
Utility functions and classes for the customer support message segmentation system.

This module contains common utility functions that are used across different modules.
"""

import requests # type: ignore
import os
from dotenv import load_dotenv # type: ignore


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
    
    # FIXME: review all the references, if need to be replaced by format_messages_for_prompt2, which consider FILE type
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

    @staticmethod
    def format_messages_for_prompt2(chunk_df) -> str:
        """
        Format DataFrame messages in table layout with 100-char width and line wrapping
        Columns: Created Time (20), Role (6), Type (6), Message/File Summary

        Args:
            chunk_df: DataFrame with message data

        Returns:
            Formatted table string with headers and wrapped messages
        """
        import pandas as pd

        lines = []
        # Header
        lines.append("-" * 100)
        lines.append(f"{'Created Time':<20} {'Role':<6} {'Type':<6} {'Message/File Summary':<64}")
        lines.append("-" * 100)

        # Process each message
        for _, row in chunk_df.iterrows():
            created_time = str(row.get('Created Time', ''))[:19]
            role = str(row.get('role', ''))

            # Replace customer_service with CS
            if role == 'customer_service':
                role = 'cs'
            role = role[:5]

            msg_type = str(row.get('Type', ''))[:5]

            # Get message content
            message = row.get('Message', '')

            # For FILE type, use file_summary if available
            if msg_type == 'FILE':
                file_summary = row.get('File Summary', '')
                if file_summary and not pd.isna(file_summary) and str(file_summary).lower() != 'nan':
                    message = file_summary

            # Handle NaN and empty messages
            if pd.isna(message):
                message = ''
            else:
                message = str(message).replace('\n', ' ').replace('\r', ' ')

            # Format with wrapping
            # First line uses: time(20) + space + role(6) + space + type(6) + space = 34 chars
            # Remaining for message: 100 - 34 = 66 chars on first line
            msg_prefix = f"{created_time:<20} {role:<6} {msg_type:<6} "
            msg_indent = " " * 34  # Align continuation lines

            if len(message) <= 66:
                lines.append(f"{msg_prefix}{message}")
            else:
                # Wrap message across multiple lines
                remaining = message
                first_line = True
                while remaining:
                    if first_line:
                        available = 66
                        line_start = msg_prefix
                        first_line = False
                    else:
                        available = 66  # Continuation lines also get 66 chars
                        line_start = msg_indent

                    if len(remaining) <= available:
                        lines.append(f"{line_start}{remaining}")
                        break

                    # Find last space within available width
                    split_pos = remaining[:available].rfind(' ')
                    if split_pos == -1:
                        split_pos = available

                    lines.append(f"{line_start}{remaining[:split_pos]}")
                    remaining = remaining[split_pos:].lstrip()

        # Footer
        lines.append("-" * 100)

        return '\n'.join(lines)

    @staticmethod
    def get_aloy_token() -> str:
        """
        Get Firebase ID token for Aloy authentication using environment variables
        Token is cached for 3 hours to avoid unnecessary API calls

        Required environment variables:
        - FIREBASE_API_KEY: Firebase project API key
        - FIREBASE_EMAIL: User email for authentication
        - FIREBASE_PASSWORD: User password for authentication

        Returns:
            Firebase ID token string

        Raises:
            ValueError: If required environment variables are missing
            RuntimeError: If authentication request fails
        """
        from datetime import datetime, timedelta

        # Check if we have a cached token that's still valid (less than 3 hours old)
        if hasattr(Utils.get_aloy_token, '_cache'):
            cache = Utils.get_aloy_token._cache
            if cache['token'] and cache['timestamp']:
                time_elapsed = datetime.now() - cache['timestamp']
                if time_elapsed < timedelta(hours=3):
                    return cache['token']

        # Load environment variables
        load_dotenv()

        # Get required environment variables
        api_key = os.getenv("FIREBASE_API_KEY")
        email = os.getenv("FIREBASE_EMAIL")
        password = os.getenv("FIREBASE_PASSWORD")

        # Check for missing environment variables
        missing_vars = []
        if not api_key:
            missing_vars.append("FIREBASE_API_KEY")
        if not email:
            missing_vars.append("FIREBASE_EMAIL")
        if not password:
            missing_vars.append("FIREBASE_PASSWORD")

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Prepare Firebase authentication request
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }

        try:
            # Make authentication request
            response = requests.post(url, json=payload)
            response.raise_for_status()

            # Parse response
            data = response.json()
            id_token = data.get("idToken")

            if not id_token:
                raise RuntimeError("Firebase authentication succeeded but no idToken was returned")

            # Cache the token with current timestamp
            Utils.get_aloy_token._cache = {
                'token': id_token,
                'timestamp': datetime.now()
            }

            return id_token

        except requests.RequestException as e:
            raise RuntimeError(f"Firebase authentication request failed: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse Firebase authentication response: {e}")

    @staticmethod
    def call_sop_api(chat_logs: str) -> dict:
        """
        Call Aloy API to find the most relevant SOP for given chat logs

        Args:
            chat_logs: Formatted chat logs string

        Returns:
            Parsed dict containing: { "sop_content": "...", "sop_url": "...", "sop_score": "..." }
            Returns empty strings if parsing fails

        Raises:
            RuntimeError: If API call fails
        """
        # Get Firebase token
        try:
            token = Utils.get_aloy_token()
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(f"Failed to get authentication token: {e}")

        # Prepare API request
        url = "https://api.plantstory.app/aloy/ai/generate-text"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "systemPromptName": "sheng-find-most-sop",
            "frontendVariables": {
                "chatLogs": chat_logs
            },
            "showDetails": False,
            "messages": [
                {
                    "parts": [
                        {
                            "type": "text",
                            "text": "\n"
                        }
                    ],
                    "role": "user"
                }
            ]
        }

        # Call API with timeout and retry logic
        import re
        max_retries = 2
        timeout_seconds = 300  # 5 minutes

        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
                response.raise_for_status()
                data = response.json()

                # Parse the XML-like tags from data.result
                # Expected format: { "status": "success", "data": { "result": "<SOP Content>...</SOP Content>..." } }
                result = {}
                if data.get('status') == 'success' and data.get('data', {}).get('result'):
                    result_str = data['data']['result']

                    # Extract content between tags
                    # Extract <SOP Content>...</SOP Content>
                    sop_content_match = re.search(r'<SOP Content>(.*?)</SOP Content>', result_str, re.DOTALL)
                    result['sop_content'] = sop_content_match.group(1).strip() if sop_content_match else 'N/A'

                    # Extract <SOP URL>...</SOP URL>
                    sop_url_match = re.search(r'<SOP URL>(.*?)</SOP URL>', result_str, re.DOTALL)
                    result['sop_url'] = sop_url_match.group(1).strip() if sop_url_match else 'N/A'

                    # Extract <SOP Score>...</SOP Score>
                    sop_score_match = re.search(r'<SOP Score>(.*?)</SOP Score>', result_str, re.DOTALL)
                    result['sop_score'] = sop_score_match.group(1).strip() if sop_score_match else 'N/A'
                else:
                    result = {
                        'sop_content': 'N/A',
                        'sop_url': 'N/A',
                        'sop_score': 'N/A'
                    }

                return result

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"        ⏱️  SOP API timeout, retrying... (attempt {attempt + 2}/{max_retries})")
                    continue
                else:
                    raise RuntimeError(f"SOP API request timed out after {max_retries} attempts ({timeout_seconds}s each)")

            except requests.RequestException as e:
                raise RuntimeError(f"SOP API request failed: {e}")
            except (KeyError, ValueError) as e:
                raise RuntimeError(f"Failed to parse SOP API response: {e}")