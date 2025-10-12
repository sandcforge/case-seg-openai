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
    def get_aloy_token() -> str:
        """
        Get Firebase ID token for Aloy authentication using environment variables
        
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

        # Call API
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Parse the XML-like tags from data.result
            # Expected format: { "status": "success", "data": { "result": "<SOP Content>...</SOP Content>..." } }
            result = {}
            if data.get('status') == 'success' and data.get('data', {}).get('result'):
                result_str = data['data']['result']

                # Extract content between tags
                import re

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

        except requests.RequestException as e:
            raise RuntimeError(f"SOP API request failed: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Failed to parse SOP API response: {e}")