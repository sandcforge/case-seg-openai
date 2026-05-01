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
    def generate_short_case_id() -> str:
        """
        生成短 UUID 作为 case_id

        使用 Base64 URL-safe 编码的 UUID4，长度为 22 字符

        Returns:
            22 字符的唯一 ID，例如：'VQ6EAOKbQdSnFkRmVUQAAA'

        Example:
            >>> case_id = Utils.generate_short_case_id()
            >>> len(case_id)
            22
            >>> # 每次调用生成不同的 ID
            >>> id1 = Utils.generate_short_case_id()
            >>> id2 = Utils.generate_short_case_id()
            >>> id1 != id2
            True
        """
        import uuid
        import base64

        uuid_bytes = uuid.uuid4().bytes
        return base64.urlsafe_b64encode(uuid_bytes).rstrip(b'=').decode('ascii')

    @staticmethod
    def format_messages_for_prompt2(chunk_df) -> str:
        """
        Format DataFrame messages in table layout with tab-separated columns
        Columns: Message ID (22), Created Time (19), Role (6), Type (4), Message (100 chars wrap)

        Args:
            chunk_df: DataFrame with message data

        Returns:
            Formatted table string with headers and wrapped messages
        """
        import pandas as pd

        lines = []
        # Header - use tabs between columns
        lines.append("-" * 150)
        lines.append(f"{'Message ID':<22}\t{'Created Time (UTC)':<19}\t{'Role':<6}\t{'Type':<4}\tMessage/File Summary")
        lines.append("-" * 150)

        # Process each message
        for _, row in chunk_df.iterrows():
            message_id = str(row.get('Message ID', ''))[:21]  # nanoid is 21 chars
            created_time = str(row.get('Created Time', ''))[:19]  # Keep full timestamp
            role = str(row.get('role', ''))

            # Map role names
            if role == 'customer_service':
                role = 'cs'
            elif role == 'user':
                role = 'user'
            role = role[:6]  # Truncate to 6 chars

            # Get Type - should be MESG or FILE (4 chars)
            msg_type = str(row.get('Type', ''))
            if msg_type.upper() == 'MESSAGE':
                msg_type = 'MESG'
            elif msg_type.upper() == 'FILE':
                msg_type = 'FILE'
            msg_type = msg_type[:4]  # Ensure max 4 chars

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

            # Format with wrapping - use tabs
            # Prefix: msg_id(22) + tab + time(19) + tab + role(6) + tab + type(4) + tab
            msg_prefix = f"{message_id:<22}\t{created_time:<19}\t{role:<6}\t{msg_type:<4}\t"
            # Continuation indent: align to start of message column (after all tabs)
            msg_indent = " " * 22 + "\t" + " " * 19 + "\t" + " " * 6 + "\t" + " " * 4 + "\t"

            # Message wraps at 100 chars
            if len(message) <= 100:
                lines.append(f"{msg_prefix}{message}")
            else:
                # Wrap message across multiple lines
                remaining = message
                first_line = True
                while remaining:
                    if first_line:
                        available = 100
                        line_start = msg_prefix
                        first_line = False
                    else:
                        available = 100  # Continuation lines also get 100 chars
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
        lines.append("-" * 150)

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

    @staticmethod
    def query_bigquery(
        sql: str,
        query_params: 'Optional[List[Dict[str, Any]]]' = None
    ) -> 'List[Dict[str, Any]]':
        """
        Query Google BigQuery and return results as JSON format

        Reads BigQuery service account credentials from environment variable BIGQUERY_CREDENTIALS_JSON.
        The credentials should be stored as a single-line JSON string in .env file.

        Args:
            sql: SQL query string. Use @param_name for parameterized queries
            query_params: Optional list of query parameters for parameterized queries
                Format: [{"name": "param_name", "type": "STRING|INT64|FLOAT64|BOOL|TIMESTAMP", "value": value}, ...]
                Example: [{"name": "channel", "type": "STRING", "value": "sendbird_xxx"}]

        Returns:
            List of dictionaries, each representing a row from the query results
            Example: [{"column1": value1, "column2": value2}, ...]

        Raises:
            ValueError: If BIGQUERY_CREDENTIALS_JSON environment variable is missing or invalid JSON
            RuntimeError: If BigQuery API request fails

        Example:
            # Simple query
            results = Utils.query_bigquery(
                sql="SELECT * FROM `plantstory.dataset.table` LIMIT 10"
            )

            # Parameterized query (recommended for security)
            results = Utils.query_bigquery(
                sql="SELECT * FROM `plantstory.support.messages` WHERE channel_url = @channel",
                query_params=[{"name": "channel", "type": "STRING", "value": "sendbird_xxx"}]
            )
        """
        import json
        from typing import List, Dict, Any, Optional

        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
        except ImportError:
            raise RuntimeError(
                "BigQuery client library not installed. "
                "Please run: pip install google-cloud-bigquery>=3.11.0"
            )

        # Load environment variables
        load_dotenv()

        # Get BigQuery credentials JSON from environment variable
        credentials_json = os.getenv('BIGQUERY_CREDENTIALS_JSON')

        if not credentials_json:
            raise ValueError(
                "Missing required environment variable: BIGQUERY_CREDENTIALS_JSON\n"
                "Please add your BigQuery service account JSON credentials to .env file.\n"
                "Example: BIGQUERY_CREDENTIALS_JSON='{\"type\":\"service_account\",...}'"
            )

        try:
            # Parse JSON string to dictionary
            credentials_dict = json.loads(credentials_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in BIGQUERY_CREDENTIALS_JSON: {e}")

        try:
            # Create credentials from service account info (not from file)
            credentials = service_account.Credentials.from_service_account_info(
                credentials_dict
            )

            # Create BigQuery client with plantstory project
            client = bigquery.Client(
                credentials=credentials,
                project='plantstory'
            )

            # Configure query job with parameters if provided
            job_config = None
            if query_params:
                # Convert query parameters to BigQuery format
                bq_params = []
                for param in query_params:
                    param_name = param.get('name')
                    param_type = param.get('type', 'STRING')
                    param_value = param.get('value')

                    if not param_name:
                        raise ValueError("Query parameter missing 'name' field")

                    # Check if parameter is an array type
                    if param_type.startswith('ARRAY'):
                        # Extract element type from ARRAY<TYPE>
                        element_type = param_type.replace('ARRAY<', '').replace('>', '')
                        bq_params.append(
                            bigquery.ArrayQueryParameter(param_name, element_type, param_value)
                        )
                    else:
                        bq_params.append(
                            bigquery.ScalarQueryParameter(param_name, param_type, param_value)
                        )

                job_config = bigquery.QueryJobConfig(query_parameters=bq_params)

            # Execute query
            query_job = client.query(sql, job_config=job_config)

            # Wait for query to complete and get results
            results = query_job.result()

            # Convert results to list of dictionaries (JSON format)
            rows_list = []
            for row in results:
                # Convert Row object to dictionary
                row_dict = dict(row.items())
                rows_list.append(row_dict)

            return rows_list

        except Exception as e:
            raise RuntimeError(f"BigQuery query failed: {e}")

    @staticmethod
    def get_channels_to_process(
        chunk_size: int = 80,
        idle_days: int = 7,
        channel_urls: 'Optional[List[str]]' = None
    ) -> 'Any':
        """
        获取需要处理的所有未分析消息（SQL 层面过滤）

        触发条件和处理策略（针对每个 channel）：
        1. 未分析消息数 < chunk_size:
           - 如果最后一条消息距今 >= idle_days 天 → 处理所有消息
           - 否则 → 不处理
        2. 未分析消息数 >= chunk_size:
           - 只处理 chunk_size 的整数倍数量
           - 余数保留，等待下次处理
           - 例如：90条 → 处理80条，保留10条；185条 → 处理160条，保留25条

        Args:
            chunk_size: 触发分析的消息数量阈值（默认 80）
            idle_days: 触发分析的空闲天数阈值（默认 7）
            channel_urls: 可选的 channel URL 列表，如果提供则只检查这些 channels

        Returns:
            DataFrame 包含 support_message 的所有列，只包含应该处理的消息
            按 channel_url, created_time, message_id 排序
        """
        # 构建 SQL 和参数
        if channel_urls:
            channel_filter = "AND sm.channel_url IN UNNEST(@channel_urls)"
            query_params = [
                {"name": "chunk_size", "type": "INT64", "value": chunk_size},
                {"name": "idle_days", "type": "INT64", "value": idle_days},
                {"name": "channel_urls", "type": "ARRAY<STRING>", "value": channel_urls}
            ]
        else:
            channel_filter = ""
            query_params = [
                {"name": "chunk_size", "type": "INT64", "value": chunk_size},
                {"name": "idle_days", "type": "INT64", "value": idle_days}
            ]

        sql = f"""
        WITH channel_stats AS (
            SELECT
                sm.channel_url,
                COUNT(*) as unanalyzed_count,
                MAX(sm.created_time) as last_message_time,
                -- 计算应处理的消息数量
                CASE
                    -- 规则1: 小于 chunk_size 且 idle_days 触发 → 全部处理
                    WHEN COUNT(*) < @chunk_size
                         AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(sm.created_time), DAY) >= @idle_days
                    THEN COUNT(*)
                    -- 规则2: >= chunk_size → 处理整数倍，余数保留
                    WHEN COUNT(*) >= @chunk_size
                    THEN CAST(FLOOR(COUNT(*) / @chunk_size) * @chunk_size AS INT64)
                    -- 其他情况：不处理
                    ELSE 0
                END as messages_to_process
            FROM `plantstory.public.support_message` sm
            WHERE sm.deleted = FALSE
              AND sm.created_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
              {channel_filter}
              AND NOT EXISTS (
                SELECT 1
                FROM `plantstory.customer_service.support_message_cases` seg,
                UNNEST(seg.msg_id_list) AS seg_msg_id
                WHERE seg.channel_url = sm.channel_url
                  AND seg_msg_id = sm.id
              )
            GROUP BY sm.channel_url
            HAVING messages_to_process > 0  -- 只返回有消息需要处理的 channel
        ),
        ranked_messages AS (
            SELECT
                sm.*,
                cs.messages_to_process,
                ROW_NUMBER() OVER (
                    PARTITION BY sm.channel_url
                    ORDER BY sm.created_time
                ) as row_num
            FROM `plantstory.public.support_message` sm
            INNER JOIN channel_stats cs ON sm.channel_url = cs.channel_url
            WHERE sm.deleted = FALSE
              AND sm.created_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
              AND NOT EXISTS (
                SELECT 1
                FROM `plantstory.customer_service.support_message_cases` seg,
                UNNEST(seg.msg_id_list) AS seg_msg_id
                WHERE seg.channel_url = sm.channel_url
                  AND seg_msg_id = sm.id
              )
        )
        SELECT * EXCEPT(messages_to_process, row_num)
        FROM ranked_messages
        WHERE row_num <= messages_to_process
        ORDER BY channel_url, created_time
        """

        import pandas as pd
        results = Utils.query_bigquery(sql, query_params)
        return pd.DataFrame(results)

    @staticmethod
    def preprocess_dataframe(df: 'pd.DataFrame', verbose: bool = True) -> 'pd.DataFrame':
        """
        预处理 DataFrame，准备用于消息分段处理

        此方法从 Session.process_file_data() 提取，用于复用预处理逻辑

        处理步骤：
        0. 自动检测并转换列名（snake_case → Title Case）
        1. 过滤 Deleted = True 的行
        2. 添加 role 列（基于 Sender ID 模式）
        3. 排序（Channel URL, Created Time, Message ID）
        4. 添加 File Summary 列（用于存储 vision 分析结果）
        5. 生成包含必需列的 clean DataFrame

        Args:
            df: 原始 DataFrame（支持两种列名格式：snake_case 或 Title Case）
            verbose: 是否打印处理日志

        Returns:
            处理后的 clean DataFrame，包含必需列
        """
        import pandas as pd

        if verbose:
            print("Starting DataFrame preprocessing...")

        # 0. Auto-detect and convert column names if needed
        # Check if DataFrame uses snake_case (BigQuery format) by looking for key column
        if 'id' in df.columns:
            if verbose:
                print("        Detected snake_case column names (BigQuery format), converting to Title Case...")

            # Define column mapping from snake_case to Title Case
            column_mapping = {
                'id': 'Message ID',
                'type': 'Type',
                'message': 'Message',
                'raw': 'Raw',
                'sender_id': 'Sender ID',
                'real_sender_id': 'Real Sender ID',
                'created_time': 'Created Time',
                'updated_time': 'Updated Time',
                'channel_url': 'Channel URL',
                'file_content_size': 'File Content Size',
                'file_content_type': 'File Content Type',
                'file_url': 'File URL',
                'filename': 'Filename',
                'sender_type': 'Sender Type',
                'datastream_metadata': 'Datastream Metadata',
                'deleted': 'Deleted',
                'ticket_id': 'Ticket ID'
            }

            # Count columns to rename before renaming
            columns_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}

            # Rename columns that exist in the DataFrame
            df = df.rename(columns=columns_to_rename)

            if verbose:
                print(f"        Converted {len(columns_to_rename)} column names to Title Case")

            if verbose:
                if 'Message ID' in df.columns:
                    print(f"        Message ID column mapped (string type, kept as-is)")

        elif 'Message ID' in df.columns:
            if verbose:
                print("        Detected Title Case column names (CSV format), no conversion needed")
        else:
            if verbose:
                print("        ⚠️  Warning: Could not detect column format (neither 'id' nor 'Message ID' found)")

        # 0. Filter out rows where Message ID is NaN/empty
        if 'Message ID' in df.columns:
            original_count = len(df)
            df['Message ID'] = df['Message ID'].astype(str)
            df = df[df['Message ID'].notna() & (df['Message ID'] != '') & (df['Message ID'] != 'nan') & (df['Message ID'] != 'None')].reset_index(drop=True)
            filtered_count = original_count - len(df)
            if filtered_count > 0 and verbose:
                print(f"        Filtered out {filtered_count} rows with invalid Message ID ({len(df)} remaining)")

        # 1. Filter out rows where Deleted = True
        if 'Deleted' in df.columns:
            original_count = len(df)
            df = df[df['Deleted'] != True].reset_index(drop=True)
            filtered_count = original_count - len(df)
            if verbose:
                print(f"        Filtered out {filtered_count} deleted rows ({len(df)} remaining)")
        else:
            if verbose:
                print("        No 'Deleted' column found, skipping deletion filter")

        # 2. Add role column based on Sender ID pattern
        if 'role' not in df.columns:
            df['role'] = df['Sender ID'].apply(
                lambda x: 'customer_service' if str(x).startswith('psops') else 'user'
            )
            if verbose:
                print(f"        Added role column: {df['role'].value_counts().to_dict()}")
        else:
            if verbose:
                print("        Role column already exists, skipping...")

        # 3. Sort data by Channel URL, Created Time
        # Note: Created Time is kept as ISO 8601 string format for correct lexicographic sorting
        df = df.sort_values([
            'Channel URL',
            'Created Time'
        ]).reset_index(drop=True)
        if verbose:
            print(f"        Sorted data by Channel URL and Created Time")

        # 4. Add File Summary column for vision analysis results
        if 'File Summary' not in df.columns:
            df['File Summary'] = ''
            if verbose:
                print(f"        Added File Summary column for storing vision analysis results")

        # 5. Generate clean DataFrame with essential columns
        essential_columns = [
            'Created Time', 'Sender ID', 'Message', 'Channel URL',
            'role', 'Message ID', 'Type', 'File URL', 'File Summary'
        ]
        available_columns = [col for col in essential_columns if col in df.columns]
        df_clean = df[available_columns].copy()
        if verbose:
            print(f"        Created clean DataFrame with {len(available_columns)} columns: {available_columns}")

        # 6. Convert Timestamp columns to ISO format strings
        timestamp_columns = ['Created Time', 'Updated Time']
        for col in timestamp_columns:
            if col in df_clean.columns:
                # Check if column contains Timestamp objects
                if pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                    df_clean[col] = pd.to_datetime(df_clean[col]).apply(lambda x: x.isoformat() if pd.notna(x) else None)
                    if verbose:
                        print(f"        Converted {col} to ISO format strings")
                elif df_clean[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    df_clean[col] = df_clean[col].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)
                    if verbose:
                        print(f"        Converted {col} to ISO format strings")

        if verbose:
            print(f"        Processed {len(df_clean)} messages across {df_clean['Channel URL'].nunique()} channels")

        # 7. Display channel summary
        if verbose:
            for channel_url in df_clean['Channel URL'].unique():
                channel_df = df_clean[df_clean['Channel URL'] == channel_url]
                print(f"                Channel: {Utils.format_channel_for_display(channel_url)} - {len(channel_df)} messages")

        return df_clean