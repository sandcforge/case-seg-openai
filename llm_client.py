#!/usr/bin/env python3
"""
LLM client module for handling interactions with language models.

This module provides a unified interface for working with both OpenAI and Anthropic models,
including structured output generation and prompt template loading.
"""

import os
import openai # type: ignore
import anthropic # type: ignore
from datetime import datetime


class LLMClient:
    """LLM client supporting both Claude (Anthropic) and OpenAI models"""
    
    def __init__(self, model: str = "gpt-5-2025-08-01"):
        self.model = model
        self.provider, env_key_name = self._get_provider_and_key(model)
        
        # Load API key from environment
        self.api_key = os.getenv(env_key_name)
        if not self.api_key:
            raise ValueError(f"{env_key_name} not found. Please set it in .env file")
        
        # Initialize appropriate client
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_provider_and_key(self, model: str) -> tuple[str, str]:
        """Determine provider and environment key name based on model prefix"""
        model_lower = model.lower()
        
        if model_lower.startswith('claude-'):
            return "anthropic", "ANTHROPIC_API_KEY"
        elif model_lower.startswith('gpt-') or 'gpt' in model_lower:
            return "openai", "OPENAI_API_KEY"
        elif model_lower.startswith('gemini-'):
            return "google", "GOOGLE_API_KEY"  # Future support
        else:
            raise ValueError(f"Cannot determine provider from model name: {model}. Supported prefixes: claude-, gpt-, gemini-")
    
    def _get_indent_from_call_label(self, call_label: str) -> str:
        """Determine indentation level based on call_label content"""
        if "case_classification_" in call_label:
            return "                        "  # 24 spaces (case level)
        elif "chunk_" in call_label:
            return "                "  # 16 spaces (chunk level)
        elif "channel_" in call_label:
            return "        "  # 8 spaces (channel level)
        else:
            return ""  # No indentation
    
    def generate(self, prompt: str, call_label: str = "unknown", max_tokens: int = 12000) -> str:
        """Generate response using appropriate API with debug logging"""
        import time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()
        
        # Create debug output directory if it doesn't exist
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Generate debug log filename
        debug_file = os.path.join(debug_dir, f"{call_label}_{timestamp}.log")
        
        try:
            # Log the request
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== LLM CALL DEBUG LOG ===\n")
                f.write(f"Start Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Call Label: {call_label}\n")
                f.write(f"Model: {self.model} ({self.provider.title()})\n")
                f.write(f"Max Tokens: {max_tokens}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write("\n=== PROMPT ===\n")
                f.write(prompt)
                f.write("\n\n")
            
            # Make the API call
            if self.provider == "openai":
                response = self._call_openai(prompt, max_tokens)
            else:
                response = self._call_anthropic(prompt, max_tokens)
            
            end_time = time.time()
            duration_seconds = end_time - start_time
            
            # Log the successful response
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=== RESPONSE ===\n")
                f.write(response)
                f.write(f"\n\nResponse Length: {len(response)} characters\n")
                f.write(f"End Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"LLM Call Duration: {duration_seconds:.2f} seconds\n")
                f.write("\n=== STATUS ===\n")
                f.write("Success: LLM call completed successfully\n")
            
            print(f"{self._get_indent_from_call_label(call_label)}Debug log saved: {debug_file}")
            return response
            
        except Exception as e:
            end_time = time.time()
            duration_seconds = end_time - start_time
            
            # Log the error
            try:
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write("=== ERROR ===\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Error Type: {type(e).__name__}\n")
                    f.write(f"End Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"LLM Call Duration: {duration_seconds:.2f} seconds\n")
                    f.write("\n=== STATUS ===\n")
                    f.write(f"Failed: LLM call failed - {str(e)}\n")
            except:
                pass  # If debug logging fails, don't break the main functionality
            
            raise RuntimeError(f"LLM generation failed ({call_label}): {e}")
    
    def generate_structured(self, prompt: str, response_format, call_label: str = "unknown", max_tokens: int = 1200):
        """Generate structured response using OpenAI with JSON schema"""
        if self.provider != "openai":
            raise RuntimeError("Structured output is only supported for OpenAI models")
        
        import time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()
        
        # Create debug output directory if it doesn't exist
        debug_dir = "debug_output"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Generate debug log filename
        debug_file = os.path.join(debug_dir, f"{call_label}_{timestamp}.log")
        
        try:
            # Log the request
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== LLM STRUCTURED CALL DEBUG LOG ===\n")
                f.write(f"Start Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Call Label: {call_label}\n")
                f.write(f"Model: {self.model} (OpenAI Structured)\n")
                f.write(f"Max Tokens: {max_tokens}\n")
                f.write(f"Response Format: {response_format.__name__}\n")
                f.write(f"Prompt Length: {len(prompt)} characters\n")
                f.write("\n=== PROMPT ===\n")
                f.write(prompt)
                f.write("\n\n")
            
            # Make the structured API call
            response = self.client.responses.parse(
                model=self.model,
                input=[{"role": "user", "content": prompt}],
                text_format=response_format,
            )

            end_time = time.time()
            duration_seconds = end_time - start_time
            parsed_response = response.output_parsed
            
            # Log the successful response
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=== RESPONSE (RAW JSON) ===\n")
                f.write(parsed_response.model_dump_json(indent=2))
                f.write(f"\n\nResponse Length: {len(parsed_response.model_dump_json(indent=2))} characters\n")
                f.write(f"End Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"LLM Call Duration: {duration_seconds:.2f} seconds\n")
                f.write("\n=== STATUS ===\n")
                f.write("Success: Structured LLM call completed successfully\n")
            
            print(f"{self._get_indent_from_call_label(call_label)}Debug log saved: {debug_file}")
            return parsed_response
            
        except Exception as e:
            end_time = time.time()
            duration_seconds = end_time - start_time
            
            # Log the error
            try:
                with open(debug_file, 'a', encoding='utf-8') as f:
                    f.write("=== ERROR ===\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Error Type: {type(e).__name__}\n")
                    f.write(f"End Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"LLM Call Duration: {duration_seconds:.2f} seconds\n")
                    f.write("\n=== STATUS ===\n")
                    f.write(f"Failed: Structured LLM call failed - {str(e)}\n")
            except:
                pass  # If debug logging fails, don't break the main functionality
            
            raise RuntimeError(f"Structured LLM generation failed ({call_label}): {e}")
    
    def _call_openai(self, prompt: str, max_tokens: int) -> str:
        """Call OpenAI API"""
        # GPT-5 models use max_completion_tokens instead of max_tokens
        if "gpt-5" in self.model.lower():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, max_tokens: int) -> str:
        """Call Anthropic API"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    
    def load_prompt(self, filename: str) -> str:
        """Load prompt template from prompts directory"""
        prompt_path = os.path.join("prompts", filename)
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")