#!/usr/bin/env python3
"""
Vision processing module for analyzing images in customer support contexts.

This module provides the VisionProcessor class that analyzes images with context
from surrounding messages to extract detailed descriptions and metadata for
customer service representatives.
"""

import pandas as pd  # type: ignore
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from utils import Utils

if TYPE_CHECKING:
    from llm_client import LLMClient, VisionResponse
else:
    from llm_client import VisionResponse


class VisionProcessor:
    """
    Utility class for processing images with context to generate detailed descriptions and extract metadata.
    
    This class provides static methods for taking DataFrame context and image URLs to produce comprehensive
    descriptions that allow customer service staff to understand image content
    without viewing the actual image.
    """
        
    @staticmethod
    def get_context_for_image(channel_df: pd.DataFrame, 
                             image_msg_ch_idx: int,
                             context_size: int = 5) -> pd.DataFrame:
        """
        Get context messages around an image message (MESG type only)
        
        Args:
            channel_df: DataFrame containing messages from a single channel
            image_msg_ch_idx: The msg_ch_idx of the image message within this channel
            context_size: Number of MESG messages to take before and after (default 5)
            
        Returns:
            DataFrame containing context messages (MESG type + original image message), sorted by msg_ch_idx
        """
        # Find the image message in this channel
        image_msg = channel_df[channel_df['msg_ch_idx'] == image_msg_ch_idx]
        if len(image_msg) == 0:
            raise ValueError(f"Image message with msg_ch_idx {image_msg_ch_idx} not found in channel")
        
        # Filter to MESG type messages only for context
        mesg_df = channel_df[channel_df['Type'] == 'MESG'].copy()
        
        # Find MESG messages before and after the image message
        before_msgs = mesg_df[mesg_df['msg_ch_idx'] < image_msg_ch_idx].tail(context_size)
        after_msgs = mesg_df[mesg_df['msg_ch_idx'] > image_msg_ch_idx].head(context_size)
        
        # Combine context messages with image message
        context_msgs = pd.concat([before_msgs, image_msg, after_msgs], ignore_index=True)
        
        # Sort by msg_ch_idx to ensure correct chronological order
        context_msgs = context_msgs.sort_values('msg_ch_idx').reset_index(drop=True)
        
        return context_msgs
        
    @classmethod
    def analyze_image_with_context(cls, 
                                 context_df: pd.DataFrame, 
                                 image_url: str,
                                 llm_client: 'LLMClient') -> Dict[str, Any]:
        """
        Analyze image with DataFrame context and return structured JSON description.
        
        Args:
            context_df: pandas DataFrame slice from df_clean containing context messages
                       Columns: 'msg_ch_idx', 'Sender ID', 'role', 'Created Time', 'Message', 'Type', 'File URL'
            image_url: URL to the image file for analysis
            
        Returns:
            Dictionary containing visual analysis with detailed description and metadata
            
        Example:
            {
                "visual_analysis": {
                    "description": "Image shows a damaged plant package...",
                    "customer_intent": "Customer is reporting shipping damage",
                    "meta_info": {
                        "tracking_ids": ["1Z999AA1234567890"],
                        "has_damage": True,
                        "damage_type": ["packaging_damage", "plant_damage"],
                        ...
                    },
                    "confidence": 0.85
                }
            }
        """
        try:
            # Format context DataFrame into readable text for LLM
            if context_df.empty:
                context_text = "No context messages available."
            else:
                formatted_lines = []
                for _, row in context_df.iterrows():
                    # Use the standardized formatting method from Utils
                    formatted_line = Utils.format_one_msg_for_prompt(row)
                    formatted_lines.append(formatted_line)
                context_text = '\n'.join(formatted_lines)
            
            # Load vision prompt template
            prompt_template = llm_client.load_prompt("vision_analysis_prompt.md")
            
            # Replace placeholders in the template
            final_prompt = prompt_template.replace("{{CONTEXT_MESSAGES}}", context_text)
            final_prompt = final_prompt.replace("{{IMAGE_URL}}", image_url)
            
            # Call vision-capable LLM directly - returns structured data
            call_label = f"vision_analysis_{image_url.split('/')[-1]}"
            analysis = llm_client.generate_structured(
                prompt=final_prompt,
                response_format=VisionResponse,
                call_label=call_label,
                image_url=image_url
            )
            
            return analysis
            
        except Exception as e:
            # Return default structure with error information
            return None

    @staticmethod
    def synthesize_visual_text(analysis: Optional[Dict[str, Any]]) -> str:
        """
        Convert vision analysis to readable text for message replacement.
        
        Args:
            analysis: Vision analysis dictionary, or None if analysis failed
            
        Returns:
            Synthesized text representation or error message
        """
        # Handle failed analysis
        if analysis is None:
            return "The user has uploaded a file but can not parse it."
            
        visual_fact = analysis["visual_analysis"]
        
        # Start with the description
        text_parts = [f"[VISUAL_FACT] {visual_fact.get('description', 'No description available')}"]
        
        # Add anchors if available
        meta = visual_fact.get("meta_info", {})
        anchor_parts = []
        
        if meta.get("tracking_ids"):
            anchor_parts.append(f"tracking={','.join(meta['tracking_ids'])}")
        if meta.get("order_ids"):
            anchor_parts.append(f"orders={','.join(meta['order_ids'])}")
        if meta.get("buyer_handles"):
            anchor_parts.append(f"buyers={','.join(meta['buyer_handles'])}")
            
        if anchor_parts:
            text_parts.append(" | " + " ".join(anchor_parts))
            
        # Add key conditions
        condition_parts = []
        if meta.get("has_damage"):
            damage_types = meta.get("damage_type", [])
            severity = meta.get("damage_severity", "unknown")
            if damage_types:
                condition_parts.append(f"damage={','.join(damage_types)}/{severity}")
                
        if meta.get("plant_health_status") and meta.get("plant_health_status") != "unknown":
            condition_parts.append(f"plant_health={meta['plant_health_status']}")
            
        if meta.get("box_condition") and meta.get("box_condition") != "unknown":
            condition_parts.append(f"packaging={meta['box_condition']}")
            
        if condition_parts:
            text_parts.append(" | " + " ".join(condition_parts))
            
        # Add confidence
        confidence = visual_fact.get("confidence", 0.0)
        text_parts.append(f" | vision_confidence={confidence:.2f}")
        
        return "".join(text_parts)