#!/usr/bin/env python3
"""
Unit test for VisionProcessor functionality using real CSV data.

This test validates that VisionProcessor can correctly handle FILE type messages
from support_messages_andy.csv and generate proper vision analysis output.
"""

import unittest
import pandas as pd  # type: ignore
import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vision_processor import VisionProcessor


class TestVisionProcessor(unittest.TestCase):
    """Test VisionProcessor with real CSV data and mocked LLM responses"""
    
    def setUp(self):
        """Set up test fixtures with real CSV data and mocked LLM client"""
        # Load real CSV data
        self.csv_file = Path(__file__).parent.parent / "assets" / "support_messages_andy.csv"
        
        if not self.csv_file.exists():
            self.skipTest(f"CSV file not found: {self.csv_file}")
            
        self.df = pd.read_csv(self.csv_file)
        
        # Create mock LLM client
        self.mock_llm_client = Mock()
        self.mock_llm_client.load_prompt.return_value = "Mock prompt template with {{CONTEXT_MESSAGES}} and {{IMAGE_URL}} and {{MSG_CH_IDX}}"
        
        # Initialize VisionProcessor with mocked client
        self.vision_processor = VisionProcessor(self.mock_llm_client)
        
        # Sample vision analysis response for mocking
        self.sample_vision_response = {
            "visual_analysis": {
                "description": "Image shows multiple packages with shipping labels. Customer is demonstrating packaging conditions and visible tracking information for their recent orders.",
                "customer_intent": "Customer is providing evidence of received packages to show shipping condition and verify delivery",
                "meta_info": {
                    "tracking_ids": ["1Z999AA1234567890"],
                    "order_ids": ["ORDER-12345"],
                    "buyer_handles": ["@testuser"],
                    "visible_text": ["UPS", "DELIVERED"],
                    "has_damage": False,
                    "damage_type": [],
                    "damage_severity": "none",
                    "plant_health_status": "unknown",
                    "plant_symptoms": [],
                    "plant_condition": "unknown",
                    "box_condition": "intact",
                    "protection_used": ["bubble_wrap"],
                    "labeling_status": "correct",
                    "carrier": "UPS",
                    "delivery_status": "delivered",
                    "address_visible": True
                },
                "confidence": 0.85
            }
        }
    
    def test_identify_file_messages(self):
        """Test that CSV contains FILE type messages with image URLs"""
        # Check that CSV has FILE type messages
        file_messages = self.df[self.df['Type'] == 'FILE']
        
        self.assertGreater(len(file_messages), 0, "CSV should contain FILE type messages")
        
        # Verify specific image URLs are present
        expected_images = [
            "lyoqyYb.jpg",
            "wdSGedA.jpg", 
            "HQba1m6.jpg",
            "lH5jOw4.jpg",
            "W8XsId8.jpg",
            "Tnyc0Iv.jpg"
        ]
        
        file_urls = file_messages['File URL'].astype(str)
        for image in expected_images:
            image_found = any(image in url for url in file_urls)
            self.assertTrue(image_found, f"Expected image {image} should be found in FILE messages")
            
        print(f"Found {len(file_messages)} FILE type messages with images")
    
    def test_csv_data_structure(self):
        """Test that CSV has required columns for vision processing"""
        required_columns = ['Type', 'File URL', 'Message', 'Sender ID', 'Created Time']
        
        for col in required_columns:
            self.assertIn(col, self.df.columns, f"CSV should contain {col} column")
            
        # Test that we have the channel URL that was selected
        expected_channel = "sendbird_group_channel_215482988_b374305ff3e440674e786d63916f1d5aacda8249"
        channel_found = any(expected_channel in str(url) for url in self.df['Channel URL'])
        self.assertTrue(channel_found, f"Expected channel {expected_channel} should be found")
    
    def test_context_formatting_integration(self):
        """Test context formatting integration within analyze_image_with_context"""
        # Mock the structured LLM call to return our sample response
        self.mock_llm_client.generate_structured.return_value = self.sample_vision_response
        
        # Get a small sample of messages for context
        sample_messages = self.df.head(5).copy()
        
        # Add msg_ch_idx column (simulating processed data)
        sample_messages['msg_ch_idx'] = range(len(sample_messages))
        sample_messages['role'] = 'user'  # Add role column
        
        # Test that analyze_image_with_context handles context formatting correctly
        test_image_url = "https://example.com/test.jpg"
        
        result = self.vision_processor.analyze_image_with_context(
            context_df=sample_messages,
            image_url=test_image_url
        )
        
        # Verify the method completed successfully (which means context formatting worked)
        self.assertIn('visual_analysis', result)
        
        # Verify that the LLM client was called with a formatted prompt
        self.mock_llm_client.generate_structured.assert_called_once()
        call_args = self.mock_llm_client.generate_structured.call_args
        
        # The prompt should contain formatted context messages
        prompt = call_args.kwargs['prompt'] if 'prompt' in call_args.kwargs else call_args[0][0]
        self.assertIsInstance(prompt, str)
        self.assertIn('|', prompt)  # Should contain pipe separators from formatted context
        
        print("Context formatting integration test passed")
    
    def test_analyze_image_with_context(self):
        """Test main analysis functionality with mocked LLM response"""
        # Mock the structured LLM call to return our sample response
        self.mock_llm_client.generate_structured.return_value = self.sample_vision_response
        
        # Get context from real CSV data
        context_messages = self.df.head(3).copy()
        context_messages['msg_ch_idx'] = range(len(context_messages))
        context_messages['role'] = 'user'
        
        # Test with a real image URL from the CSV
        file_messages = self.df[self.df['Type'] == 'FILE']
        if len(file_messages) > 0:
            test_image_url = file_messages.iloc[0]['File URL']
            test_msg_idx = 5
            
            # Call analyze_image_with_context
            result = self.vision_processor.analyze_image_with_context(
                context_df=context_messages,
                image_url=test_image_url
            )
            
            # Verify result structure
            self.assertIn('visual_analysis', result)
            analysis = result['visual_analysis']
            
            # Check required fields
            self.assertIn('description', analysis)
            self.assertIn('customer_intent', analysis)
            self.assertIn('meta_info', analysis)
            self.assertIn('confidence', analysis)
            
            # Verify flattened meta_info structure
            meta_info = analysis['meta_info']
            expected_fields = [
                'tracking_ids', 'order_ids', 'buyer_handles', 'visible_text',
                'has_damage', 'damage_type', 'damage_severity',
                'plant_health_status', 'plant_symptoms', 'plant_condition',
                'box_condition', 'protection_used', 'labeling_status',
                'carrier', 'delivery_status', 'address_visible'
            ]
            
            for field in expected_fields:
                self.assertIn(field, meta_info, f"meta_info should contain {field}")
                
            # msg_ch_idx is no longer used in the response
            self.assertNotIn('msg_ch_idx', analysis)
            
            print(f"Successfully analyzed image: {test_image_url}")
    
    def test_synthesize_visual_text(self):
        """Test conversion of vision analysis to readable text"""
        # Create sample analysis data
        sample_analysis = {
            "visual_analysis": {
                "description": "Customer shows damaged plant packaging with visible tracking number",
                "customer_intent": "Reporting shipping damage",
                "meta_info": {
                    "tracking_ids": ["1Z999AA1234567890"],
                    "order_ids": ["ORDER-12345"],
                    "buyer_handles": ["@customer"],
                    "has_damage": True,
                    "damage_type": ["packaging_damage"],
                    "damage_severity": "moderate",
                    "plant_health_status": "damaged",
                    "box_condition": "crushed",
                    "carrier": "UPS",
                    "confidence": 0.92
                },
                "confidence": 0.92
            }
        }
        
        # Test synthesis
        visual_text = self.vision_processor.synthesize_visual_text(sample_analysis)
        
        # Verify structure
        self.assertTrue(visual_text.startswith('[VISUAL_FACT]'))
        self.assertIn('damaged plant packaging', visual_text)
        self.assertIn('tracking=1Z999AA1234567890', visual_text)
        self.assertIn('orders=ORDER-12345', visual_text)
        self.assertIn('buyers=@customer', visual_text)
        self.assertIn('damage=packaging_damage/moderate', visual_text)
        self.assertIn('plant_health=damaged', visual_text)
        self.assertIn('packaging=crushed', visual_text)
        self.assertIn('vision_confidence=0.92', visual_text)
        
        print(f"Synthesized text: {visual_text[:100]}...")
    
    def test_error_handling(self):
        """Test error handling when LLM call fails"""
        # Mock LLM to raise an exception
        self.mock_llm_client.generate_structured.side_effect = Exception("API call failed")
        
        context_messages = pd.DataFrame({
            'msg_ch_idx': [0, 1, 2],
            'Sender ID': ['user1', 'support', 'user1'], 
            'role': ['user', 'customer_service', 'user'],
            'Created Time': ['2025-01-01', '2025-01-01', '2025-01-01'],
            'Message': ['Hello', 'How can I help?', 'I have an issue']
        })
        
        # Test error handling
        result = self.vision_processor.analyze_image_with_context(
            context_df=context_messages,
            image_url="https://example.com/test.jpg"
        )
        
        # Should return error response structure
        self.assertIn('visual_analysis', result)
        analysis = result['visual_analysis']
        
        self.assertIn('Error analyzing image', analysis['description'])
        self.assertEqual(analysis['confidence'], 0.0)
        
        print("Error handling test passed")
    
    def test_default_meta_info_structure(self):
        """Test that default meta_info has all required fields"""
        default_meta = self.vision_processor._get_default_meta_info()
        
        # Verify all expected fields are present
        expected_fields = {
            'tracking_ids': list,
            'order_ids': list,
            'buyer_handles': list,
            'visible_text': list,
            'has_damage': bool,
            'damage_type': list,
            'damage_severity': str,
            'plant_health_status': str,
            'plant_symptoms': list,
            'plant_condition': str,
            'box_condition': str,
            'protection_used': list,
            'labeling_status': str,
            'carrier': str,
            'delivery_status': str,
            'address_visible': bool
        }
        
        for field, expected_type in expected_fields.items():
            self.assertIn(field, default_meta, f"default_meta should contain {field}")
            self.assertIsInstance(default_meta[field], expected_type, 
                               f"{field} should be of type {expected_type}")
                               
        print("Default meta_info structure validated")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)