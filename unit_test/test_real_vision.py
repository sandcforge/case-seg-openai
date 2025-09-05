#!/usr/bin/env python3
"""
Test script to see actual LLM vision analysis results.

This script loads real images from support_messages_multi_ch.csv and calls
the LLM to analyze them with context using Session and VisionProcessor classes.
"""

import pandas as pd
import json
import os
from pathlib import Path
from session import Session
from llm_client import LLMClient
from vision_processor import VisionProcessor


def find_image_in_dataframe(df_clean: pd.DataFrame, image_url: str) -> tuple:
    """
    Find image URL in DataFrame and return channel and msg_ch_idx
    
    Args:
        df_clean: Cleaned DataFrame from Session
        image_url: Image URL to search for
        
    Returns:
        tuple: (channel_url, msg_ch_idx, file_message_row) or (None, None, None) if not found
    """
    # Search for the image URL in FILE messages
    file_messages = df_clean[df_clean['Type'] == 'FILE']
    
    # Try exact match first
    exact_match = file_messages[file_messages['File URL'] == image_url]
    if len(exact_match) > 0:
        row = exact_match.iloc[0]
        return row['Channel URL'], row['msg_ch_idx'], row
    
    # Try partial match (in case URL has extra parameters)
    for _, row in file_messages.iterrows():
        if image_url in str(row['File URL']) or str(row['File URL']) in image_url:
            return row['Channel URL'], row['msg_ch_idx'], row
            
    return None, None, None


def get_user_image_url() -> str:
    """Get image URL from user input"""
    print("\n" + "="*60)
    print("🖼️  Enter Image URL for Analysis")
    print("="*60)
    print("Enter an image URL to analyze (or 'quit' to exit):")
    
    try:
        url = input("🔗 Image URL: ").strip()
        return url
    except (EOFError, KeyboardInterrupt):
        return "quit"


def main():
    """Test VisionProcessor with specific image URLs provided by user"""
    
    # Load environment variables from .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Use Session to process multi-channel CSV
    csv_file = Path("assets/support_messages_5ch.csv")
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return
        
    print("📁 Processing CSV data with Session...")
    session = Session(
        input_file=str(csv_file),
        output_dir='out',
        chunk_size=60,
        overlap=10,
        model='gpt-4o',  # Use vision-capable model
        enable_classification=False  # Skip classification for this test
    )
    
    # Process file data to get cleaned DataFrame
    session.process_file_data()
    df_clean = session.df_clean
    
    # Get statistics about available images
    file_messages = df_clean[df_clean['Type'] == 'FILE']
    unique_channels = df_clean['Channel URL'].unique()
    
    print(f"📊 Dataset Statistics:")
    print(f"   📁 Total channels: {len(unique_channels)}")
    print(f"   📨 Total messages: {len(df_clean)}")
    print(f"   🖼️  Total FILE messages: {len(file_messages)}")
    
    if len(file_messages) == 0:
        print("❌ No FILE messages found in dataset")
        return
        
    # Show some example image URLs for reference
    print(f"\n💡 Example image URLs from dataset:")
    for i, (_, row) in enumerate(file_messages.head(3).iterrows()):
        print(f"   {i+1}. {row['File URL']}")
    if len(file_messages) > 3:
        print(f"   ... and {len(file_messages) - 3} more images")
    
    # Initialize LLM client and VisionProcessor
    try:
        print("🤖 Initializing LLM client...")
        # Create LLM client directly with vision model
        llm_client = LLMClient(model='gpt-4o')  # Use vision-capable model
        vision_processor = VisionProcessor(llm_client)
        print("✅ LLM client initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize LLM client: {e}")
        print("💡 Make sure you have OPENAI_API_KEY set in your environment")
        return
    
    # Interactive image analysis loop
    analysis_count = 0
    while True:
        # Get image URL from user
        image_url = get_user_image_url()
        
        if image_url.lower() in ['quit', 'exit', 'q']:
            break
            
        if not image_url:
            print("❌ Please enter a valid image URL")
            continue
            
        analysis_count += 1
        print(f"\n{'='*60}")
        print(f"🖼️  Analysis #{analysis_count}")
        print(f"{'='*60}")
        
        # Find image in dataset
        print(f"🔍 Searching for image URL in dataset...")
        channel_url, msg_ch_idx, file_msg = find_image_in_dataframe(df_clean, image_url)
        
        if channel_url is None:
            print(f"❌ Image URL not found in dataset: {image_url}")
            print(f"💡 Make sure the URL exactly matches one from the dataset")
            continue
            
        print(f"✅ Found image in dataset!")
        print(f"📍 Message Index: {msg_ch_idx}")
        print(f"📁 Channel: {channel_url}")
        print(f"🔗 Image URL: {image_url}")
        print(f"⏰ Created: {file_msg.get('Created Time', 'N/A')}")
        print(f"👤 Sender: {file_msg.get('Sender ID', 'N/A')}")
        
        # Debug: Check channel message distribution
        channel_messages = df_clean[df_clean['Channel URL'] == channel_url]
        print(f"🐛 Debug - Channel has {len(channel_messages)} total messages")
        print(f"🐛 Debug - msg_ch_idx range: {channel_messages['msg_ch_idx'].min()} to {channel_messages['msg_ch_idx'].max()}")
        print(f"🐛 Debug - Unique msg_ch_idx count: {channel_messages['msg_ch_idx'].nunique()}")
        
        # Check for duplicate msg_ch_idx
        duplicates = channel_messages[channel_messages.duplicated('msg_ch_idx', keep=False)]
        if len(duplicates) > 0:
            print(f"🐛 ⚠️  WARNING: Found {len(duplicates)} messages with duplicate msg_ch_idx!")
            print("🐛 Duplicate msg_ch_idx values:")
            for idx in duplicates['msg_ch_idx'].unique():
                dupe_msgs = channel_messages[channel_messages['msg_ch_idx'] == idx]
                print(f"🐛   idx={idx}: {len(dupe_msgs)} messages")
                for _, row in dupe_msgs.iterrows():
                    print(f"🐛     - {row['Type']} | {row.get('Sender ID', 'N/A')} | {str(row.get('Message', ''))[:30]}")
        else:
            print("🐛 ✅ No duplicate msg_ch_idx found in channel")
        
        # Get context using VisionProcessor method (pass channel DataFrame)
        try:
            context_df = vision_processor.get_context_for_image(
                channel_df=channel_messages,  # Use the filtered channel DataFrame
                image_msg_ch_idx=msg_ch_idx,
                context_size=5
            )
            print(f"📄 Context: {len(context_df)} messages (MESG + image message)")
            
            # Debug: Show context msg_ch_idx sequence
            context_indices = sorted(context_df['msg_ch_idx'].tolist())
            print(f"🐛 Debug - Context msg_ch_idx values: {context_indices}")
                
        except Exception as e:
            print(f"❌ Error getting context: {e}")
            continue
        
        # Show context messages
        print("\n📝 Context Messages:")
        for _, ctx_msg in context_df.iterrows():
            ctx_idx = ctx_msg.get('msg_ch_idx', 'N/A')
            ctx_sender = str(ctx_msg.get('Sender ID', 'N/A'))[:15]
            ctx_type = ctx_msg.get('Type', 'MESG')
            ctx_message = str(ctx_msg.get('Message', ''))[:50] + ('...' if len(str(ctx_msg.get('Message', ''))) > 50 else '')
            
            marker = "👉" if ctx_idx == msg_ch_idx else "  "
            print(f"{marker} {str(ctx_idx):>2} | {ctx_sender:<15} | {ctx_type:<4} | {ctx_message}")
        
        # Analyze image with context
        print(f"\n🔍 Analyzing image with LLM...")
        
        try:
            result = vision_processor.analyze_image_with_context(
                context_df=context_df,
                image_url=image_url
            )
            
            # Display results
            print("✅ Analysis completed!")
            print("\n📊 Vision Analysis Results:")
            print("-" * 40)
            
            analysis = result.get('visual_analysis', {})
            
            # Description
            description = analysis.get('description', 'No description')
            print(f"📖 Description: {description}")
            
            # Intent
            intent = analysis.get('customer_intent', 'Unknown intent')
            print(f"🎯 Customer Intent: {intent}")
            
            # Confidence
            confidence = analysis.get('confidence', 0.0)
            print(f"🎲 Confidence: {confidence:.2f}")
            
            # Key metadata
            meta_info = analysis.get('meta_info', {})
            print(f"\n🔍 Key Metadata:")
            
            if meta_info.get('tracking_ids'):
                print(f"  📦 Tracking IDs: {meta_info['tracking_ids']}")
            if meta_info.get('order_ids'):
                print(f"  🛒 Order IDs: {meta_info['order_ids']}")
            if meta_info.get('has_damage'):
                damage_types = meta_info.get('damage_type', [])
                severity = meta_info.get('damage_severity', 'unknown')
                print(f"  ⚠️  Damage: {', '.join(damage_types)} ({severity})")
            if meta_info.get('plant_health_status', 'unknown') != 'unknown':
                print(f"  🌱 Plant Health: {meta_info['plant_health_status']}")
            if meta_info.get('carrier', 'unknown') != 'unknown':
                print(f"  🚚 Carrier: {meta_info['carrier']}")
            
            # Synthesized text
            print(f"\n🔄 Synthesized Text:")
            visual_text = vision_processor.synthesize_visual_text(result)
            print(f"   {visual_text}")
            
        except Exception as e:
            print(f"❌ Error analyzing image: {e}")
    
    print(f"\n🎉 Completed {analysis_count} image analyses!")


if __name__ == "__main__":
    main()