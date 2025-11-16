#!/usr/bin/env python3
"""
Test script for BigQuery integration

This script tests the Utils.query_bigquery() method.
Make sure BIGQUERY_CREDENTIALS_JSON is set in your .env file before running.

Usage:
    cd unit_test
    python test_bigquery.py
"""

import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import Utils
from dotenv import load_dotenv

def test_simple_query():
    """Test a simple BigQuery query"""
    print("=" * 60)
    print("Testing BigQuery Connection")
    print("=" * 60)

    try:
        # Simple test query
        sql = "SELECT 'Hello from BigQuery!' AS message, CURRENT_TIMESTAMP() AS timestamp"

        print(f"\n📊 Executing query:")
        print(f"   {sql}")
        print()

        results = Utils.query_bigquery(sql=sql)

        print(f"✅ Query successful!")
        print(f"   Returned {len(results)} row(s)")
        print()

        # Print results
        for i, row in enumerate(results, 1):
            print(f"Row {i}:")
            for key, value in row.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

    return True


def test_parameterized_query():
    """Test a parameterized query"""
    print("\n" + "=" * 60)
    print("Testing Parameterized Query")
    print("=" * 60)

    try:
        # Parameterized query
        sql = """
            SELECT
                @param_string AS string_value,
                @param_int AS int_value,
                @param_float AS float_value
        """

        query_params = [
            {"name": "param_string", "type": "STRING", "value": "test_value"},
            {"name": "param_int", "type": "INT64", "value": 42},
            {"name": "param_float", "type": "FLOAT64", "value": 3.14}
        ]

        print(f"\n📊 Executing parameterized query:")
        print(f"   SQL: {sql.strip()}")
        print(f"   Parameters: {query_params}")
        print()

        results = Utils.query_bigquery(sql=sql, query_params=query_params)

        print(f"✅ Query successful!")
        print(f"   Returned {len(results)} row(s)")
        print()

        # Print results
        for i, row in enumerate(results, 1):
            print(f"Row {i}:")
            for key, value in row.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

    return True


def test_admin_logs_query():
    """Test querying public.admin_logs table"""
    print("\n" + "=" * 60)
    print("Testing public.admin_logs Table Query")
    print("=" * 60)

    try:
        # Query the first 10 records from admin_logs
        sql = """
            SELECT *
            FROM `plantstory.public.admin_logs`
            LIMIT 10
        """

        print(f"\n📊 Querying admin_logs table...")
        print(f"   SQL: {sql.strip()}")
        print()

        results = Utils.query_bigquery(sql=sql)

        print(f"✅ Query successful!")
        print(f"   Returned {len(results)} row(s)")
        print()

        # Print sample record if results exist
        if results:
            print(f"   Sample record (first row):")
            for key, value in results[0].items():
                # Truncate long values for display
                display_value = str(value)[:100] + "..." if len(str(value)) > 100 else value
                print(f"     {key}: {display_value}")

            # Show available columns
            print(f"\n   Available columns: {list(results[0].keys())}")
        else:
            print("   No records found in admin_logs table")

    except Exception as e:
        print(f"❌ Query failed: {e}")
        return False

    return True


def print_order_info_readable(order_info):
    """Print order info in a readable format"""
    import json

    order_data = order_info['order_data']
    admin_logs = order_info['admin_logs']
    status_history = order_info['order_status_history']

    # Header
    print("\n" + "=" * 80)
    print("ORDER INFORMATION")
    print("=" * 80)

    # Basic Info
    print("\n📦 BASIC INFO")
    print("-" * 80)
    print(f"  Order ID:          {order_data.get('order_id', 'N/A')}")
    print(f"  Order Number:      {order_data.get('order_number', 'N/A')}")
    print(f"  Status:            {order_data.get('order_completion_status', 'N/A')}")
    print(f"  Created Time:      {order_data.get('order_created_time', 'N/A')}")
    print(f"  Updated Time:      {order_data.get('order_updated_time', 'N/A')}")
    print(f"  Order Source:      {order_data.get('order_source', 'N/A')}")
    print(f"  Platform:          {order_data.get('req_platform', 'N/A')}")
    print(f"  Is Test Order:     {order_data.get('is_test_order', 'N/A')}")

    # Financial Info
    print("\n💰 FINANCIAL INFO")
    print("-" * 80)
    print(f"  GMV:                     ${order_data.get('gmv', 0):.2f}")
    print(f"  Order Amount:            ${order_data.get('order_amount', 0):.2f}")
    print(f"  Net Order Amount:        ${order_data.get('net_order_amount', 0):.2f}")
    print(f"  Shipping Fee:            ${order_data.get('shipping_fee', 0):.2f}")
    print(f"  Tax Amount:              ${order_data.get('tax_amount', 0):.2f}")
    print(f"  Platform Fee:            ${order_data.get('platform_fee_amount', 0):.2f}")
    print(f"  Payment Fee:             ${order_data.get('payment_fee_amount', 0):.2f}")
    print(f"  Seller Income:           ${order_data.get('seller_income_amount', 0):.2f}")
    print(f"  Platform Coupon:         ${order_data.get('platform_coupon_amount', 0):.2f}")
    print(f"  Seller Coupon:           ${order_data.get('seller_coupon_amount', 0):.2f}")

    # Refund Info (if any)
    total_refund = order_data.get('total_refund_amount', 0)
    if total_refund and total_refund > 0:
        print(f"\n  Refund Status:           {order_data.get('refund_status', 'N/A')}")
        print(f"  Total Refund Amount:     ${total_refund:.2f}")
        print(f"  Refund Created Time:     {order_data.get('refund_created_time', 'N/A')}")

    # Parties
    print("\n👥 PARTIES")
    print("-" * 80)
    print(f"  Buyer ID:          {order_data.get('buyer_id', 'N/A')}")
    print(f"  Buyer Name:        {order_data.get('buyer_name', 'N/A')}")
    print(f"  Seller ID:         {order_data.get('seller_id', 'N/A')}")
    print(f"  Seller Name:       {order_data.get('seller_name', 'N/A')}")

    # Shipping Info
    print("\n🚚 SHIPPING INFO")
    print("-" * 80)
    print(f"  Shipment Method:   {order_data.get('shipment_method', 'N/A')}")
    print(f"  Shipping Status:   {order_data.get('shipping_status', 'N/A')}")
    print(f"  Carrier:           {order_data.get('shipment_carrier', 'N/A')}")
    print(f"  Tracking Number:   {order_data.get('tracking_number', 'N/A')}")
    print(f"  Shipped Time:      {order_data.get('shipment_created_time', 'N/A')}")
    print(f"  Delivered Time:    {order_data.get('shipment_delievered_time', 'N/A')}")
    print(f"\n  Buyer Address:")
    print(f"    Name:            {order_data.get('buyer_address_name', 'N/A')}")
    print(f"    Line 1:          {order_data.get('buyer_address_line1', 'N/A')}")
    print(f"    Line 2:          {order_data.get('buyer_address_line2', 'N/A')}")
    print(f"    City:            {order_data.get('buyer_city', 'N/A')}")
    print(f"    State:           {order_data.get('buyer_state', 'N/A')}")
    print(f"    ZIP:             {order_data.get('buyer_zip_code', 'N/A')}")
    print(f"    Country:         {order_data.get('buyer_country', 'N/A')}")
    print(f"    Phone:           {order_data.get('buyer_phone', 'N/A')}")

    # Items Info
    print("\n📋 ORDER ITEMS")
    print("-" * 80)
    print(f"  Item Count:        {order_data.get('order_items_count', 'N/A')}")
    items_desc = order_data.get('order_items_description', 'N/A')
    if items_desc and items_desc != 'N/A':
        # Truncate if too long
        if len(str(items_desc)) > 200:
            print(f"  Description:       {str(items_desc)[:200]}...")
        else:
            print(f"  Description:       {items_desc}")

    # Admin Logs
    print("\n" + "=" * 80)
    print(f"ADMIN LOGS ({len(admin_logs)} records)")
    print("=" * 80)

    if admin_logs:
        for i, log in enumerate(admin_logs, 1):
            print(f"\n[{i}] {log.get('action', 'N/A').upper()} - {log.get('action_status', 'N/A')}")
            print(f"    Table:         {log.get('table_name', 'N/A')}")
            print(f"    Action Type:   {log.get('action_type', 'N/A')}")
            print(f"    Operator:      {log.get('ops_name', 'N/A')} (ID: {log.get('ops_user_id', 'N/A')})")
            print(f"    Time:          {log.get('created_time', 'N/A')}")
            if log.get('reason'):
                print(f"    Reason:        {log.get('reason')}")
            if log.get('payload'):
                print(f"    Payload:")
                try:
                    payload_str = json.dumps(log.get('payload'), indent=6, ensure_ascii=False)
                    print(payload_str)
                except:
                    print(f"      {log.get('payload')}")
    else:
        print("\n  No admin logs found for this order.")

    # Status History
    print("\n" + "=" * 80)
    print(f"STATUS HISTORY ({len(status_history)} records)")
    print("=" * 80)

    if status_history:
        print(f"\n  {'#':<4} {'Status':<15} {'Actor':<25} {'Time':<30}")
        print("  " + "-" * 76)
        for i, status in enumerate(status_history, 1):
            status_val = status.get('status', 'N/A')
            actor_val = status.get('actor', 'N/A')
            time_val = str(status.get('created_time', 'N/A'))
            print(f"  {i:<4} {status_val:<15} {actor_val:<25} {time_val:<30}")
    else:
        print("\n  No status history found for this order.")

    print("\n" + "=" * 80)


def test_get_order_info(order_number="e6vfNXx2VFnOG6K33DGWL"):
    """Test get_order_info() method"""
    print("\n" + "=" * 60)
    print("Testing get_order_info() Method")
    print("=" * 60)

    try:
        # Test with specified order_number
        print(f"\n📊 Querying order: {order_number}")
        order_info = Utils.get_order_info(order_number)

        if order_info:
            print(f"✅ Query successful!")

            # Print in readable format
            print_order_info_readable(order_info)
        else:
            print(f"❌ Order not found: {order_number}")
            return False

        # Test with non-existent order
        print(f"\n\n📊 Testing with non-existent order...")
        non_existent = Utils.get_order_info("NONEXISTENT12345")

        if non_existent is None:
            print(f"✅ Correctly returned None for non-existent order")
        else:
            print(f"❌ Should have returned None for non-existent order")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests"""
    load_dotenv()

    print("\n🧪 BigQuery Integration Tests\n")

    # Test 1: Simple query
    # test1_passed = test_simple_query()

    # # Test 2: Parameterized query
    # test2_passed = test_parameterized_query()

    # # Test 3: Admin logs query
    # test3_passed = test_admin_logs_query()

    # Test 4: get_order_info method
    test4_passed = test_get_order_info()


if __name__ == "__main__":
    main()
