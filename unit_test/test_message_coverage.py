#!/usr/bin/env python3
"""
Unit test for message coverage validation in JSON output.

This test validates that all messages are properly assigned to cases without
duplicates or missing messages in the segmentation output.
"""

import json
import unittest
import re
from pathlib import Path
from typing import Set, List, Dict, Any


class TestMessageCoverage(unittest.TestCase):
    """Test message coverage and duplication in JSON output"""
    
    def setUp(self):
        """Load JSON data for testing"""
        self.json_file = Path(__file__).parent.parent / "out" / "cases_channel_a8249.json"
        
        if not self.json_file.exists():
            self.skipTest(f"JSON file not found: {self.json_file}")
            
        with open(self.json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.total_messages = self.data.get("total_messages", 0)
        self.global_cases = self.data.get("global_cases", [])
        
    def test_json_structure(self):
        """Test basic JSON structure is correct"""
        self.assertIn("channel_url", self.data)
        self.assertIn("total_messages", self.data) 
        self.assertIn("global_cases", self.data)
        self.assertIsInstance(self.global_cases, list)
        self.assertGreater(len(self.global_cases), 0, "No cases found in JSON")
        
    def test_message_completeness(self):
        """Verify all message IDs 0-(total_messages-1) are assigned exactly once"""
        expected_messages = set(range(self.total_messages))
        assigned_messages = set()
        
        # Collect all assigned message IDs
        for case in self.global_cases:
            msg_list = case.get("msg_list", [])
            assigned_messages.update(msg_list)
        
        # Find missing messages
        missing_messages = expected_messages - assigned_messages
        
        # Check for messages beyond expected range
        extra_messages = assigned_messages - expected_messages
        
        # Assertions
        self.assertEqual(
            missing_messages, 
            set(), 
            f"Missing messages: {sorted(missing_messages)}"
        )
        
        self.assertEqual(
            extra_messages,
            set(),
            f"Unexpected message IDs (beyond 0-{self.total_messages-1}): {sorted(extra_messages)}"
        )
        
        self.assertEqual(
            len(assigned_messages),
            self.total_messages,
            f"Expected {self.total_messages} unique messages, got {len(assigned_messages)}"
        )
        
    def test_no_duplicates(self):
        """Check for duplicate message assignments across all cases"""
        all_messages = []
        case_assignments = {}
        
        for case in self.global_cases:
            case_id = case.get("global_case_id", "unknown")
            msg_list = case.get("msg_list", [])
            
            for msg_id in msg_list:
                all_messages.append(msg_id)
                if msg_id in case_assignments:
                    case_assignments[msg_id].append(case_id)
                else:
                    case_assignments[msg_id] = [case_id]
        
        # Find duplicates
        duplicates = {
            msg_id: cases 
            for msg_id, cases in case_assignments.items() 
            if len(cases) > 1
        }
        
        self.assertEqual(
            duplicates,
            {},
            f"Duplicate message assignments found: {duplicates}"
        )
        
        # Also check total count matches
        unique_messages = len(set(all_messages))
        self.assertEqual(
            len(all_messages),
            unique_messages,
            f"Total messages ({len(all_messages)}) != unique messages ({unique_messages})"
        )
        
    def test_global_case_id_format(self):
        """Validate global_case_id format consistency"""
        pattern = re.compile(r'^\d+#\d+$')
        invalid_formats = []
        
        for case in self.global_cases:
            global_case_id = case.get("global_case_id", "")
            if not pattern.match(global_case_id):
                invalid_formats.append(global_case_id)
        
        self.assertEqual(
            invalid_formats,
            [],
            f"Invalid global_case_id formats found: {invalid_formats}"
        )
        
    def test_chunk_continuity(self):
        """Ensure chunks are properly numbered and cases exist"""
        chunk_cases = {}
        
        for case in self.global_cases:
            global_case_id = case.get("global_case_id", "")
            if '#' in global_case_id:
                chunk_id, case_id = global_case_id.split('#')
                chunk_id = int(chunk_id)
                case_id = int(case_id)
                
                if chunk_id not in chunk_cases:
                    chunk_cases[chunk_id] = []
                chunk_cases[chunk_id].append(case_id)
        
        # Check chunks are continuous starting from 0
        chunk_ids = sorted(chunk_cases.keys())
        expected_chunks = list(range(len(chunk_ids)))
        
        self.assertEqual(
            chunk_ids,
            expected_chunks,
            f"Chunks should be continuous starting from 0. Found: {chunk_ids}"
        )
        
        # Check each chunk has cases starting from 0
        for chunk_id, case_ids in chunk_cases.items():
            case_ids.sort()
            expected_case_ids = list(range(len(case_ids)))
            
            self.assertEqual(
                case_ids,
                expected_case_ids,
                f"Chunk {chunk_id} cases should start from 0. Found: {case_ids}"
            )
    
    def test_statistics(self):
        """Validate statistics match actual data"""
        # Count actual messages in all cases
        actual_message_count = 0
        for case in self.global_cases:
            msg_list = case.get("msg_list", [])
            actual_message_count += len(msg_list)
        
        self.assertEqual(
            actual_message_count,
            self.total_messages,
            f"Declared total_messages ({self.total_messages}) != actual messages in cases ({actual_message_count})"
        )
        
    def test_case_structure(self):
        """Validate each case has required fields"""
        required_fields = ["case_id", "msg_list", "summary", "status", "global_case_id"]
        
        for i, case in enumerate(self.global_cases):
            for field in required_fields:
                self.assertIn(
                    field, 
                    case, 
                    f"Case {i} missing required field: {field}"
                )
            
            # Validate msg_list is non-empty list
            msg_list = case.get("msg_list", [])
            self.assertIsInstance(msg_list, list, f"Case {i} msg_list should be a list")
            self.assertGreater(len(msg_list), 0, f"Case {i} msg_list should not be empty")
            
            # Validate all msg_list items are integers
            for msg_id in msg_list:
                self.assertIsInstance(
                    msg_id, 
                    int, 
                    f"Case {i} contains non-integer message ID: {msg_id}"
                )


def generate_coverage_report(json_file: str) -> Dict[str, Any]:
    """Generate detailed coverage report"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_messages = data.get("total_messages", 0)
    global_cases = data.get("global_cases", [])
    
    # Collect all message assignments
    assigned_messages = set()
    case_assignments = {}
    chunk_stats = {}
    
    for case in global_cases:
        global_case_id = case.get("global_case_id", "")
        msg_list = case.get("msg_list", [])
        
        # Track chunk stats
        if '#' in global_case_id:
            chunk_id = global_case_id.split('#')[0]
            if chunk_id not in chunk_stats:
                chunk_stats[chunk_id] = {"cases": 0, "messages": 0}
            chunk_stats[chunk_id]["cases"] += 1
            chunk_stats[chunk_id]["messages"] += len(msg_list)
        
        # Track message assignments
        for msg_id in msg_list:
            assigned_messages.add(msg_id)
            if msg_id in case_assignments:
                case_assignments[msg_id].append(global_case_id)
            else:
                case_assignments[msg_id] = [global_case_id]
    
    # Find issues
    expected_messages = set(range(total_messages))
    missing_messages = expected_messages - assigned_messages
    extra_messages = assigned_messages - expected_messages
    duplicates = {k: v for k, v in case_assignments.items() if len(v) > 1}
    
    return {
        "total_declared": total_messages,
        "total_assigned": len(assigned_messages),
        "total_cases": len(global_cases),
        "missing_messages": sorted(missing_messages),
        "extra_messages": sorted(extra_messages),
        "duplicate_assignments": duplicates,
        "chunk_statistics": chunk_stats,
        "coverage_complete": len(missing_messages) == 0 and len(extra_messages) == 0,
        "no_duplicates": len(duplicates) == 0
    }


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    # Generate and print coverage report
    json_file = Path(__file__).parent.parent / "out" / "cases_channel_a8249.json"
    if json_file.exists():
        print("\n" + "="*60)
        print("MESSAGE COVERAGE REPORT")
        print("="*60)
        
        report = generate_coverage_report(str(json_file))
        
        print(f"Total declared messages: {report['total_declared']}")
        print(f"Total assigned messages: {report['total_assigned']}")
        print(f"Total cases: {report['total_cases']}")
        print(f"Coverage complete: {report['coverage_complete']}")
        print(f"No duplicates: {report['no_duplicates']}")
        
        if report['missing_messages']:
            print(f"\nMissing messages: {report['missing_messages']}")
        
        if report['extra_messages']:
            print(f"\nExtra messages: {report['extra_messages']}")
            
        if report['duplicate_assignments']:
            print(f"\nDuplicate assignments:")
            for msg_id, cases in report['duplicate_assignments'].items():
                print(f"  Message {msg_id}: {cases}")
        
        print(f"\nChunk statistics:")
        for chunk_id, stats in report['chunk_statistics'].items():
            print(f"  Chunk {chunk_id}: {stats['cases']} cases, {stats['messages']} messages")
        
        if report['coverage_complete'] and report['no_duplicates']:
            print(f"\n✅ SUCCESS: All {report['total_declared']} messages properly assigned!")
        else:
            print(f"\n❌ ISSUES FOUND: Check missing/duplicate messages above")