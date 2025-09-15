#!/usr/bin/env python3
import json
from collections import defaultdict, Counter
import sys

def inspect_metadata_file(filepath, sample_size=1000, check_specific_ids=None, id_range=None):
    """
    Inspect metadata file to understand structure and identify issues.
    
    Args:
        filepath: Path to the metadata .jsonl file
        sample_size: Number of entries to analyze for general statistics
        check_specific_ids: List of specific IDs to check
        id_range: Tuple (start, end) for checking a range of IDs
    
    Usage examples:
        python inspect_metadata.py vector_db.meta.jsonl 1000 100-200  # Check IDs 100-200
        python inspect_metadata.py vector_db.meta.jsonl 1000 150      # Check single ID 150
        python inspect_metadata.py vector_db.meta.jsonl 1000 100,101,102  # Check specific IDs
    """
    
    print(f"[INFO] Inspecting metadata file: {filepath}")
    if check_specific_ids:
        print(f"[INFO] Checking specific IDs: {check_specific_ids}")
    if id_range:
        print(f"[INFO] Checking ID range: {id_range[0]} to {id_range[1]}")
    print(f"[INFO] Analyzing first {sample_size} entries...\n")
    
    # Track statistics
    total_entries = 0
    key_counts = Counter()
    key_examples = defaultdict(list)
    empty_entries = 0
    entries_with_issues = []
    
    # Value type analysis
    value_types = defaultdict(Counter)
    
    # Track specific IDs if requested
    specific_id_results = {}
    if check_specific_ids:
        check_specific_ids = set(map(int, check_specific_ids))  # Convert to integers
    
    # Generate ID range if requested
    if id_range:
        start_id, end_id = id_range
        range_ids = set(range(start_id, end_id + 1))
        if check_specific_ids:
            check_specific_ids.update(range_ids)
        else:
            check_specific_ids = range_ids
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line_num > sample_size and not check_specific_ids:
                    break
                    
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    entry = json.loads(line)
                    
                    # Check if this is a specific ID we're looking for
                    if check_specific_ids and 'id' in entry and entry['id'] in check_specific_ids:
                        specific_id_results[entry['id']] = {
                            'line_number': line_num,
                            'entry': entry,
                            'has_document': bool(entry.get('document') or entry.get('doc')),
                            'has_id': bool(entry.get('id')),
                            'has_coords': bool(entry.get('coords')),
                            'metadata_would_be_empty': not bool(
                                (entry.get('document') or entry.get('doc')) and 
                                entry.get('id') and 
                                entry.get('coords') is not None
                            )
                        }
                        # Continue to next line to keep searching for specific IDs
                        if line_num > sample_size:
                            continue
                    
                    # Regular analysis (only within sample_size)
                    if line_num > sample_size:
                        continue
                        
                    total_entries += 1
                    
                    # Check if entry is empty or has no useful data
                    if not entry or all(v is None or v == "" for v in entry.values()):
                        empty_entries += 1
                        entries_with_issues.append((line_num, "Empty entry", entry))
                        continue
                    
                    # Track all keys present
                    for key in entry.keys():
                        key_counts[key] += 1
                        
                        # Store examples (first 3 for each key)
                        if len(key_examples[key]) < 3:
                            key_examples[key].append(entry[key])
                        
                        # Track value types
                        value = entry[key]
                        if value is None:
                            value_types[key]["None"] += 1
                        else:
                            value_types[key][type(value).__name__] += 1
                    
                    # Check for potential issues
                    if 'document' not in entry and 'doc' not in entry:
                        entries_with_issues.append((line_num, "Missing document field", entry))
                    
                    if 'id' not in entry:
                        entries_with_issues.append((line_num, "Missing id field", entry))
                    
                    if 'coords' in entry and (not entry['coords'] or len(entry['coords']) != 4):
                        entries_with_issues.append((line_num, "Invalid coords", entry))
                
                except json.JSONDecodeError as e:
                    entries_with_issues.append((line_num, f"JSON decode error: {e}", line[:100]))
    
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Print results
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total entries analyzed: {total_entries}")
    print(f"Empty/invalid entries: {empty_entries}")
    print(f"Entries with issues: {len(entries_with_issues)}")
    print()
    
    print("=" * 60)
    print("KEY FREQUENCY ANALYSIS")
    print("=" * 60)
    for key, count in key_counts.most_common():
        percentage = (count / total_entries) * 100 if total_entries > 0 else 0
        print(f"{key:15} : {count:6} entries ({percentage:5.1f}%)")
    print()
    
    print("=" * 60)
    print("KEY EXAMPLES & TYPES")
    print("=" * 60)
    for key in sorted(key_counts.keys()):
        print(f"\nKey: '{key}'")
        print(f"   Examples: {key_examples[key]}")
        print(f"   Types: {dict(value_types[key])}")
    
    if entries_with_issues:
        print("\n" + "=" * 60)
        print("POTENTIAL ISSUES FOUND")
        print("=" * 60)
        issue_summary = Counter()
        for line_num, issue_type, data in entries_with_issues[:20]:  # Show first 20 issues
            issue_summary[issue_type] += 1
            print(f"Line {line_num}: {issue_type}")
            if isinstance(data, dict):
                print(f"   Data: {json.dumps(data, ensure_ascii=False)[:200]}")
            else:
                print(f"   Data: {str(data)[:200]}")
        
        print(f"\nIssue Summary:")
        for issue_type, count in issue_summary.most_common():
            print(f"   {issue_type}: {count} occurrences")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Check what the Pinecone code expects vs what we have
    expected_mappings = {
        'doc': ['document', 'doc'],
        'patch': ['patch', 'id'], 
        'coords': ['coords']
    }
    
    print("Current Pinecone mapping expectations:")
    for pinecone_key, possible_keys in expected_mappings.items():
        available = [k for k in possible_keys if k in key_counts]
        print(f"   {pinecone_key:10} <- {possible_keys} | Available: {available}")
    
    # Suggest fixes
    if 'document' in key_counts and 'doc' not in key_counts:
        print("\n[OK] 'document' field is present - mapping should work")
    elif 'doc' in key_counts and 'document' not in key_counts:
        print("\n[OK] 'doc' field is present - mapping should work")
    else:
        print("\n[WARN] Check document field mapping")
    
    if 'id' in key_counts:
        print("[OK] 'id' field is present - will be mapped to 'patch' in Pinecone")
    
    if 'coords' in key_counts:
        coords_with_data = value_types['coords'].get('list', 0)
        coords_none = value_types['coords'].get('None', 0) 
        print(f"[OK] 'coords' field present: {coords_with_data} with data, {coords_none} None values")
    
    # Display specific ID results if any were requested
    if specific_id_results:
        print("\n" + "=" * 60)
        print("SPECIFIC ID ANALYSIS")
        print("=" * 60)
        
        found_count = len(specific_id_results)
        requested_count = len(check_specific_ids) if check_specific_ids else 0
        
        print(f"Found {found_count} out of {requested_count} requested IDs")
        
        if id_range:
            print(f"ID Range: {id_range[0]} to {id_range[1]} ({id_range[1] - id_range[0] + 1} IDs total)")
        
        # Show summary of metadata status
        has_metadata = sum(1 for result in specific_id_results.values() if not result['metadata_would_be_empty'])
        no_metadata = len(specific_id_results) - has_metadata
        
        print(f"\nMetadata Status Summary:")
        print(f"  IDs with metadata: {has_metadata}")
        print(f"  IDs without metadata: {no_metadata}")
        
        print("\nDetailed Results:")
        for id_val in sorted(specific_id_results.keys()):
            result = specific_id_results[id_val]
            status = "NO METADATA" if result['metadata_would_be_empty'] else "HAS METADATA"
            print(f"  ID {id_val:6} (line {result['line_number']:6}): {status}")
            print(f"    Document: {'✓' if result['has_document'] else '✗'}")
            print(f"    ID field: {'✓' if result['has_id'] else '✗'}")
            print(f"    Coords:   {'✓' if result['has_coords'] else '✗'}")
            if result['metadata_would_be_empty']:
                print(f"    Entry: {json.dumps(result['entry'], ensure_ascii=False)[:100]}...")
        
        # Missing IDs
        if requested_count > found_count:
            missing_ids = check_specific_ids - set(specific_id_results.keys())
            print(f"\nMissing IDs (not found in file): {sorted(list(missing_ids))[:20]}")
            if len(missing_ids) > 20:
                print(f"  ... and {len(missing_ids) - 20} more")


def parse_id_range(range_str):
    """Parse ID range string like '100-200' into tuple (100, 200)"""
    try:
        if '-' in range_str:
            start, end = range_str.split('-', 1)
            return (int(start), int(end))
        else:
            # Single ID
            id_val = int(range_str)
            return (id_val, id_val)
    except ValueError:
        raise ValueError(f"Invalid ID range format: {range_str}. Use format like '100-200' or '150'")

if __name__ == "__main__":
    metadata_file = "vector_db.meta.jsonl"
    sample_size = 10000  # Analyze first 10k entries
    check_specific_ids = None
    id_range = None
    
    if len(sys.argv) > 1:
        metadata_file = sys.argv[1]
    if len(sys.argv) > 2:
        sample_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        # Third argument can be ID range like "100-200" or specific IDs comma-separated
        range_arg = sys.argv[3]
        if '-' in range_arg and not range_arg.startswith('-'):
            # It's a range like "100-200"
            id_range = parse_id_range(range_arg)
        else:
            # It's specific IDs like "100,101,102"
            check_specific_ids = range_arg.split(',')
    
    inspect_metadata_file(metadata_file, sample_size, check_specific_ids, id_range)