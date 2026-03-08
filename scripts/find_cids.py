import re
from collections import Counter

def scan_for_cids(input_path, output_path):
    # Pattern to match (cid: followed by one or more digits and a closing )
    cid_pattern = r"\(cid:\d+\)"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    found_cids = re.findall(cid_pattern, content)
    counts = Counter(found_cids)
    
    print(f"🔍 Found {len(counts)} unique CID codes in {input_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("CID Code | Frequency | Suspected Character\n")
        f.write("---------|-----------|-------------------\n")
        for cid, count in counts.most_common():
            # We leave the suspected character blank for you to fill in
            f.write(f"{cid:<8} | {count:<9} | \n")

if __name__ == "__main__":
    raw_input = "data/raw/camp_raw_raw.txt"
    report_output = "data/raw/camp_raw_cid.txt"
    
    scan_for_cids(raw_input, report_output)
    print(f"CID report generated: {report_output}")