import csv
import re

def parse_corpora_tsv(tsv_path):
    entries = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            # Example columns: ID, Name, Date, File Count, Download Path, Filename + details
            if len(row) < 6:
                continue
            dataset_id = row[0]
            file_count = row[3]
            file_info = row[5]

            # Extract file size and checksum from file_info (simple regex)
            size_match = re.search(r'File Size: ([\d\.]+) (GB|MB|KB)', file_info)
            checksum_match = re.search(r'MD5 Checksum: ([a-f0-9]+)', file_info)
            size = size_match.group(1) + " " + size_match.group(2) if size_match else "unknown"
            checksum = checksum_match.group(1) if checksum_match else "unknown"

            entries.append({
                "dataset_id": dataset_id,
                "file_count": file_count,
                "file_size": size,
                "md5_checksum": checksum
            })
    return entries

def parse_download_log(log_path):
    # Parse log lines for size/progress info (example for last line)
    total_size = "unknown"
    last_line = ""
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    # Try to extract size from last line
    size_match = re.search(r'\s([\d\.]+[GMK])', last_line)
    if size_match:
        total_size = size_match.group(1)
    return {"download_log_summary": last_line, "downloaded_size": total_size}

def combine_and_save_csv(corpora_tsv, log_file, output_csv):
    corpora_data = parse_corpora_tsv(corpora_tsv)
    log_data = parse_download_log(log_file)

    # For now, just one dataset, add log info to first entry
    if corpora_data:
        corpora_data[0].update(log_data)

    # Write CSV
    fieldnames = list(corpora_data[0].keys())
    with open(output_csv, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        for entry in corpora_data:
            writer.writerow(entry)
    print(f"✅ Combined CSV saved to {output_csv}")

# Usage
corpora_tsv_path = "/vol/bigdata3/corpora3/MyST/ldc_downloader/corpora.tsv"
log_path = "/vol/bigdata3/corpora3/MyST/ldc_downloader/LDC2021S05__MyST_Children's_Conversational_Speech__myst_child_conv_speech_LDC2021S05.log"
output_csv_path = "/vol/tensusers6/rchissich/ASR/MyST"

combine_and_save_csv(corpora_tsv_path, log_path, output_csv_path)
