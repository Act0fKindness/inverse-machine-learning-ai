import os
import subprocess
import argparse
from huggingface_hub import HfApi
from tqdm import tqdm

def list_remote_files(dataset_name="ZechengLi19/CSL-News"):
    api = HfApi()
    info = api.dataset_info(dataset_name, files_metadata=True)
    return sorted([f.rfilename for f in info.siblings if f.rfilename.endswith(".zip")])

def download_and_extract(remote_path, output_zip_folder, extracted_folder, force=False):
    filename = os.path.basename(remote_path)
    local_path = os.path.join(output_zip_folder, filename)
    extract_path = os.path.join(extracted_folder, filename.replace(".zip", ""))

    if not force:
        if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
            print(f"[✓] Already extracted: {filename}")
            return

    # Download
    url = f"https://huggingface.co/datasets/ZechengLi19/CSL-News/resolve/main/{remote_path}"
    print(f"[↓] Downloading: {filename}")
    result = subprocess.run(["wget", "-q", "-O", local_path, url])
    if result.returncode != 0:
        print(f"[✗] Failed to download: {filename}")
        return

    # Remove existing extract dir to avoid partial contents
    if os.path.exists(extract_path):
        subprocess.run(["rm", "-rf", extract_path])
    os.makedirs(extract_path, exist_ok=True)

    # Extract
    print(f"[⤵] Extracting: {filename}")
    result = subprocess.run(["unzip", "-q", local_path, "-d", extract_path])
    if result.returncode != 0:
        print(f"[✗] Failed to extract: {filename}")
        return

    print(f"[✓] Done: {filename}")

def main(output_directory, restart_from):
    output_zip_folder = os.path.join(output_directory, "RGB_download")
    extracted_folder = os.path.join(output_directory, "rgb_format")
    os.makedirs(output_zip_folder, exist_ok=True)
    os.makedirs(extracted_folder, exist_ok=True)

    print("[📡] Fetching file list from HuggingFace...")
    all_files = list_remote_files()

    # Locate start index
    start_idx = next((i for i, f in enumerate(all_files) if f == restart_from), None)
    if start_idx is None:
        print(f"[!] Could not find '{restart_from}' in the list of archive files.")
        return

    files_to_process = all_files[start_idx:]

    print(f"[🧾] {len(files_to_process)} files to process starting from {restart_from}.")
    for i, zip_file in enumerate(tqdm(files_to_process)):
        force = (i == 0)  # Force redownload for the first file only (archive_341.zip)
        download_and_extract(zip_file, output_zip_folder, extracted_folder, force=force)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", required=True, help="Path to dataset/CSL_News")
    parser.add_argument("--restart_from", required=True, help="Exact filename to restart from (e.g. archive_341.zip)")
    args = parser.parse_args()
    main(args.output_directory, args.restart_from)

