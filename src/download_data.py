"""
Script to download and extract the Google Speech Commands dataset v2.
"""
import os
import tarfile
import urllib.request
from tqdm import tqdm

from config import DATASET_URL, RAW_DATA_DIR, DATA_DIR


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dataset(url=DATASET_URL, dest_dir=DATA_DIR):
    """
    Download the Google Speech Commands dataset v2.
    
    Args:
        url: URL to download from
        dest_dir: Directory to save the downloaded file
        
    Returns:
        Path to the downloaded tar.gz file
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    filename = os.path.basename(url)
    filepath = os.path.join(dest_dir, filename)
    
    if os.path.exists(filepath):
        print(f"Dataset already downloaded: {filepath}")
        return filepath
    
    print(f"Downloading dataset from {url}...")
    print(f"This may take a while (~2.3GB)...")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as pbar:
        urllib.request.urlretrieve(url, filepath, reporthook=pbar.update_to)
    
    print(f"Download complete: {filepath}")
    return filepath


def extract_dataset(tar_path, extract_dir=RAW_DATA_DIR):
    """
    Extract the downloaded tar.gz file.
    
    Args:
        tar_path: Path to the tar.gz file
        extract_dir: Directory to extract to
    """
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
        print(f"Dataset already extracted: {extract_dir}")
        return
    
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"Extracting dataset to {extract_dir}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, extract_dir)
    
    print("Extraction complete!")


def main():
    """Download and extract the dataset."""
    # Download
    tar_path = download_dataset()
    
    # Extract
    extract_dataset(tar_path)
    
    print("\nDataset is ready!")
    print(f"Location: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
