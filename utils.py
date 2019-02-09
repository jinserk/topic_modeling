import sys
from pathlib import Path
import logging


logging.basicConfig(format="%(asctime)s [%(levelname)-5s] %(message)s", level=logging.INFO)


def check_dir(dir_path):
    dir_path = Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir(mode=0o755, parents=True, exist_ok=True)

def reporthook(count, block_size, total_size):
    try:
        progress_size = count * block_size
        if progress_size < total_size:
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r... {percent} %, {progress_size / (1024 * 1024):.3f} MB")
            sys.stdout.flush()
        else:
            print("\r... done.")
    except:
        print("\r... error to download, aborted.")

def download_file(base_url, target_file, target_dir):
    check_dir(target_dir)
    target_file_path = Path(target_dir, target_file)
    if not target_file_path.exists():
        try:
            print(f"downloading {target_file} ...")
            import urllib.parse
            import urllib.request
            full_url = urllib.parse.urljoin(base_url, target_file)
            urllib.request.urlretrieve(full_url, str(target_file_path), reporthook=reporthook)
        except:
            logging.error("download failed.")
            sys.exit(1)


if __name__ == "__main__":
    #base_url = "http://download.wikimedia.org/enwiki/latest/"
    #base_url = "http://dumps.wikimedia.org/simplewiki/latest/"
    #bin_file = "simplewiki-latest-pages-articles.xml.bz2"

    target_dir = "./data"
    download_file(target_dir, bin_file)
