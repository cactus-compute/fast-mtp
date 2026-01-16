import os
import argparse
import urllib.request
from datasets import load_dataset

def save_split(dataset, split_name, out_dir, text_key):
    split = dataset[split_name]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for ex in split:
            text = ex[text_key]
            if text is None:
                continue
            text = str(text).strip()
            if text == "":
                continue
            f.write(text + "\n")

def download_ptb(base_dir):
    out_dir = os.path.join(base_dir, "ptb")
    os.makedirs(out_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
    files = {
        "train": "ptb.train.txt",
        "validation": "ptb.valid.txt",
        "test": "ptb.test.txt",
    }
    for split, filename in files.items():
        url = base_url + filename
        dest = os.path.join(out_dir, f"{split}.txt")
        print("downloading", url, "to", dest)
        urllib.request.urlretrieve(url, dest)

def download_wikitext2(base_dir):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    out_dir = os.path.join(base_dir, "wikitext2")
    save_split(ds, "train", out_dir, "text")
    save_split(ds, "validation", out_dir, "text")
    save_split(ds, "test", out_dir, "text")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptb", action="store_true")
    parser.add_argument("--wikitext2", action="store_true")
    args = parser.parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.ptb and not args.wikitext2:
        args.ptb = True
        args.wikitext2 = True
    if args.ptb:
        download_ptb(base_dir)
    if args.wikitext2:
        download_wikitext2(base_dir)

if __name__ == "__main__":
    main()
