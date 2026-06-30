"""Download the Last.fm HetRec 2011 dataset into data/raw/.

Dataset: hetrec2011-lastfm-2k (GroupLens, HetRec 2011 workshop).
Source:  https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
License: free for research/academic use. Cite the HetRec 2011 dataset.

Files extracted (tab-separated):
    user_artists.dat        userID  artistID  weight(=playcount)
    artists.dat             id  name  url  pictureURL
    tags.dat                tagID  tagValue
    user_taggedartists.dat  userID  artistID  tagID  day  month  year
    user_friends.dat        userID  friendID

Usage:
    python scripts/download_data.py
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    print(f"Downloaded {len(blob)/1e6:.1f} MB. Extracting into {RAW_DIR} ...")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            target = RAW_DIR / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                dst.write(src.read())
            print(f"  - {target.name}")
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
