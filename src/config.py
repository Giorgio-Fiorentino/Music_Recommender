"""Configuration: paths, column names, and project constants.

This project uses the **Last.fm HetRec 2011** dataset (music track). Last.fm gives
us *implicit feedback* (how many times a user played an artist), not explicit
1-5 star ratings. To keep every algorithm dataset-agnostic we expose **generic**
column names (USER_COL / ITEM_COL / WEIGHT_COL) that the loaders map onto the
raw Last.fm columns. If you ever swapped in MovieLens, you would only change this
file, not the model code.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --- Last.fm HetRec 2011 raw files (tab-separated, latin-1 encoded) ----------
RATINGS_PATH = RAW_DATA_DIR / "user_artists.dat"          # userID, artistID, weight
ITEMS_PATH = RAW_DATA_DIR / "artists.dat"                 # id, name, url, pictureURL
TAGS_PATH = RAW_DATA_DIR / "tags.dat"                     # tagID, tagValue
USER_TAGGED_ARTISTS_PATH = RAW_DATA_DIR / "user_taggedartists.dat"  # userID, artistID, tagID, ...
USER_FRIENDS_PATH = RAW_DATA_DIR / "user_friends.dat"    # userID, friendID

DATA_ENCODING = "utf-8"  # the .dat data files are UTF-8 (only readme.txt is latin-1)

# --- Generic column names used throughout the codebase -----------------------
# These are the names the rest of the code relies on. load_ratings() renames the
# raw Last.fm columns to these, so models never hard-code "artistID" etc.
USER_COL = "userId"
ITEM_COL = "itemId"        # an "item" here is an artist
WEIGHT_COL = "weight"      # log1p(play count); the implicit "rating" signal
RAW_WEIGHT_COL = "plays"   # the original integer play count (kept for EDA)
RATING_COL = WEIGHT_COL    # alias so placeholder code expecting RATING_COL works

# Item metadata columns (after renaming).
ITEM_ID_COL = "itemId"
TITLE_COL = "name"         # artist name (the human-readable label)
GENRES_COL = "tags"        # space-joined tag string used as content features

# --- Preprocessing knobs -----------------------------------------------------
MIN_USER_INTERACTIONS = 5  # drop users with fewer than this many artists
MIN_ITEM_INTERACTIONS = 5  # drop artists listened to by fewer than this many users
TEST_SIZE = 0.2            # fraction of each user's artists held out for testing

# --- Recommendation / evaluation defaults ------------------------------------
TOP_K = 10
RANDOM_STATE = 42
