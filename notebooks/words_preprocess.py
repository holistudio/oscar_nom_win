import os
import re
import time
import pickle
import datetime
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_words
from nltk.stem import WordNetLemmatizer
from wordfreq import word_frequency

# --- One-time NLTK downloads ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('words', quiet=True)

# --- Config ---
processed_dir = os.path.join('..', 'data', 'processed')

df_train = pd.read_parquet(os.path.join(processed_dir, 'train_clean.parquet'))
df_val   = pd.read_parquet(os.path.join(processed_dir, 'val_clean.parquet'))
df_test  = pd.read_parquet(os.path.join(processed_dir, 'test_clean.parquet'))

SPLITS = [
    ('train', df_train),
    ('val',   df_val),
    ('test',  df_test),
]

N_TOTAL_GLOBAL = sum(len(df) for _, df in SPLITS)

# --- Stopword list ---
# Start with NLTK base, then punch holes for meaningful words
base_stopwords = set(stopwords.words('english'))

ALLOWLIST = {
    # Gender pronouns
    # 'he', 'she', 'they', 'him', 'her', 'his', 'hers', 'their', 'theirs',
    # 'himself', 'herself', 'themselves',
    # Modal verbs
    'must', 'should', 'would', 'could', 'might', 'may', 'will', 'shall',
    "won't", "can't", "shouldn't", "wouldn't", "couldn't", "mustn't",
    # Emotionally loaded function words
    'never', 'always', 'every', 'nothing', 'everything', 'nobody', 'no',
    'not', #"n't",
}

# --- Screenplay-specific stopwords ---
# Words that dominate every screenplay regardless of quality or genre.
# Grouped by category for maintainability.
SCREENPLAY_STOPWORDS = {
    # Format / scene heading artifacts
    'int', 'ext', 'cont', 'contd', 'vo', 'os', 'oc', 'pod', 'cu', 'pov', 'bos',
    'establishing', 'intercut', 'smash', 'fade', 'cut', 'dissolve', 'montage',
    'title', 'card', 'subtitle', 'superimpose', 'super', 'camera', 'angle', 'shot',

    # Generic high-frequency action verbs (appear in virtually every script)
    'look', 'see', 'come', 'go', 'take', 'walk', 'turn', 'move',
    'stand', 'sit', 'open', 'close', 'pull', 'push', 'reach', 'grab',
    'hold', 'put', 'step', 'stop', 'start', 'run', 'hear', 'watch',
    'say', 'tell', 'think', 'know', 'want', 'need', 'try', 'get',
    'give', 'make', 'find', 'keep', 'let', 'enter', 'exit',
    'cross', 'head', 'hand', 'pass', 'beat', 'stare',

    # Generic spatial / positional words
    'back', 'away', 'around', 'toward', 'inside', 'outside', 'front',
    'behind', 'across', 'over', 'down', 'near', 'side',

    # Generic character placeholders and titles
    'man', 'woman', 'girl', 'boy', 'guy', 'kid', 'mr', 'mrs', 'ms', 'dr',
    'one', 'two', 'three', 'young', 'old', 'new',

    # Filler / acknowledgment words
    'yeah', 'okay', 'ok', 'oh', 'hey', 'uh', 'um', 'ah', 'well',
    'right', 'sure', 'got', 'just', 'like', 'really', 'good',
    'still', 'even', 'also', 

    # Common generic nouns with no discriminative signal
    'room', 'door', 'house', 'time', 'day', 'night', 'moment', 'place',
    'window', 'table', 'floor', 'wall', 'street', 'car', 'phone',
    'face', 'eye', 'voice', 'thing', 'something', 'anything', 'way',
}

custom_stopwords = {'yes', 'as', 'jack', 'mark', 'bill', 'joe', 'sam', 
                    'mary', 'er', 'view', 'th', 'per', 'pause', 'en', 'al',
                    'ho', 'ya', 'e'}

STOPWORDS = (base_stopwords - ALLOWLIST) | SCREENPLAY_STOPWORDS | custom_stopwords

# --- Lemmatizer ---
ENGLISH_WORDS = set(nltk_words.words())
lemmatizer = WordNetLemmatizer()


def preprocess_script(text: str) -> list[str]:
    """
    Takes a script_clean string and returns a list of lemmatized tokens.
    - Lowercases
    - Strips punctuation (keeps apostrophes for contractions)
    - Removes stopwords (minus allowlist) + screenplay-specific stopwords
    - Lemmatizes
    """
    # Lowercase
    text = text.lower()

    # Normalize contractions apostrophe style (curly -> straight)
    text = text.replace('\u2019', "'").replace('\u2018', "'")

    # Strip contractions: "don't" → "dont", "can't" → "cant", etc.
    text = re.sub(r"'", "", text)
    

    # Keep only alphabetic characters and apostrophes (for contractions)
    # Then split on whitespace
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    # Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS
        and len(tok) > 1
        and tok in ENGLISH_WORDS
        and word_frequency(tok, 'en') >= 1e-6
    ]

    return tokens


# --- Main loop ---
elapsed_times = []   # tracks time across ALL splits for global ETA
n_processed_global = 0

print(f"\nStarting word pre-processing across all splits ({N_TOTAL_GLOBAL} total scripts)...")

for split_name, df in SPLITS:
    print(f"\n{'='*60}")
    print(f"  Split: {split_name.upper()}  ({len(df)} scripts)")
    print(f"{'='*60}")

    results = []

    for idx, row in df.iterrows():
        t_start = time.perf_counter()

        script_text = row['script_clean']
        tokens = preprocess_script(script_text)

        t_elapsed = time.perf_counter() - t_start
        elapsed_times.append(t_elapsed)
        n_processed_global += 1

        results.append({
            'imdb_id':     row['imdb_id'],
            'movie_name':  row['movie_name'],
            'nominated':   row['nominated'],
            'winner':      row['winner'],
            'tokens':      tokens,
            'token_count': len(tokens),
        })

        avg_t = sum(elapsed_times) / len(elapsed_times)
        n_remaining = N_TOTAL_GLOBAL - n_processed_global
        eta_seconds = avg_t * n_remaining
        eta_timestamp = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)

        if idx % 100 == 0:
            print(f"[{split_name}:{idx}] {row['movie_name']}")
            print(f"  nominated={row['nominated']}  winner={row['winner']}")
            print(f"  token_count={len(tokens)}")
            print(f"  first 10 tokens: {tokens[:10]}")
            print(f"  time this script : {t_elapsed:.2f}s")
            print(f"  avg time/script  : {avg_t:.2f}s  ({n_processed_global}/{N_TOTAL_GLOBAL} processed globally)")
            print(f"  ETA full dataset : {eta_seconds/60:.1f} min  (done ~{eta_timestamp.strftime('%H:%M:%S')})")
            print()

    # --- Save split results ---
    df_results = pd.DataFrame(results)

    out_path = os.path.join(processed_dir, f'words_{split_name}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(df_results, f)

    print(f"  Saved {len(df_results)} records → {out_path}")

print("\nDone. All splits processed and saved.")