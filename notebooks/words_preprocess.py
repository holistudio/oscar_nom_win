import os
import re
import time
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- One-time NLTK downloads ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# --- Config ---
processed_dir = os.path.join('..', 'data', 'processed')
df_train = pd.read_parquet(os.path.join(processed_dir, 'train_clean.parquet'))

# --- Stopword list ---
# Start with NLTK base, then punch holes for meaningful words
base_stopwords = set(stopwords.words('english'))

ALLOWLIST = {
    # Gender pronouns
    'he', 'she', 'they', 'him', 'her', 'his', 'hers', 'their', 'theirs',
    'himself', 'herself', 'themselves',
    # Modal verbs
    'must', 'should', 'would', 'could', 'might', 'may', 'will', 'shall',
    "won't", "can't", "shouldn't", "wouldn't", "couldn't", "mustn't",
    # Emotionally loaded function words
    'never', 'always', 'every', 'nothing', 'everything', 'nobody', 'no',
    'not', "n't",
}

STOPWORDS = base_stopwords - ALLOWLIST

# --- Lemmatizer ---
lemmatizer = WordNetLemmatizer()


def preprocess_script(text: str) -> list[str]:
    """
    Takes a script_clean string and returns a list of lemmatized tokens.
    - Lowercases
    - Strips punctuation (keeps apostrophes for contractions)
    - Removes stopwords (minus allowlist)
    - Lemmatizes
    """
    # Lowercase
    text = text.lower()

    # Normalize contractions apostrophe style (curly -> straight)
    text = text.replace('\u2019', "'").replace('\u2018', "'")

    # Keep only alphabetic characters and apostrophes (for contractions)
    # Then split on whitespace
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)

    # Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 1
    ]

    return tokens


# --- Main loop ---
results = []  # will hold dicts of {imdb_id, nominated, winner, tokens}
elapsed_times = []

print("\nStarting train dataset word pre-processing...")

for idx, row in df_train.iterrows():
    t_start = time.perf_counter()

    script_text = row['script_clean']
    tokens = preprocess_script(script_text)

    t_elapsed = time.perf_counter() - t_start
    elapsed_times.append(t_elapsed)

    results.append({
        'imdb_id':    row['imdb_id'],
        'movie_name': row['movie_name'],
        'nominated':  row['nominated'],
        'winner':     row['winner'],
        'tokens':     tokens,
        'token_count': len(tokens),
    })

    avg_t = sum(elapsed_times) / len(elapsed_times)
    n_processed = len(elapsed_times)
    n_total = len(df_train)
    eta_seconds = avg_t * (n_total - n_processed)

    print(f"[{idx}] {row['movie_name']}")
    print(f"  nominated={row['nominated']}  winner={row['winner']}")
    print(f"  token_count={len(tokens)}")
    print(f"  first 10 tokens: {tokens[:10]}")
    print(f"  time this script : {t_elapsed:.2f}s")
    print(f"  avg time/script  : {avg_t:.2f}s  ({n_processed}/{n_total} processed)")
    print(f"  ETA full dataset : {eta_seconds/60:.1f} min at this rate")
    print()

    # Break after first sample for runtime assessment
    break

print("Done. Extend the loop by removing the break.")