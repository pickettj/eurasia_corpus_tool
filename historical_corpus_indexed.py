#!/usr/bin/env python3
"""
Historical Corpus Query Functions (Indexed Corpus)
Query and analyze the indexed Eurasia corpus with frequency data and n-grams
"""

import pandas as pd
import os
import re
from datetime import datetime

# Set home directory path
hdir = os.path.expanduser('~')

# Define data paths - these point to the OUTPUT of historical_corpus_indexer.py
indexed_corpus_dir = os.path.join(hdir, 'Dropbox/Active_Directories/Digital_Humanities/Datasets/eurasia_indexed_corpus')
indexed_corpus_path = os.path.join(indexed_corpus_dir, 'eurasia_corpus_indexed.csv')

freq_dict_dir = os.path.join(hdir, 'Dropbox/Active_Directories/Digital_Humanities/Datasets/eurasia_corpus_frequency_dictionary')
freq_dict_path = os.path.join(freq_dict_dir, 'eurasia_corpus_frequency_dictionary.csv')

# N-gram file paths (in same directory as frequency dictionary)
ngram_paths = {
    2: os.path.join(freq_dict_dir, 'eurasia_corpus_bigrams.csv'),
    3: os.path.join(freq_dict_dir, 'eurasia_corpus_trigrams.csv'),
    4: os.path.join(freq_dict_dir, 'eurasia_corpus_4grams.csv'),
    5: os.path.join(freq_dict_dir, 'eurasia_corpus_5grams.csv'),
    6: os.path.join(freq_dict_dir, 'eurasia_corpus_6grams.csv')
}

# Inbox for reports
inbox_path = os.path.join(hdir, 'Dropbox/Active_Directories/Inbox')


"""
Library Information Display
"""

print("\n" + "=" * 70)
print("📚 HISTORICAL CORPUS QUERY LIBRARY (Indexed Corpus)")
print("=" * 70)

print("\n📊 DATA SOURCES:")
print("   By default, this library uses the LATEST datasets produced by")
print("   historical_corpus_indexer.py:")
print(f"\n   • Indexed Corpus: {indexed_corpus_path}")
print(f"   • Frequency Dictionary: {freq_dict_path}")
print(f"   • N-grams: {freq_dict_dir}/eurasia_corpus_[bigrams|trigrams|etc].csv")

print("\n🔄 UPDATING DATA:")
print("   If you need to rebuild the corpus/frequency/n-grams from source,")
print("   use the update_data() function:")
print("\n   >>> update_data()                    # Rebuild everything")
print("   >>> update_data(rebuild_corpus=False) # Only rebuild freq dict & n-grams")
print("   >>> update_data(max_ngrams=4)         # Rebuild with only 2-4 grams")

print("\n⚠️  NOTE: By default, this library does NOT import historical_corpus_indexer.py")
print("   This keeps things fast - we just read the CSV files that were already created.")
print("   Only use update_data() when you need fresh data from source documents.")

print("\n" + "=" * 70)
print("🔧 AVAILABLE FUNCTIONS")
print("=" * 70)

print("\n📥 Data Loading:")
print("   • load_corpus() - Load indexed corpus")
print("   • load_frequency_dict() - Load frequency dictionary")
print("   • load_ngrams(n) - Load specific n-gram file (e.g., n=2 for bigrams)")
print("   • update_data() - Rebuild all data from source (imports indexer library)")

print("\n🔍 Query Functions:")
print("   • kwic(search_term, window_size=5, max_results=100, top_matches=5)")
print("     → Keywords in context search (shows top N by frequency)")
print("   • freqdic(search_term, min_ngram_freq=2)")
print("     → Frequency info + phrases containing the term")

print("\n" + "=" * 70)
print("💡 QUICK START")
print("=" * 70)
print("   # Load data (fast - just reads CSV):")
print("   df = load_corpus()")
print("   freq = load_frequency_dict()")
print("   bigrams = load_ngrams(2)")
print("\n   # Search and analyze:")
print("   results = kwic('باج')        # Keywords in context")
print("   info = freqdic('باج')         # Frequency + phrases")
print("\n   # Update data from source (slow - rebuilds everything):")
print("   update_data()")
print("=" * 70 + "\n")


"""
Data Loading Functions
"""

def load_corpus(nrows=None):
    """
    Load the indexed corpus from CSV.
    
    Args:
        nrows (int, optional): Number of rows to load (for testing). None = load all.
    
    Returns:
        pd.DataFrame: Indexed corpus with columns:
            - document: Source filename
            - position: Token position in document
            - token_original: Original token (before cleaning)
            - token_clean: Cleaned token (Arabic script only)
            - token_id: Global unique ID
            - bib_uid: Bibliography UID
            - Language, Status, Tags, Type: Metadata from bibliography
            - Repository_Acronym: Repository info
    
    Example:
        >>> df = load_corpus()
        📚 Loading indexed corpus...
        ✅ Loaded 1,234,567 tokens from 1,067 documents
    """
    print("\n📚 Loading indexed corpus...")
    
    if not os.path.exists(indexed_corpus_path):
        print(f"❌ Error: Corpus file not found at {indexed_corpus_path}")
        print("   Run update_data() to build the corpus first.")
        return None
    
    df = pd.read_csv(indexed_corpus_path, nrows=nrows)
    
    num_tokens = len(df)
    num_docs = df['document'].nunique()
    
    print(f"✅ Loaded {num_tokens:,} tokens from {num_docs:,} documents")
    
    return df


def load_frequency_dict():
    """
    Load the frequency dictionary from CSV.
    
    Returns:
        pd.DataFrame: Frequency dictionary with columns:
            - UID: Unique identifier
            - term: The word
            - frequency: Number of occurrences
            - percentile: Ranking (lower = more common)
            - bib_uids: Bibliography UIDs where term appears
    
    Example:
        >>> freq = load_frequency_dict()
        📊 Loading frequency dictionary...
        ✅ Loaded 45,678 unique words
    """
    print("\n📊 Loading frequency dictionary...")
    
    if not os.path.exists(freq_dict_path):
        print(f"❌ Error: Frequency dictionary not found at {freq_dict_path}")
        print("   Run update_data() to build it first.")
        return None
    
    df = pd.read_csv(freq_dict_path, dtype={'bib_uids': str})
    
    num_words = len(df)
    total_occurrences = df['frequency'].sum()
    
    print(f"✅ Loaded {num_words:,} unique words ({total_occurrences:,} total occurrences)")
    
    return df


def load_ngrams(n):
    """
    Load a specific n-gram file from CSV.
    
    Args:
        n (int): N-gram size (2=bigrams, 3=trigrams, etc.)
    
    Returns:
        pd.DataFrame: N-gram data with columns:
            - UID: Unique identifier
            - ngram: The n-gram (space-separated words)
            - frequency: Number of occurrences
            - percentile: Ranking
            - bib_uids: Bibliography UIDs
    
    Example:
        >>> bigrams = load_ngrams(2)
        📊 Loading 2-grams (bigrams)...
        ✅ Loaded 123,456 unique bigrams
        
        >>> trigrams = load_ngrams(3)
        📊 Loading 3-grams (trigrams)...
        ✅ Loaded 234,567 unique trigrams
    """
    ngram_names = {2: 'bigrams', 3: 'trigrams', 4: '4-grams', 5: '5-grams', 6: '6-grams'}
    ngram_name = ngram_names.get(n, f'{n}-grams')
    
    print(f"\n📊 Loading {n}-grams ({ngram_name})...")
    
    if n not in ngram_paths:
        print(f"❌ Error: N-gram size {n} not supported (only 2-6)")
        return None
    
    ngram_path = ngram_paths[n]
    
    if not os.path.exists(ngram_path):
        print(f"❌ Error: {ngram_name} file not found at {ngram_path}")
        print("   Run update_data() to build it first.")
        return None
    
    df = pd.read_csv(ngram_path, dtype={'bib_uids': str})
    
    num_ngrams = len(df)
    total_occurrences = df['frequency'].sum()
    
    print(f"✅ Loaded {num_ngrams:,} unique {ngram_name} ({total_occurrences:,} total occurrences)")
    
    return df


def update_data(rebuild_corpus=True, max_ngrams=6):
    """
    Rebuild all data from source documents using historical_corpus_indexer.py
    
    ⚠️  WARNING: This can take 5-10 minutes for the full corpus!
    Only use this when you need fresh data from source XML files.
    
    Args:
        rebuild_corpus (bool): If True, rebuilds indexed corpus from scratch.
                              If False, uses existing corpus and only rebuilds freq/ngrams.
        max_ngrams (int): Maximum n-gram size to generate (default: 6)
    
    Returns:
        dict: Paths to all generated files
    
    Example:
        >>> update_data()  # Full rebuild (slow)
        🔄 UPDATING ALL DATA FROM SOURCE
        [imports historical_corpus_indexer and runs build pipeline]
        
        >>> update_data(rebuild_corpus=False)  # Just freq dict & n-grams (faster)
        🔄 UPDATING FREQUENCY DICTIONARY & N-GRAMS
        [uses existing corpus, rebuilds freq/ngrams only]
    """
    print("\n" + "=" * 70)
    if rebuild_corpus:
        print("🔄 UPDATING ALL DATA FROM SOURCE")
        print("   This will rebuild: Corpus → Frequency Dict → N-grams")
    else:
        print("🔄 UPDATING FREQUENCY DICTIONARY & N-GRAMS")
        print("   This will rebuild: Frequency Dict → N-grams (using existing corpus)")
    print("=" * 70)
    
    # Import the indexer library (only when needed)
    print("\n📦 Importing historical_corpus_indexer library...")
    try:
        import sys
        indexer_path = os.path.join(hdir, 'Projects/eurasia_corpus_tool')
        if indexer_path not in sys.path:
            sys.path.insert(0, indexer_path)
        
        import historical_corpus_indexer as hci
        print("✅ Indexer library loaded")
    except ImportError as e:
        print(f"❌ Error importing historical_corpus_indexer: {e}")
        print("   Make sure historical_corpus_indexer.py is in:")
        print(f"   {indexer_path}")
        return None
    
    results = {}
    
    # Step 1: Build/load corpus
    if rebuild_corpus:
        print("\n" + "=" * 70)
        print("STEP 1/3: BUILDING INDEXED CORPUS")
        print("=" * 70)
        df = hci.build_and_enrich()
        results['corpus'] = indexed_corpus_path
    else:
        print("\n" + "=" * 70)
        print("STEP 1/3: LOADING EXISTING CORPUS")
        print("=" * 70)
        df = load_corpus()
        if df is None:
            print("❌ No existing corpus found - you must rebuild from source")
            print("   Try: update_data(rebuild_corpus=True)")
            return None
    
    # Step 2: Generate frequency dictionary
    print("\n" + "=" * 70)
    print("STEP 2/3: GENERATING FREQUENCY DICTIONARY")
    print("=" * 70)
    freq_df = hci.create_frequency_dictionary(df)
    results['frequency_dict'] = freq_dict_path
    
    # Step 3: Generate n-grams
    print("\n" + "=" * 70)
    print(f"STEP 3/3: GENERATING N-GRAMS (2-{max_ngrams})")
    print("=" * 70)
    ngram_dfs = hci.create_ngrams(df, max_n=max_ngrams)
    
    # Add n-gram paths to results
    for n in range(2, max_ngrams + 1):
        if n in ngram_paths:
            results[f'{n}grams'] = ngram_paths[n]
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ DATA UPDATE COMPLETE")
    print("=" * 70)
    print("\n📁 Files updated:")
    for name, path in results.items():
        print(f"   • {name}: {path}")
    print("\n💡 You can now use load_corpus(), load_frequency_dict(), load_ngrams()")
    print("=" * 70 + "\n")
    
    return results


"""
Query Functions (Coming Soon)
"""

# TODO: Add search and analysis functions here
# Example functions to implement:
# - search_term(term) - Search for a term in corpus
# - term_frequency(term) - Get frequency stats for a term
# - find_ngrams_with_term(term, n) - Find n-grams containing a term
# - compare_frequencies(term1, term2) - Compare two terms
# - get_context(term, window_size) - Get surrounding context for term occurrences


"""
KWIC (Keywords in Context) Search
"""

def kwic(search_term, window_size=5, max_results=100, case_sensitive=False, top_matches=5):
    """
    Search for a term and return Keywords in Context (KWIC) results.
    
    When regex matches multiple terms, shows most frequent matches first.
    
    Args:
        search_term (str): Regex pattern to search for
        window_size (int): Number of tokens before/after match (default: 5)
        max_results (int): Maximum context results to display per match (default: 100)
        case_sensitive (bool): Whether search is case-sensitive (default: False)
        top_matches (int): Number of top frequency matches to show in detail (default: 5)
                          Other matches listed by term only
    
    Returns:
        pd.DataFrame: KWIC results with columns:
            - document: Source document
            - bib_uid: Bibliography UID
            - matched_term: Actual matched term (from regex)
            - left_context: Tokens before match
            - match: The matched token(s)
            - right_context: Tokens after match
            - position: Position in document
            - frequency: Frequency of matched term (for sorting)
    
    Example:
        >>> results = kwic('ب.ج')  # Regex matches: باج, بیج, etc.
        
        🔍 Searching for: 'ب.ج'
        ══════════════════════════════════════════════════════════════════════
        
        Found 3 unique matches: باج (1,234×), بیج (45×), بوج (2×)
        Showing top 5 matches in detail (sorted by frequency)
        
        ════════════════════════════════════════════════════════════════════
        📊 MATCH 1/3: 'باج' (1,234 occurrences, Top 5.2% - very common)
        ════════════════════════════════════════════════════════════════════
        
        Found 856 contexts (showing first 100)
        
        📄 ser1234.xml (UID: 1234) - 3 contexts
        ───────────────────────────────────────────────────────────────────
        که در آن زمان ... | **باج** | ... و خراج می‌گرفتند
        از مردم این شهر ... | **باج** | ... می‌ستدند به زور
        ...
        
        ════════════════════════════════════════════════════════════════════
        📊 MATCH 2/3: 'بیج' (45 occurrences, Top 23.4% - moderate)
        ════════════════════════════════════════════════════════════════════
        ...
        
        ══════════════════════════════════════════════════════════════════
        📋 OTHER MATCHES (not shown in detail):
           • بوج (2 occurrences, Top 87.3%)
        ══════════════════════════════════════════════════════════════════
    
    Notes:
        - Uses token_clean for matching (normalized forms)
        - Results sorted by frequency (most common first)
        - Only top N matches shown in detail (configurable)
        - Frequency data from frequency dictionary
    """
    print(f"\n🔍 Searching for: '{search_term}'")
    print("=" * 70)
    
    # Step 1: Process search (separated for reuse in complex queries)
    results_df = _kwic_search(search_term, window_size, case_sensitive)
    
    if results_df is None or len(results_df) == 0:
        print("❌ No matches found\n")
        return None
    
    # Step 2: Enrich with frequency data and sort
    results_df = _kwic_add_frequency(results_df)
    
    # Step 3: Display results (frequency-sorted)
    _kwic_display(results_df, search_term, max_results, top_matches)
    
    return results_df


def _kwic_add_frequency(results_df):
    """
    Add frequency information to KWIC results for sorting.
    
    Args:
        results_df (pd.DataFrame): Results from _kwic_search()
    
    Returns:
        pd.DataFrame: Results with added frequency, rate_per_10k, and tier columns
    """
    # Load frequency dictionary
    freq_df = load_frequency_dict()
    
    if freq_df is None:
        # No frequency data available - add dummy values
        results_df['frequency'] = 0
        results_df['rate_per_10k'] = 0.0
        results_df['tier'] = 'unknown'
        return results_df
    
    # Match each result with frequency data
    # Create lookup dict for fast access
    freq_lookup = freq_df.set_index('term')[['frequency', 'rate_per_10k', 'tier']].to_dict('index')
    
    # Add frequency, rate, and tier to each result
    results_df['frequency'] = results_df['match'].map(
        lambda x: freq_lookup.get(x, {}).get('frequency', 0)
    )
    results_df['rate_per_10k'] = results_df['match'].map(
        lambda x: freq_lookup.get(x, {}).get('rate_per_10k', 0.0)
    )
    results_df['tier'] = results_df['match'].map(
        lambda x: freq_lookup.get(x, {}).get('tier', 'unknown')
    )
    
    # Sort by frequency (descending) - most common first
    results_df = results_df.sort_values('frequency', ascending=False)
    
    return results_df


def _kwic_search(search_term, window_size=5, case_sensitive=False):
    """
    Internal function: Search corpus and extract KWIC contexts.
    
    Separated from display logic for use in complex queries.
    
    Args:
        search_term (str): Regex pattern to search for
        window_size (int): Number of tokens before/after match
        case_sensitive (bool): Whether search is case-sensitive
    
    Returns:
        pd.DataFrame: KWIC results (see kwic() for column details)
    """
    # Load corpus
    df = load_corpus()
    if df is None:
        return None
    
    # Set regex flags
    flags = 0 if case_sensitive else re.IGNORECASE
    
    # Find matching tokens
    # Use token_clean for matching (normalized forms)
    mask = df['token_clean'].str.contains(search_term, regex=True, na=False, flags=flags)
    matches_df = df[mask].copy()
    
    if len(matches_df) == 0:
        return None
    
    # For each match, extract context window
    results = []
    
    for idx, row in matches_df.iterrows():
        doc_name = row['document']
        position = row['position']
        bib_uid = row['bib_uid']
        
        # Get all tokens from this document
        doc_tokens = df[df['document'] == doc_name].sort_values('position')
        
        # Calculate context window positions
        start_pos = max(0, position - window_size)
        end_pos = position + window_size + 1
        
        # Extract context tokens
        left_tokens = doc_tokens[
            (doc_tokens['position'] >= start_pos) & 
            (doc_tokens['position'] < position)
        ]['token_clean'].tolist()
        
        match_token = row['token_clean']
        
        right_tokens = doc_tokens[
            (doc_tokens['position'] > position) & 
            (doc_tokens['position'] <= end_pos)
        ]['token_clean'].tolist()
        
        results.append({
            'document': doc_name,
            'bib_uid': bib_uid,
            'position': position,
            'left_context': ' '.join(left_tokens),
            'match': match_token,
            'right_context': ' '.join(right_tokens)
        })
    
    return pd.DataFrame(results)


def _kwic_display(results_df, search_term, max_results=100, top_matches=5):
    """
    Internal function: Display KWIC results in formatted output.
    
    Shows top N matches (by frequency) in detail, lists others by term only.
    
    Args:
        results_df (pd.DataFrame): Results from _kwic_search() with frequency added
        search_term (str): Original search term (for display)
        max_results (int): Maximum context examples per match
        top_matches (int): Number of matches to show in detail
    """
    total_contexts = len(results_df)
    
    # Get unique matched terms with their frequencies
    match_summary = results_df.groupby('match').agg({
        'frequency': 'first',  # Frequency is same for all instances of a term
        'rate_per_10k': 'first',
        'tier': 'first',
        'match': 'size'  # Count contexts
    }).rename(columns={'match': 'context_count'})
    
    # Sort by frequency (already done in main df, but ensure it here)
    match_summary = match_summary.sort_values('frequency', ascending=False)
    
    num_unique_matches = len(match_summary)
    
    # Display summary
    print(f"\nFound {num_unique_matches} unique match(es):", end='')
    
    # Show frequency summary for all matches
    match_summaries = []
    for term, row in match_summary.iterrows():
        freq = int(row['frequency']) if row['frequency'] > 0 else '?'
        match_summaries.append(f"{term} ({freq:,}×)" if isinstance(freq, int) else f"{term} (?)")
    
    print(" " + ", ".join(match_summaries))
    
    if num_unique_matches > top_matches:
        print(f"Showing top {top_matches} matches in detail (sorted by frequency)")
    
    print()
    
    # Show top N matches in detail
    for i, (term, summary) in enumerate(match_summary.head(top_matches).iterrows(), 1):
        freq = int(summary['frequency']) if summary['frequency'] > 0 else None
        rate = summary['rate_per_10k']
        tier = summary['tier']
        context_count = int(summary['context_count'])
        
        # Tier labels for display
        tier_labels = {
            'extremely_common': 'extremely common',
            'very_common': 'very common',
            'common': 'common',
            'moderate': 'moderate',
            'rare': 'rare',
            'hapax': 'hapax',
            'unknown': 'frequency unknown'
        }
        tier_label = tier_labels.get(tier, tier)
        
        print("=" * 70)
        if freq:
            print(f"📊 MATCH {i}/{num_unique_matches}: '{term}' ({freq:,} occurrences, {rate:.2f} per 10k - {tier_label})")
        else:
            print(f"📊 MATCH {i}/{num_unique_matches}: '{term}' (frequency unknown)")
        print("=" * 70)
        
        # Get contexts for this specific term
        term_contexts = results_df[results_df['match'] == term]
        
        print(f"\nFound {context_count} contexts", end='')
        if context_count > max_results:
            print(f" (showing first {max_results})")
            term_contexts = term_contexts.head(max_results)
        else:
            print()
        
        print()
        
        # Group by document for cleaner display
        for doc_name, group in term_contexts.groupby('document'):
            bib_uid = group['bib_uid'].iloc[0]
            bib_display = f"UID: {int(bib_uid)}" if pd.notna(bib_uid) else "UID: N/A"
            
            print(f"📄 {doc_name} ({bib_display}) - {len(group)} contexts")
            print("─" * 70)
            
            for idx, row in group.iterrows():
                # Format context with ellipsis
                left = ("... " + row['left_context']) if row['left_context'] else ""
                right = (row['right_context'] + " ...") if row['right_context'] else ""
                
                # Highlight match with bold formatting
                print(f"{left} | **{row['match']}** | {right}")
            
            print()
    
    # List remaining matches (not shown in detail)
    if num_unique_matches > top_matches:
        remaining = match_summary.iloc[top_matches:]
        
        print()
        print("=" * 70)
        print("📋 OTHER MATCHES (not shown in detail):")
        
        for term, row in remaining.iterrows():
            freq = int(row['frequency']) if row['frequency'] > 0 else '?'
            percentile = row['percentile']
            context_count = int(row['context_count'])
            
            if isinstance(freq, int):
                print(f"   • {term} ({freq:,} occurrences, Top {percentile:.1f}%, {context_count} contexts)")
            else:
                print(f"   • {term} (frequency unknown, {context_count} contexts)")
        
        print("=" * 70)
    
    print()


"""
Frequency Dictionary Lookup
"""

def freqdic(search_term, min_ngram_freq=2):
    """
    Look up frequency information and phrases containing a term.
    
    Searches:
    1. Frequency dictionary for the exact term
    2. All n-gram files for phrases containing the term
    
    Args:
        search_term (str): Term to look up (exact match in freq dict, regex in n-grams)
        min_ngram_freq (int): Minimum frequency for n-grams to display (default: 2)
                             Ignores hapax n-grams (occurring only once)
    
    Returns:
        dict: Frequency data with keys:
            - term_info: Frequency dictionary entry (if found)
            - bigrams: DataFrame of bigrams containing term
            - trigrams: DataFrame of trigrams containing term
            - etc.
    
    Example:
        >>> info = freqdic('باج')
        
        📊 FREQUENCY LOOKUP: 'باج'
        ══════════════════════════════════════════════════════════════════════
        
        📖 TERM FREQUENCY:
           Word: باج
           Frequency: 1,234 occurrences
           Percentile: Top 5.2% (very common)
           Found in: 45 documents
        
        📝 PHRASES (2+ occurrences only):
        
        BIGRAMS (2-word phrases):
           1. باج و خراج (89 times) - Top 2.1%
           2. باج می‌گیرد (34 times) - Top 5.8%
           ...
        
        TRIGRAMS (3-word phrases):
           1. باج و خراج می (23 times) - Top 3.4%
           ...
    
    Notes:
        - Frequency lookup is exact match
        - N-gram search uses regex (finds term within phrases)
        - Only shows n-grams with frequency >= min_ngram_freq
        - SQL equivalent: JOIN freq_dict with each n-gram table WHERE ngram LIKE '%term%'
    """
    print(f"\n📊 FREQUENCY LOOKUP: '{search_term}'")
    print("=" * 70)
    
    # Step 1: Process lookup (separated for reuse)
    results = _freqdic_lookup(search_term, min_ngram_freq)
    
    # Step 2: Display results
    _freqdic_display(results, search_term, min_ngram_freq)
    
    return results


def _freqdic_lookup(search_term, min_ngram_freq=2):
    """
    Internal function: Look up term in frequency dict and n-grams.
    
    Separated from display logic for use in complex queries.
    
    Args:
        search_term (str): Term to look up
        min_ngram_freq (int): Minimum n-gram frequency
    
    Returns:
        dict: Frequency data (see freqdic() for structure)
    """
    results = {}
    
    # Load frequency dictionary
    freq_df = load_frequency_dict()
    
    if freq_df is not None:
        # Regex match in frequency dictionary (consistent with n-gram search)
        # Case-insensitive search
        mask = freq_df['term'].str.contains(search_term, regex=True, na=False, flags=re.IGNORECASE)
        term_matches = freq_df[mask]
        
        if len(term_matches) > 0:
            # If multiple matches, take the most frequent one as primary
            results['term_info'] = term_matches.nlargest(1, 'frequency').iloc[0].to_dict()
            
            # Also store all matches for potential display
            results['all_term_matches'] = term_matches
        else:
            results['term_info'] = None
            results['all_term_matches'] = None
    else:
        results['term_info'] = None
        results['all_term_matches'] = None
    
    # Search all n-gram files (2-6)
    for n in range(2, 7):
        ngrams_df = load_ngrams(n)
        
        if ngrams_df is not None:
            # Find n-grams containing the term (regex search)
            # Case-insensitive search
            mask = ngrams_df['ngram'].str.contains(search_term, regex=True, na=False, flags=re.IGNORECASE)
            
            # Filter to minimum frequency
            mask = mask & (ngrams_df['frequency'] >= min_ngram_freq)
            
            matches = ngrams_df[mask].copy()
            
            # Sort by frequency (most common first)
            matches = matches.sort_values('frequency', ascending=False)
            
            ngram_key = {2: 'bigrams', 3: 'trigrams', 4: 'fourgrams', 
                        5: 'fivegrams', 6: 'sixgrams'}.get(n, f'{n}grams')
            
            results[ngram_key] = matches if len(matches) > 0 else None
        else:
            ngram_key = {2: 'bigrams', 3: 'trigrams', 4: 'fourgrams', 
                        5: 'fivegrams', 6: 'sixgrams'}.get(n, f'{n}grams')
            results[ngram_key] = None
    
    return results


def _freqdic_display(results, search_term, min_ngram_freq):
    """
    Internal function: Display frequency lookup results.
    
    Separated from lookup logic for use in complex queries.
    
    Args:
        results (dict): Results from _freqdic_lookup()
        search_term (str): Original search term
        min_ngram_freq (int): Minimum frequency used
    """
    # Display term frequency info
    print("\n📖 TERM FREQUENCY:")
    
    if results['term_info']:
        info = results['term_info']
        freq = info['frequency']
        rate = info['rate_per_10k']
        tier = info['tier']
        
        # Tier labels for display
        tier_labels = {
            'extremely_common': 'extremely common',
            'very_common': 'very common',
            'common': 'common',
            'moderate': 'moderate',
            'rare': 'rare',
            'hapax': 'hapax (unique)'
        }
        tier_label = tier_labels.get(tier, tier)
        
        # Count documents
        bib_uids = info.get('bib_uids', '')
        # Handle NaN/float values safely
        if pd.isna(bib_uids) or not isinstance(bib_uids, str):
            num_docs = 0
        else:
            num_docs = len(bib_uids.split(', ')) if bib_uids else 0
        
        print(f"   Word: {info['term']}")
        print(f"   Frequency: {freq:,} occurrences")
        print(f"   Rate: {rate:.2f} per 10,000 words ({tier_label})")
        if num_docs > 0:
            print(f"   Found in: {num_docs} documents")
        
        # If regex found multiple matches, show them
        if results.get('all_term_matches') is not None and len(results['all_term_matches']) > 1:
            all_matches = results['all_term_matches']
            print(f"\n   📋 Regex also matched {len(all_matches) - 1} other term(s):")
            for idx, row in all_matches.iloc[1:6].iterrows():  # Show up to 5 more
                tier_label_other = tier_labels.get(row['tier'], row['tier'])
                print(f"      • {row['term']} ({row['frequency']:,} occurrences, {row['rate_per_10k']:.2f} per 10k - {tier_label_other})")
            if len(all_matches) > 6:
                print(f"      ... and {len(all_matches) - 6} more")
    else:
        print(f"   ❌ Term '{search_term}' not found in frequency dictionary")
    
    # Display n-gram phrases
    print(f"\n📝 PHRASES (frequency ≥ {min_ngram_freq} only):")
    
    ngram_labels = {
        'bigrams': 'BIGRAMS (2-word phrases)',
        'trigrams': 'TRIGRAMS (3-word phrases)',
        'fourgrams': '4-GRAMS (4-word phrases)',
        'fivegrams': '5-GRAMS (5-word phrases)',
        'sixgrams': '6-GRAMS (6-word phrases)'
    }
    
    found_any_ngrams = False
    
    for ngram_key, label in ngram_labels.items():
        if results.get(ngram_key) is not None and len(results[ngram_key]) > 0:
            found_any_ngrams = True
            ngrams = results[ngram_key]
            
            print(f"\n{label}:")
            
            # Show top 10
            for i, row in ngrams.head(10).iterrows():
                tier_labels = {
                    'extremely_common': 'extremely common',
                    'very_common': 'very common',
                    'common': 'common',
                    'moderate': 'moderate',
                    'rare': 'rare',
                    'hapax': 'hapax'
                }
                tier_label = tier_labels.get(row['tier'], row['tier'])
                print(f"   {i+1:2d}. {row['ngram']} ({row['frequency']:,} times, {row['rate_per_10k']:.2f} per 10k - {tier_label})")
            
            if len(ngrams) > 10:
                print(f"   ... and {len(ngrams) - 10} more")
    
    if not found_any_ngrams:
        print(f"\n   ❌ No phrases found with '{search_term}' (frequency ≥ {min_ngram_freq})")
    
    print("\n" + "=" * 70 + "\n")