#!/usr/bin/env python3
"""
Build Indexed, Tokenized Corpus from Plain Text Sources
========================================================

This library creates an indexed DataFrame of all tokens in the historical corpus,
preserving both original and cleaned versions of each word, then enriches it with
bibliography metadata from the database.

Purpose:
    - Heavy processing: Run periodically when corpus changes
    - Output: Saves indexed corpus to CSV for fast loading
    - Enables: Frequency dictionaries, n-grams, KWIC searches, filtered by metadata

Usage:
    # Two-step workflow (recommended for debugging):
    df = build_indexed_corpus()           # Build base index
    df_enriched = enrich_and_save(df)     # Add metadata + save
    
    # One-step workflow (quick):
    df = build_and_enrich()               # Does everything at once
"""

import pandas as pd
import os
from datetime import datetime

# Import corpus loader from existing library
# Using _get_corpus (private function) with alias to avoid exposing it to end users
from historical_corpus_plain_text import _get_corpus as get_corpus

"""
Configuration & Paths
"""

# Set home directory path
hdir = os.path.expanduser('~')

# Default output location for indexed corpus
default_output_dir = os.path.join(hdir, 'Dropbox/Active_Directories/Digital_Humanities/Datasets/eurasia_indexed_corpus')
default_output_filename = 'eurasia_corpus_indexed.csv'
default_output_path = os.path.join(default_output_dir, default_output_filename)

# Archive directory for timestamped versions
archive_dir = os.path.join(default_output_dir, 'archived_eurasia_indexed_corpus')

# Frequency dictionary output location
freq_dict_dir = os.path.join(hdir, 'Dropbox/Active_Directories/Digital_Humanities/Datasets/eurasia_corpus_frequency_dictionary')
freq_dict_filename = 'eurasia_corpus_frequency_dictionary.csv'
freq_dict_path = os.path.join(freq_dict_dir, freq_dict_filename)

# Archive directory for frequency dictionaries
freq_dict_archive_dir = os.path.join(freq_dict_dir, 'archived_eurasia_corpus_frequency_dictionary')

# N-grams output location (same directory as frequency dictionary for consistency)
# All n-gram files will be saved to: .../eurasia_corpus_frequency_dictionary/
ngrams_dir = freq_dict_dir  # Same directory as frequency dictionary
ngrams_archive_dir = freq_dict_archive_dir  # Same archive directory

# Inbox for reports
inbox_path = os.path.join(hdir, 'Dropbox/Active_Directories/Inbox')


"""
Main Builder Function
"""

def build_indexed_corpus():
    """
    Build indexed token DataFrame with both original and cleaned tokens.
    
    This is the core indexing function - creates the base DataFrame structure
    without any database enrichment or CSV export.
    
    Returns:
        pd.DataFrame: Indexed corpus with columns:
            - document: Source filename
            - position: Token position within document (0-indexed)
            - token_original: Original token (with diacritics, punctuation)
            - token_clean: Cleaned token (normalized for searching)
            - token_id: Unique ID for this token instance
    
    Example:
        >>> df = build_indexed_corpus()
        📚 Loading corpus...
        ✅ Loaded 142 documents
        🔨 Tokenizing and indexing...
        ✅ Indexed 1,234,567 tokens
        
        >>> df.head()
           document  position token_original token_clean  token_id
        0  text1.xml         0         بُخارا       بخارا         0
        1  text1.xml         1           شهر        شهر         1
    
    Notes:
        - Uses get_corpus() from historical_corpus_plain_text.py
        - Preserves both original (with diacritics) and cleaned tokens
        - Creates unique token_id for each word occurrence
        - Position resets to 0 for each document
        - Does NOT save to CSV - use enrich_and_save() for that
    """
    
    print("\n" + "=" * 70)
    print("🏗️  BUILDING INDEXED CORPUS")
    print("=" * 70)
    
    # Step 1: Load raw version of corpus only (we'll clean per-token)
    print("\n📚 Loading corpus (raw version)...")
    corpus_raw = get_corpus(clean=False)
    
    print(f"✅ Loaded {len(corpus_raw)} documents")
    
    # Step 2: Import cleaning function
    try:
        import arabic_cleaning
        clean_document = arabic_cleaning.clean_document
        print("✅ Arabic cleaning module loaded")
    except ImportError:
        print("⚠️  Warning: arabic_cleaning module not found, skipping token cleaning")
        clean_document = None
    
    # Step 3: Tokenize and index all documents
    print("\n🔨 Tokenizing and indexing...")
    all_tokens = []
    global_token_id = 0
    doc_count = 0
    docs_with_mismatches = []  # Track documents with token count issues
    
    for doc_name in corpus_raw.keys():
        raw_text = corpus_raw[doc_name]
        
        # Tokenize raw text first
        raw_tokens = raw_text.split()
        
        # Clean each token individually (not the whole document)
        # This preserves token count and alignment
        if clean_document:
            clean_tokens = [clean_document(token) for token in raw_tokens]
        else:
            clean_tokens = raw_tokens  # Fallback: use raw if no cleaning available
        
        # Check if any tokens became empty after cleaning (non-Arabic tokens)
        # This happens with English/Russian words
        if any(not token.strip() for token in clean_tokens):
            # Count non-empty vs empty
            non_empty = sum(1 for t in clean_tokens if t.strip())
            empty = sum(1 for t in clean_tokens if not t.strip())
            
            if empty == len(clean_tokens):  # All tokens empty (pure English doc)
                docs_with_mismatches.append({
                    'document': doc_name,
                    'raw_tokens': len(raw_tokens),
                    'clean_tokens': 0
                })
                # Skip this document entirely
                continue
            elif empty > 0:  # Some tokens empty (mixed language)
                docs_with_mismatches.append({
                    'document': doc_name,
                    'raw_tokens': len(raw_tokens),
                    'clean_tokens': non_empty
                })
        
        # Create index entry for each token
        for position, (raw_token, clean_token) in enumerate(zip(raw_tokens, clean_tokens)):
            # Skip empty cleaned tokens (English words that got removed)
            if not clean_token.strip():
                continue
            
            all_tokens.append({
                'document': doc_name,
                'position': position,
                'token_original': raw_token,
                'token_clean': clean_token.strip(),
                'token_id': global_token_id
            })
            global_token_id += 1
        
        doc_count += 1
        if doc_count % 10 == 0:
            print(f"   Processed {doc_count}/{len(corpus_raw)} documents...", end='\r')
    
    print(f"   Processed {doc_count}/{len(corpus_raw)} documents... Done!")
    
    # Show summary of documents without Arabic script
    if docs_with_mismatches:
        # Count how many had zero clean tokens (completely non-Arabic)
        non_arabic_docs = [d for d in docs_with_mismatches if d['clean_tokens'] == 0]
        
        if non_arabic_docs:
            print(f"\n   ℹ️  Note: {len(non_arabic_docs)} document(s) contained no Arabic-script data")
            print(f"      (These appear to be English/Russian documents and were filtered by cleaning)")
        
        # If there are other mismatches (partial cleaning), show those separately
        partial_mismatches = [d for d in docs_with_mismatches if d['clean_tokens'] > 0]
        if partial_mismatches:
            print(f"\n   ⚠️  Warning: {len(partial_mismatches)} document(s) had unexpected token count changes:")
            for doc_info in partial_mismatches[:5]:  # Show first 5
                print(f"      • {doc_info['document']}: {doc_info['raw_tokens']} → {doc_info['clean_tokens']} tokens")
            if len(partial_mismatches) > 5:
                print(f"      ... and {len(partial_mismatches) - 5} more")
    
    # Step 3: Create DataFrame
    print("\n📊 Creating DataFrame...")
    df = pd.DataFrame(all_tokens)
    
    # Display statistics
    print(f"\n📈 Corpus Statistics:")
    print(f"   Total tokens: {len(df):,}")
    print(f"   Documents: {df['document'].nunique():,}")
    print(f"   Unique tokens (cleaned): {df['token_clean'].nunique():,}")
    print(f"   Unique tokens (original): {df['token_original'].nunique():,}")
    print(f"   Avg tokens per document: {len(df) / df['document'].nunique():.1f}")
    
    print("\n" + "=" * 70)
    print("✅ INDEXING COMPLETE")
    print("=" * 70)
    print("\n💡 Next step: Use enrich_and_save(df) to add database metadata and save to CSV")
    print("=" * 70 + "\n")
    
    return df


def enrich_and_save(df, database_path=None, output_path=None):
    """
    Enrich indexed corpus with bibliography metadata from database, then save to CSV.
    
    This function:
    1. Extracts bibliography UID from document filenames (pattern: ser<number>)
    2. Joins with bibliography table to add: Language, Status, Tags, Type, Repository_ID
    3. Joins with repositories table to add: Acronym
    4. Saves enriched DataFrame to CSV (latest + timestamped archive)
    
    Args:
        df (pd.DataFrame): Indexed corpus from build_indexed_corpus()
        database_path (str, optional): Path to database. If None, uses default from
                                       database_query_functions.py
        output_path (str, optional): Where to save CSV. If None, uses default location.
    
    Returns:
        pd.DataFrame: Enriched corpus with added columns:
            - bib_uid: Bibliography UID extracted from filename
            - Language: Language(s) from bibliography table
            - Status: Status from bibliography table  
            - Tags: Tags from bibliography table
            - Type: Type from bibliography table
            - Repository_ID: Repository foreign key from bibliography table
            - Repository_Acronym: Acronym from repositories table
    
    Example:
        >>> df = build_indexed_corpus()
        >>> df_enriched = enrich_and_save(df)
        
        🔗 ENRICHING CORPUS WITH DATABASE METADATA
        ══════════════════════════════════════════════════════════════════════
        
        📋 Extracting bibliography UIDs from filenames...
        ✅ Extracted UIDs from 89 documents (53 documents without 'ser' pattern)
        
        🔗 Joining with bibliography table...
        ✅ Matched 1,234,567 tokens to 89 bibliography entries
        
        🔗 Joining with repositories table...
        ✅ Added repository acronyms
        
        💾 Saving to /path/to/eurasia_corpus_indexed.csv...
        ✅ Saved successfully (28.45 MB)
        
        📦 Archiving timestamped copy...
        ✅ Archived to: .../eurasia_corpus_indexed_20241220_153022.csv
        
        ✅ ENRICHMENT COMPLETE
    
    Notes:
        - Documents without 'ser<number>' pattern get NULL for bib_uid and metadata
        - Uses LEFT JOIN so all tokens are preserved even if no database match
        - SQL concept: This is equivalent to:
            SELECT tokens.*, bib.Language, bib.Status, bib.Tags, bib.Type, 
                   bib.Repository_ID, repo.Acronym as Repository_Acronym
            FROM tokens
            LEFT JOIN bibliography bib ON tokens.bib_uid = bib.UID
            LEFT JOIN repositories repo ON bib.Repository_ID = repo.UID
    """
    import sqlite3
    import sys
    
    # Import database_path from database_query_functions if not provided
    if database_path is None:
        sys.path.insert(0, os.path.join(os.path.expanduser('~'), 'Projects/database'))
        from database_query_functions import database_path
    
    print("\n" + "=" * 70)
    print("🔗 ENRICHING CORPUS WITH DATABASE METADATA")
    print("=" * 70)
    
    # Step 1: Extract bibliography UIDs from document filenames
    print("\n📋 Extracting bibliography UIDs from filenames...")
    
    def extract_bib_uid(filename):
        """
        Extract bibliography UID from filename.
        
        Looks for pattern 'ser<number>' in filename and extracts the number.
        
        Args:
            filename: Document filename (e.g., "text_ser123.xml")
        
        Returns:
            int: Bibliography UID, or None if pattern not found
        
        Examples:
            >>> extract_bib_uid("bukhara_ser456.xml")
            456
            >>> extract_bib_uid("no_serial.txt")
            None
        """
        import re
        ser_match = re.search(r'ser(\d+)', filename)
        return int(ser_match.group(1)) if ser_match else None
    
    # Add bib_uid column
    df['bib_uid'] = df['document'].apply(extract_bib_uid)
    
    # Convert to nullable integer type (avoids float representation with NaN)
    df['bib_uid'] = df['bib_uid'].astype('Int64')
    
    # Count how many documents have UIDs
    docs_with_uid = df[df['bib_uid'].notna()]['document'].nunique()
    docs_without_uid = df[df['bib_uid'].isna()]['document'].nunique()
    
    print(f"✅ Extracted UIDs from {docs_with_uid} documents ({docs_without_uid} documents without 'ser' pattern)")
    
    # Step 2: Connect to database and fetch bibliography metadata
    print("\n🔗 Joining with bibliography table...")
    
    conn = sqlite3.connect(database_path)
    
    # Get bibliography data (Language, Status, Tags, Type, Repository_ID)
    bib_query = """
        SELECT 
            UID as bib_uid,
            Language,
            Status,
            Tags,
            Type,
            Repository_ID
        FROM bibliography
    """
    
    bib_df = pd.read_sql_query(bib_query, conn)
    
    # LEFT JOIN: Keep all tokens, add bibliography metadata where available
    # This is equivalent to SQL:
    # SELECT * FROM df LEFT JOIN bib_df ON df.bib_uid = bib_df.bib_uid
    df_enriched = df.merge(bib_df, on='bib_uid', how='left')
    
    # Count successful matches
    matched_tokens = df_enriched['Language'].notna().sum()
    print(f"✅ Matched {matched_tokens:,} tokens to {bib_df['bib_uid'].nunique()} bibliography entries")
    
    # Step 3: Join with repositories table to get Acronym
    print("\n🔗 Joining with repositories table...")
    
    repo_query = """
        SELECT 
            UID as Repository_ID,
            Acronym as Repository_Acronym
        FROM repositories
    """
    
    repo_df = pd.read_sql_query(repo_query, conn)
    
    # LEFT JOIN again: Add repository acronym based on Repository_ID
    # SQL: LEFT JOIN repositories ON df_enriched.Repository_ID = repositories.UID
    df_enriched = df_enriched.merge(repo_df, on='Repository_ID', how='left')
    
    print(f"✅ Added repository acronyms")
    
    conn.close()
    
    # Step 4: Save to CSV
    print("\n💾 Saving enriched corpus...")
    
    # Use default path if none provided
    if output_path is None:
        output_path = default_output_path
    
    # Create main directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create archive directory if it doesn't exist
    os.makedirs(archive_dir, exist_ok=True)
    
    print(f"\n💾 Saving to {output_path}...")
    df_enriched.to_csv(output_path, index=False, encoding='utf-8')
    
    # Show file size
    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✅ Saved successfully ({file_size_mb:.2f} MB)")
    
    # Save timestamped archive version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_filename = f'eurasia_corpus_indexed_{timestamp}.csv'
    archive_path = os.path.join(archive_dir, archive_filename)
    
    print(f"\n📦 Archiving timestamped copy...")
    df_enriched.to_csv(archive_path, index=False, encoding='utf-8')
    print(f"✅ Archived to: {archive_path}")
    
    # Display enriched statistics
    print("\n📊 Enriched Corpus Statistics:")
    print(f"   Total tokens: {len(df_enriched):,}")
    print(f"   Tokens with bibliography metadata: {matched_tokens:,} ({matched_tokens/len(df_enriched)*100:.1f}%)")
    print(f"   Unique languages: {df_enriched['Language'].nunique()}")
    print(f"   Unique repositories: {df_enriched['Repository_Acronym'].nunique()}")
    
    print("\n" + "=" * 70)
    print("✅ ENRICHMENT COMPLETE")
    print("=" * 70 + "\n")
    
    return df_enriched


def build_and_enrich(database_path=None, output_path=None):
    """
    Convenience function: Build indexed corpus AND enrich with database metadata in one step.
    
    This combines build_indexed_corpus() + enrich_and_save() for ease of use.
    
    Args:
        database_path (str, optional): Path to database
        output_path (str, optional): Where to save CSV
    
    Returns:
        pd.DataFrame: Fully enriched and saved corpus
    
    Example:
        >>> df = build_and_enrich()
        # Runs both steps automatically
    """
    print("\n🚀 FULL WORKFLOW: BUILD + ENRICH + SAVE\n")
    
    # Step 1: Build base indexed corpus
    df = build_indexed_corpus()
    
    # Step 2: Enrich with database metadata and save
    df_enriched = enrich_and_save(df, database_path=database_path, output_path=output_path)
    
    return df_enriched


def create_frequency_dictionary(df, output_path=None):
    """
    Create a frequency dictionary from the indexed corpus.
    
    Analyzes the token_clean column to generate word frequencies with:
    - Total occurrences
    - Percentile ranking (what % of words are MORE frequent)
    - Bibliography UIDs where each word appears
    
    Args:
        df (pd.DataFrame): Indexed corpus (enriched with bib_uid preferred)
        output_path (str, optional): Where to save CSV. If None, uses default location.
    
    Returns:
        pd.DataFrame: Frequency dictionary with columns:
            - UID: Unique identifier for this word
            - term: The word (from token_clean column)
            - frequency: Number of occurrences
            - percentile: What % of words are MORE frequent (lower = more common)
            - bib_uids: Comma-separated list of bibliography UIDs where word appears
    
    Example:
        >>> df_enriched = pd.read_csv('eurasia_corpus_indexed.csv')
        >>> freq_df = create_frequency_dictionary(df_enriched)
        
        📊 CREATING FREQUENCY DICTIONARY
        ══════════════════════════════════════════════════════════════════════
        
        🔢 Counting word frequencies...
        ✅ Found 45,678 unique words across 1,234,567 tokens
        
        📊 Calculating percentiles...
        ✅ Percentiles calculated
        
        📋 Gathering bibliography UIDs...
        ✅ Bibliography UIDs compiled
        
        💾 Saving to /path/to/eurasia_corpus_frequency_dictionary.csv...
        ✅ Saved successfully (2.34 MB)
        
        Top 10 most frequent words:
           1. و        123,456 occurrences (Top 0.0%)
           2. که       98,765 occurrences (Top 0.0%)
           ...
    
    Notes:
        - Uses token_clean for accurate frequency counting (normalized forms)
        - Percentile: Lower = more common (Top 1% = very frequent)
        - Bibliography UIDs: Only non-null UIDs are included
        - SQL equivalent:
            SELECT 
                ROW_NUMBER() as UID,
                token_clean as term,
                COUNT(*) as frequency,
                GROUP_CONCAT(DISTINCT bib_uid) as bib_uids
            FROM corpus
            GROUP BY token_clean
            ORDER BY frequency DESC
    """
    
    print("\n" + "=" * 70)
    print("📊 CREATING FREQUENCY DICTIONARY")
    print("=" * 70)
    
    # Step 1: Count word frequencies
    print("\n🔢 Counting word frequencies...")
    
    # Group by token_clean and count occurrences
    # This is equivalent to SQL: SELECT token_clean, COUNT(*) FROM df GROUP BY token_clean
    freq_counts = df['token_clean'].value_counts().reset_index()
    freq_counts.columns = ['term', 'frequency']
    
    total_unique_words = len(freq_counts)
    total_tokens = df['token_clean'].count()
    
    print(f"✅ Found {total_unique_words:,} unique words across {total_tokens:,} tokens")
    
    # Step 2: Calculate rate per 10,000 words (more intuitive than percentile)
    print("\n📊 Calculating frequency rates...")
    
    # Total occurrences in corpus
    total_occurrences = freq_counts['frequency'].sum()
    
    # Rate per 10,000 words
    # This shows: "how many times does this word appear per 10k words?"
    # Examples: 
    #   - و (very common): might be 500 per 10k
    #   - اژدها (rare): might be 0.02 per 10k
    freq_counts['rate_per_10k'] = (freq_counts['frequency'] / total_occurrences) * 10000
    
    # Classify into frequency tiers for quick intuition
    def classify_frequency(freq):
        """
        Classify word frequency into tiers.
        
        Args:
            freq (int): Number of occurrences
        
        Returns:
            str: Tier label (extremely_common, very_common, common, moderate, rare, hapax)
        """
        if freq >= 10000:
            return "extremely_common"
        elif freq >= 1000:
            return "very_common"
        elif freq >= 100:
            return "common"
        elif freq >= 10:
            return "moderate"
        elif freq >= 2:
            return "rare"
        else:
            return "hapax"
    
    freq_counts['tier'] = freq_counts['frequency'].apply(classify_frequency)
    
    print(f"✅ Frequency rates calculated")
    
    # Step 3: Gather bibliography UIDs for each word
    print("\n📋 Gathering bibliography UIDs...")
    
    # Check if bib_uid column exists
    if 'bib_uid' in df.columns:
        # Group by token_clean and collect unique bib_uids
        # This is equivalent to SQL: GROUP_CONCAT(DISTINCT bib_uid)
        bib_uid_mapping = df.groupby('token_clean')['bib_uid'].apply(
            lambda x: ', '.join(map(str, sorted(x.dropna().unique().astype(int))))
        ).to_dict()
        
        # Add bib_uids column
        freq_counts['bib_uids'] = freq_counts['term'].map(bib_uid_mapping)
        
        print(f"✅ Bibliography UIDs compiled")
    else:
        # No bib_uid column, leave empty
        freq_counts['bib_uids'] = ''
        print(f"⚠️  Warning: No bib_uid column found, bib_uids will be empty")
    
    # Step 4: Add UID column (sequential identifier)
    # Start from 1 for human-friendliness
    freq_counts.insert(0, 'UID', range(1, len(freq_counts) + 1))
    
    # Step 5: Reorder columns to match specification
    # UID, term, frequency, rate_per_10k, tier, bib_uids
    freq_counts = freq_counts[['UID', 'term', 'frequency', 'rate_per_10k', 'tier', 'bib_uids']]
    
    # Step 6: Save to CSV
    print("\n💾 Saving frequency dictionary...")
    
    # Use default path if none provided
    if output_path is None:
        output_path = freq_dict_path
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(freq_dict_archive_dir, exist_ok=True)
    
    print(f"\n💾 Saving to {output_path}...")
    freq_counts.to_csv(output_path, index=False, encoding='utf-8')
    
    # Show file size
    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)
    print(f"✅ Saved successfully ({file_size_mb:.2f} MB)")
    
    # Save timestamped archive version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_filename = f'eurasia_corpus_frequency_dictionary_{timestamp}.csv'
    archive_path = os.path.join(freq_dict_archive_dir, archive_filename)
    
    print(f"\n📦 Archiving timestamped copy...")
    freq_counts.to_csv(archive_path, index=False, encoding='utf-8')
    print(f"✅ Archived to: {archive_path}")
    
    # Display statistics
    print("\n📈 Frequency Dictionary Statistics:")
    print(f"   Total unique words: {len(freq_counts):,}")
    print(f"   Total occurrences: {freq_counts['frequency'].sum():,}")
    print(f"   Most frequent word: '{freq_counts.iloc[0]['term']}' ({freq_counts.iloc[0]['frequency']:,} times)")
    print(f"   Least frequent words: {len(freq_counts[freq_counts['frequency'] == 1]):,} hapax legomena (occur only once)")
    
    # Show top 10 most frequent
    print(f"\n🔝 Top 10 most frequent words:")
    for i, row in freq_counts.head(10).iterrows():
        tier_labels = {
            'extremely_common': 'extremely common',
            'very_common': 'very common',
            'common': 'common',
            'moderate': 'moderate',
            'rare': 'rare',
            'hapax': 'hapax'
        }
        tier_label = tier_labels.get(row['tier'], row['tier'])
        print(f"   {row['UID']:3d}. {row['term']:20s} {row['frequency']:>8,} occurrences ({row['rate_per_10k']:>6.2f} per 10k, {tier_label})")
    
    print("\n" + "=" * 70)
    print("✅ FREQUENCY DICTIONARY COMPLETE")
    print("=" * 70 + "\n")
    
    return freq_counts


def generate_ngrams(df, max_n=6):
    """
    Generate n-gram frequency DataFrames from indexed corpus.
    
    Creates n-grams (word sequences) of sizes 2 through max_n, preserving
    document boundaries (doesn't create n-grams across documents).
    
    Args:
        df (pd.DataFrame): Indexed corpus (with token_clean, document, position columns)
        max_n (int): Maximum n-gram size (default: 6)
                    Will generate 2-grams, 3-grams, ..., up to max_n-grams
    
    Returns:
        dict: Dictionary of DataFrames keyed by n-gram size
              {2: bigrams_df, 3: trigrams_df, 4: fourgrams_df, ...}
              
              Each DataFrame has columns:
              - UID: Unique identifier
              - ngram: The n-gram as space-separated words
              - frequency: Number of occurrences
              - percentile: What % of n-grams are MORE frequent
              - bib_uids: Bibliography UIDs where n-gram appears
    
    Example:
        >>> df = pd.read_csv('eurasia_corpus_indexed.csv')
        >>> ngrams = generate_ngrams(df, max_n=4)
        >>> # Returns: {2: bigrams_df, 3: trigrams_df, 4: fourgrams_df}
        
        📊 GENERATING N-GRAMS (2 through 4)
        ══════════════════════════════════════════════════════════════════════
        
        🔢 Generating 2-grams (bigrams)...
        ✅ Found 123,456 unique bigrams
        
        🔢 Generating 3-grams (trigrams)...
        ✅ Found 234,567 unique trigrams
        
        🔢 Generating 4-grams...
        ✅ Found 345,678 unique 4-grams
        
        ✅ N-GRAM GENERATION COMPLETE
    
    Notes:
        - N-grams do NOT cross document boundaries
        - Uses token_clean for normalized forms
        - Sorted by frequency (most common first)
        - SQL equivalent for bigrams:
            SELECT 
                CONCAT(t1.token_clean, ' ', t2.token_clean) as ngram,
                COUNT(*) as frequency
            FROM tokens t1
            JOIN tokens t2 ON t1.document = t2.document 
                          AND t2.position = t1.position + 1
            GROUP BY ngram
    """
    
    print("\n" + "=" * 70)
    print(f"📊 GENERATING N-GRAMS (2 through {max_n})")
    print("=" * 70)
    
    # Sort by document and position to ensure correct ordering
    df_sorted = df.sort_values(['document', 'position']).reset_index(drop=True)
    
    ngram_dfs = {}
    
    # Generate n-grams for each size from 2 to max_n
    for n in range(2, max_n + 1):
        ngram_name = {2: 'bigrams', 3: 'trigrams', 4: '4-grams', 5: '5-grams', 6: '6-grams'}.get(n, f'{n}-grams')
        print(f"\n🔢 Generating {n}-grams ({ngram_name})...")
        
        ngram_list = []
        
        # Group by document to avoid creating n-grams across document boundaries
        for doc_name, doc_group in df_sorted.groupby('document'):
            # Get bib_uid for this document (should be same for all rows)
            bib_uid = doc_group['bib_uid'].iloc[0] if 'bib_uid' in doc_group.columns else None
            
            # Get tokens for this document
            tokens = doc_group['token_clean'].tolist()
            
            # Create n-grams by sliding window
            # For n=2: tokens[0:2], tokens[1:3], tokens[2:4], ...
            # For n=3: tokens[0:3], tokens[1:4], tokens[2:5], ...
            for i in range(len(tokens) - n + 1):
                ngram_tokens = tokens[i:i+n]
                
                # Skip if any token is empty (shouldn't happen, but safety check)
                if all(ngram_tokens):
                    ngram_text = ' '.join(ngram_tokens)
                    ngram_list.append({
                        'ngram': ngram_text,
                        'bib_uid': bib_uid
                    })
        
        # Convert to DataFrame
        ngrams_df = pd.DataFrame(ngram_list)
        
        # Count frequencies
        # Group by ngram and count occurrences
        # SQL equivalent: SELECT ngram, COUNT(*) as frequency, GROUP_CONCAT(DISTINCT bib_uid) 
        #                 FROM ngrams GROUP BY ngram
        freq_counts = ngrams_df.groupby('ngram').agg(
            frequency=('ngram', 'size'),  # Count occurrences - use named aggregation
            bib_uids=('bib_uid', lambda x: ', '.join(map(str, sorted(x.dropna().unique().astype(int)))) if 'bib_uid' in ngrams_df.columns else '')
        ).reset_index()
        
        # Sort by frequency (descending)
        freq_counts = freq_counts.sort_values('frequency', ascending=False).reset_index(drop=True)
        
        # Calculate rate per 10,000 n-grams and classify into tiers
        total_ngram_occurrences = freq_counts['frequency'].sum()
        freq_counts['rate_per_10k'] = (freq_counts['frequency'] / total_ngram_occurrences) * 10000
        
        # Classify frequency tiers
        def classify_frequency(freq):
            if freq >= 10000:
                return "extremely_common"
            elif freq >= 1000:
                return "very_common"
            elif freq >= 100:
                return "common"
            elif freq >= 10:
                return "moderate"
            elif freq >= 2:
                return "rare"
            else:
                return "hapax"
        
        freq_counts['tier'] = freq_counts['frequency'].apply(classify_frequency)
        
        # Add UID column
        freq_counts.insert(0, 'UID', range(1, len(freq_counts) + 1))
        
        # Reorder columns: UID, ngram, frequency, rate_per_10k, tier, bib_uids
        freq_counts = freq_counts[['UID', 'ngram', 'frequency', 'rate_per_10k', 'tier', 'bib_uids']]
        
        # Store in dictionary
        ngram_dfs[n] = freq_counts
        
        print(f"✅ Found {len(freq_counts):,} unique {n}-grams")
        
        # Show top 5
        print(f"   Top 5 {ngram_name}:")
        for i, row in freq_counts.head(5).iterrows():
            print(f"      {i+1}. '{row['ngram']}' ({row['frequency']:,} occurrences)")
    
    print("\n" + "=" * 70)
    print("✅ N-GRAM GENERATION COMPLETE")
    print("=" * 70 + "\n")
    
    return ngram_dfs


def save_ngrams(ngram_dfs, output_dir=None):
    """
    Save n-gram DataFrames to CSV files.
    
    Saves each n-gram size to a separate CSV file with timestamp archiving.
    
    Args:
        ngram_dfs (dict): Dictionary of n-gram DataFrames from generate_ngrams()
                         {2: bigrams_df, 3: trigrams_df, ...}
        output_dir (str, optional): Directory to save files. If None, uses default.
    
    Returns:
        dict: Paths to saved files {2: 'path/to/bigrams.csv', ...}
    
    Example:
        >>> ngrams = generate_ngrams(df, max_n=4)
        >>> paths = save_ngrams(ngrams)
        
        💾 SAVING N-GRAMS TO CSV
        ══════════════════════════════════════════════════════════════════════
        
        📄 Saving 2-grams (bigrams)...
        ✅ Saved to: .../eurasia_corpus_bigrams.csv (5.67 MB)
        📦 Archived to: .../eurasia_corpus_bigrams_20241220_154523.csv
        
        📄 Saving 3-grams (trigrams)...
        ✅ Saved to: .../eurasia_corpus_trigrams.csv (8.45 MB)
        ...
    
    Output filenames:
        - eurasia_corpus_bigrams.csv
        - eurasia_corpus_trigrams.csv
        - eurasia_corpus_4grams.csv
        - eurasia_corpus_5grams.csv
        - eurasia_corpus_6grams.csv
    """
    
    print("\n" + "=" * 70)
    print("💾 SAVING N-GRAMS TO CSV")
    print("=" * 70)
    
    # Use default directory if none provided
    if output_dir is None:
        output_dir = ngrams_dir
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ngrams_archive_dir, exist_ok=True)
    
    saved_paths = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define friendly names for each n-gram size
    ngram_names = {
        2: 'bigrams',
        3: 'trigrams',
        4: '4grams',
        5: '5grams',
        6: '6grams'
    }
    
    for n, df in sorted(ngram_dfs.items()):
        ngram_name = ngram_names.get(n, f'{n}grams')
        display_name = {2: '2-grams (bigrams)', 3: '3-grams (trigrams)'}.get(n, f'{n}-grams')
        
        print(f"\n📄 Saving {display_name}...")
        
        # Create filenames
        latest_filename = f'eurasia_corpus_{ngram_name}.csv'
        archive_filename = f'eurasia_corpus_{ngram_name}_{timestamp}.csv'
        
        latest_path = os.path.join(output_dir, latest_filename)
        archive_path = os.path.join(ngrams_archive_dir, archive_filename)
        
        # Save latest version
        df.to_csv(latest_path, index=False, encoding='utf-8')
        
        # Show file size
        file_size = os.path.getsize(latest_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"✅ Saved to: {latest_path} ({file_size_mb:.2f} MB)")
        
        # Save archived version
        df.to_csv(archive_path, index=False, encoding='utf-8')
        print(f"📦 Archived to: {archive_path}")
        
        saved_paths[n] = latest_path
    
    print("\n" + "=" * 70)
    print("✅ ALL N-GRAMS SAVED")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📁 Archive directory: {ngrams_archive_dir}")
    print("=" * 70 + "\n")
    
    return saved_paths


def create_ngrams(df, max_n=6, output_dir=None):
    """
    Convenience function: Generate AND save n-grams in one step.
    
    Combines generate_ngrams() + save_ngrams() for ease of use.
    
    Args:
        df (pd.DataFrame): Indexed corpus
        max_n (int): Maximum n-gram size (default: 6, creates 2-6 grams)
        output_dir (str, optional): Directory to save files
    
    Returns:
        dict: Dictionary of n-gram DataFrames {2: bigrams_df, 3: trigrams_df, ...}
    
    Example:
        >>> df = pd.read_csv('eurasia_corpus_indexed.csv')
        >>> ngrams = create_ngrams(df)  # Creates 2-6 grams by default
        >>> ngrams = create_ngrams(df, max_n=4)  # Creates only 2-4 grams
    """
    print(f"\n🚀 FULL N-GRAM WORKFLOW: GENERATE + SAVE (2 through {max_n}-grams)\n")
    
    # Step 1: Generate n-grams
    ngram_dfs = generate_ngrams(df, max_n=max_n)
    
    # Step 2: Save to CSV
    save_ngrams(ngram_dfs, output_dir=output_dir)
    
    return ngram_dfs


"""
Helper Functions
"""

def preview_corpus():
    """
    Preview what will be indexed without actually building the corpus.
    Useful for checking before running the full build.
    
    Returns:
        None (prints formatted preview)
    
    Example:
        >>> preview_corpus()
        📋 CORPUS PREVIEW
        ══════════════════════════════════════════════════════════════════════
        Total documents: 142
        
        Sample documents (first 10):
           1. bukhara_history.xml          (45,234 chars)
           2. samarqand_chronicle.txt      (32,891 chars)
           ...
        
        Estimated tokens: ~1,234,567
        ══════════════════════════════════════════════════════════════════════
    """
    print("\n" + "=" * 70)
    print("📋 CORPUS PREVIEW")
    print("=" * 70)
    
    print("\n📚 Loading corpus info...")
    corpus_raw = get_corpus(clean=False)
    
    print(f"Total documents: {len(corpus_raw)}")
    
    # Count total characters
    total_chars = sum(len(text) for text in corpus_raw.values())
    print(f"Total characters: {total_chars:,}")
    
    # Estimate token count (rough: chars / 5)
    estimated_tokens = total_chars // 5
    print(f"Estimated tokens: ~{estimated_tokens:,}")
    
    # Show sample documents
    print(f"\nSample documents (first 10):")
    for i, (filename, text) in enumerate(list(corpus_raw.items())[:10], 1):
        char_count = len(text)
        token_estimate = char_count // 5
        print(f"  {i:2d}. {filename:50s} ({char_count:,} chars, ~{token_estimate:,} tokens)")
    
    if len(corpus_raw) > 10:
        print(f"  ... and {len(corpus_raw) - 10} more documents")
    
    print("\n" + "=" * 70)
    print("💡 Run build_indexed_corpus() to create the indexed DataFrame")
    print("=" * 70 + "\n")


def corpus_info(df):
    """
    Display detailed statistics about an indexed corpus DataFrame.
    
    Args:
        df (pd.DataFrame): Indexed corpus (from build_indexed_corpus or load)
    
    Returns:
        None (prints formatted statistics)
    
    Example:
        >>> df = pd.read_csv('eurasia_corpus_indexed.csv')
        >>> corpus_info(df)
        📊 INDEXED CORPUS STATISTICS
        ══════════════════════════════════════════════════════════════════════
        Total tokens: 1,234,567
        Unique documents: 142
        Unique tokens (cleaned): 45,678
        ...
    """
    print("\n" + "=" * 70)
    print("📊 INDEXED CORPUS STATISTICS")
    print("=" * 70)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"  Rows (tokens): {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    print(f"\nToken counts:")
    print(f"  Total tokens: {len(df):,}")
    print(f"  Unique tokens (cleaned): {df['token_clean'].nunique():,}")
    print(f"  Unique tokens (original): {df['token_original'].nunique():,}")
    
    print(f"\nDocument counts:")
    print(f"  Total documents: {df['document'].nunique():,}")
    print(f"  Avg tokens per document: {len(df) / df['document'].nunique():.1f}")
    
    # Top 10 most frequent tokens
    print(f"\nTop 10 most frequent tokens (cleaned):")
    top_tokens = df['token_clean'].value_counts().head(10)
    for i, (token, count) in enumerate(top_tokens.items(), 1):
        pct = (count / len(df)) * 100
        print(f"  {i:2d}. {token:20s} {count:>8,} occurrences ({pct:5.2f}%)")
    
    # Document sizes
    print(f"\nDocument sizes (tokens):")
    doc_sizes = df.groupby('document').size()
    print(f"  Smallest: {doc_sizes.min():,} tokens")
    print(f"  Largest: {doc_sizes.max():,} tokens")
    print(f"  Median: {doc_sizes.median():.0f} tokens")
    
    print("\n" + "=" * 70 + "\n")


def quick_stats(df):
    """
    Quick one-line statistics (useful for Jupyter notebooks).
    
    Args:
        df (pd.DataFrame): Indexed corpus
    
    Returns:
        None (prints single line summary)
    
    Example:
        >>> quick_stats(df)
        📊 1,234,567 tokens | 142 docs | 45,678 unique words
    """
    total_tokens = len(df)
    num_docs = df['document'].nunique()
    unique_words = df['token_clean'].nunique()
    
    print(f"📊 {total_tokens:,} tokens | {num_docs} docs | {unique_words:,} unique words")


"""
Quality Control Functions
"""

def check_token_alignment(sample_size=5):
    """
    Check if raw and cleaned tokens align properly.
    Useful for debugging cleaning issues.
    
    Args:
        sample_size (int): Number of random documents to check
    
    Returns:
        None (prints comparison)
    
    Example:
        >>> check_token_alignment(sample_size=3)
        🔍 CHECKING TOKEN ALIGNMENT
        ══════════════════════════════════════════════════════════════════════
        Document: bukhara.xml
          Position 0:
            Original: "بُخارا"
            Cleaned:  "بخارا"  ✓
          Position 1:
            Original: "شهر"
            Cleaned:  "شهر"  ✓
        ...
    """
    print("\n" + "=" * 70)
    print("🔍 CHECKING TOKEN ALIGNMENT")
    print("=" * 70)
    
    corpus_raw = get_corpus(clean=False)
    corpus_clean = get_corpus(clean=True)
    
    # Sample random documents
    import random
    sample_docs = random.sample(list(corpus_raw.keys()), min(sample_size, len(corpus_raw)))
    
    for doc_name in sample_docs:
        print(f"\nDocument: {doc_name}")
        
        raw_tokens = corpus_raw[doc_name].split()
        clean_tokens = corpus_clean[doc_name].split()
        
        print(f"  Token counts: Raw={len(raw_tokens)}, Clean={len(clean_tokens)}")
        
        if len(raw_tokens) != len(clean_tokens):
            print(f"  ❌ MISMATCH! Token counts don't align")
            print(f"     This will cause indexing errors")
            continue
        
        # Show first 5 tokens
        print(f"  First 5 token pairs:")
        for i in range(min(5, len(raw_tokens))):
            match_symbol = "✓" if raw_tokens[i] != clean_tokens[i] else "="
            print(f"    {i}. '{raw_tokens[i]}' → '{clean_tokens[i]}' {match_symbol}")
    
    print("\n" + "=" * 70 + "\n")


"""
Display library info on load
"""

# Always display when loaded (either as import or with python3 -i)
print("\n" + "=" * 70)
print("📚 EURASIA HISTORICAL CORPUS - INDEXER LIBRARY")
print("=" * 70)

print("\n📖 What this library does:")
print("   This library builds an INDEXED, TOKENIZED corpus from your plain text")
print("   historical sources, then enriches it with bibliography metadata from")
print("   your database. Each word becomes a row in a DataFrame, enabling:")
print("   • Fast frequency analysis and n-gram extraction")
print("   • Keyword-in-context (KWIC) searches")
print("   • Statistical analysis filtered by language, repository, etc.")

print("\n💾 Output locations:")
print(f"   Latest version: {default_output_path}")
print(f"   Archived copies: {archive_dir}/")
print("   (Timestamped archives created automatically on each build)")

print("\n⚙️  Two-step workflow:")
print("   STEP 1: build_indexed_corpus()")
print("      • Loads corpus from historical_corpus_plain_text library")
print("      • Tokenizes (splits into words) both raw and cleaned versions")
print("      • Creates base DataFrame with token data")
print("      • Returns DataFrame (does NOT save to CSV yet)")

print("\n   STEP 2: enrich_and_save(df)")
print("      • Extracts bibliography UID from filenames (ser<number> pattern)")
print("      • Joins with bibliography table (Language, Status, Tags, Type, Repository_ID)")
print("      • Joins with repositories table (Acronym)")
print("      • Saves enriched corpus to CSV (latest + timestamped archive)")

print("\n📊 Final DataFrame columns:")
print("   • document: Source filename")
print("   • position: Word position in document (0-indexed)")
print("   • token_original: Original word (with diacritics/punctuation)")
print("   • token_clean: Cleaned/normalized word (for searching)")
print("   • token_id: Unique ID for each word occurrence")
print("   • bib_uid: Bibliography UID (extracted from filename)")
print("   • Language: From bibliography table")
print("   • Status: From bibliography table")
print("   • Tags: From bibliography table")
print("   • Type: From bibliography table")
print("   • Repository_ID: From bibliography table")
print("   • Repository_Acronym: From repositories table")

print("\n" + "=" * 70)
print("🛠️  AVAILABLE FUNCTIONS")
print("=" * 70)
print("\n🏗️  Core Functions:")
print("   • build_indexed_corpus()")
print("     → Build base indexed DataFrame (no database join, no save)")
print("   • enrich_and_save(df, database_path=None, output_path=None)")
print("     → Add database metadata and save to CSV")
print("   • build_and_enrich(database_path=None, output_path=None)")
print("     → Combined: Build + Enrich + Save in one step")
print("   • create_frequency_dictionary(df, output_path=None)")
print("     → Generate word frequency dictionary with percentiles and bib UIDs")

print("\n📊 N-gram Functions:")
print("   • generate_ngrams(df, max_n=6)")
print("     → Generate n-gram DataFrames (2-grams through max_n-grams)")
print("   • save_ngrams(ngram_dfs, output_dir=None)")
print("     → Save n-gram DataFrames to CSV files")
print("   • create_ngrams(df, max_n=6, output_dir=None)")
print("     → Combined: Generate + Save n-grams in one step")

print("\n🔍 Preview & QC:")
print("   • preview_corpus()")
print("     → Preview what will be indexed (fast)")
print("   • check_token_alignment(sample_size=5)")
print("     → Check if cleaning preserves token alignment")

print("\n📊 Statistics:")
print("   • corpus_info(df)")
print("     → Detailed statistics about indexed corpus")
print("   • quick_stats(df)")
print("     → One-line summary")

print("\n" + "=" * 70)
print("💡 QUICK START WORKFLOW")
print("=" * 70)
print("   OPTION A - Two-step (recommended for debugging):")
print("   1. preview_corpus()              # See what will be indexed (fast)")
print("   2. df = build_indexed_corpus()   # Build base (slow, ~2-5 min)")
print("   3. corpus_info(df)               # Inspect before enriching")
print("   4. df_enriched = enrich_and_save(df)  # Add metadata + save")

print("\n   OPTION B - One-step (quick and easy):")
print("   1. preview_corpus()              # Optional: preview first")
print("   2. df = build_and_enrich()       # Does everything at once")
print("   3. freq_df = create_frequency_dictionary(df)  # Generate frequency dictionary")
print("   4. ngrams = create_ngrams(df)    # Generate all n-grams (2-6 by default)")
print("   4a. ngrams = create_ngrams(df, max_n=4)  # Or just 2-4 grams")

print("\n   After first build, load from CSV instead of rebuilding:")
print("   df = pd.read_csv('" + default_output_path + "')")
print("   freq_df = create_frequency_dictionary(df)  # Create freq dict from saved corpus")
print("   ngrams = create_ngrams(df, max_n=6)  # Generate n-grams from saved corpus")
print("=" * 70 + "\n")