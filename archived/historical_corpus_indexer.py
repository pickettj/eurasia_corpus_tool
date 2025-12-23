#!/usr/bin/env python3
"""
Create an indexed dataset from the historical corpus
"""

import pandas as pd
import re
import os
from datetime import datetime
from historical_corpus_plain_text import get_corpus

def create_indexed_corpus(use_clean=True):
    """
    Create an indexed dataframe from the corpus where each row is a word.
    
    Args:
        use_clean: Whether to use cleaned or raw text
    
    Returns:
        pandas.DataFrame: Indexed corpus with columns:
            - word_id: Unique identifier across entire corpus
            - text_uid: Filename/document identifier  
            - word_position: Position of word within that document (1-indexed)
            - word_text: The actual word
            - bib_foreign_key: Number extracted from "ser" pattern in text_uid
    """
    # Get the corpus using your existing function
    corpus = get_corpus(clean=use_clean)
    
    # List to store all word records
    word_records = []
    global_word_id = 1
    
    # Process each document
    for text_uid, text_content in corpus.items():
        # Extract bib_foreign_key from text_uid using regex
        ser_match = re.search(r'ser(\d+)', text_uid)
        bib_foreign_key = int(ser_match.group(1)) if ser_match else None
        
        # Simple tokenization: split on whitespace
        words = text_content.split()
        
        for word_position, word_text in enumerate(words, 1):
            if word_text.strip():  # Skip empty strings
                word_record = {
                    'word_id': global_word_id,
                    'text_uid': text_uid,
                    'word_position': word_position,
                    'word_text': word_text.strip(),
                    'bib_foreign_key': bib_foreign_key
                }
                
                word_records.append(word_record)
                global_word_id += 1
    
    # Create DataFrame
    df = pd.DataFrame(word_records)
    
    # Convert bib_foreign_key to nullable integer type
    df['bib_foreign_key'] = df['bib_foreign_key'].astype('Int64')
    
    return df

def save_corpus_csv(df):
    """
    Save the indexed corpus dataframe as a CSV file with timestamp.
    
    Args:
        df: The indexed corpus dataframe to save
    
    Returns:
        str: Full path to the saved file
    """
    # Set up directory paths
    hdir = os.path.expanduser('~')
    datasets_dir = hdir + "/Dropbox/Active_Directories/Digital_Humanities/Datasets/eurasia_indexed_corpus"
    archived_dir = os.path.join(datasets_dir, "archived_eurasia_indexed_corpus")
    
    # Create directories if they don't exist
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(archived_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filenames and paths
    latest_filename = "eurasia_indexed_corpus.csv"
    archived_filename = f"eurasia_indexed_corpus_{timestamp}.csv"
    
    latest_path = os.path.join(datasets_dir, latest_filename)
    archived_path = os.path.join(archived_dir, archived_filename)
    
    # Save both files
    df.to_csv(latest_path, index=False, encoding='utf-8')
    df.to_csv(archived_path, index=False, encoding='utf-8')
    
    print(f"Saved latest corpus to: {latest_path}")
    print(f"Saved archived corpus to: {archived_path}")
    return latest_path

def ec_freq(df):
    """
    Create a frequency dictionary dataframe from the indexed corpus.
    
    Args:
        df: The indexed corpus dataframe
    
    Returns:
        pandas.DataFrame: Frequency dataframe with columns:
            - word_text: The word/string
            - frequency: Count of instances in entire corpus
            - bib_foreign_keys: Comma-separated list of unique bib_foreign_keys for this word
    """
    # Group by word_text and aggregate
    freq_data = df.groupby('word_text').agg({
        'word_id': 'count',  # Count instances (frequency)
        'bib_foreign_key': lambda x: ', '.join(map(str, sorted(x.dropna().unique())))  # Unique bib_foreign_keys
    }).reset_index()
    
    # Rename columns
    freq_data.columns = ['word_text', 'frequency', 'bib_foreign_keys']
    
    # Sort by frequency (descending)
    freq_data = freq_data.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    return freq_data

def save_freq_csv(df):
    """
    Save the frequency dictionary dataframe as a CSV file with timestamp.
    
    Args:
        df: The frequency dictionary dataframe to save
    
    Returns:
        str: Full path to the saved file
    """
    # Set up directory paths
    hdir = os.path.expanduser('~')
    datasets_dir = hdir + "/Dropbox/Active_Directories/Digital_Humanities/Datasets/eurasia_corpus_frequency_dictionary"
    archived_dir = os.path.join(datasets_dir, "archived_eurasia_corpus_frequency_dictionary")
    
    # Create directories if they don't exist
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(archived_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filenames and paths
    latest_filename = "eurasia_corpus_frequency_dictionary.csv"
    archived_filename = f"eurasia_corpus_frequency_dictionary_{timestamp}.csv"
    
    latest_path = os.path.join(datasets_dir, latest_filename)
    archived_path = os.path.join(archived_dir, archived_filename)
    
    # Save both files
    df.to_csv(latest_path, index=False, encoding='utf-8')
    df.to_csv(archived_path, index=False, encoding='utf-8')
    
    print(f"Saved latest frequency dictionary to: {latest_path}")
    print(f"Saved archived frequency dictionary to: {archived_path}")
    return latest_path