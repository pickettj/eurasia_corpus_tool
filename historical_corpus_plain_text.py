#!/usr/bin/env python3
"""
Read in Plain Text Persianate Eurasia Historical Corpus
"""

"""
Importing libraries, setting up paths
"""

import pandas as pd
import arabic_cleaning as ac
import os, re
import xml.etree.ElementTree as ET
from datetime import datetime

# Global variables to cache the data
_text_corpus = None
_clean_corpus = None

#set home directory path
hdir = os.path.expanduser('~')

#notes directory
notes_dir = hdir + "/Dropbox/Active_Directories/Notes/Primary_Sources"

#plain text and markdown
txt_notes_dir = notes_dir + "/non-machine-readable_notes"

#parser xml transcriptions
parser_xml_dir = notes_dir + "/xml_notes_stage2"

#finished xml transcriptions
finished_xml_dir = notes_dir + "/xml_notes_stage3"

"""
Dependent / Background Functions
"""

def _read_text_and_md_files(directory, existing_data=None):
    text_files = {}
    
    for root, dirs, files in os.walk(directory):  # Use os.walk for recursive directory traversal
        for filename in files:
            if filename.endswith('.txt') or filename.endswith('.md'):  # Check for both text and markdown files
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    text_files[filename] = file.read()  # Use filename as key and file content as value

    if existing_data is not None:
        # Combine the two dictionaries
        text_files.update(existing_data)

    return text_files

def _read_xml_files_as_text(directory, existing_data=None):
    text_files = {}
    
    for root, dirs, files in os.walk(directory):  # Use os.walk for recursive directory traversal
        for filename in files:
            if filename.endswith('.xml'):  # Check for XML files
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    xml_content = file.read()
                    # Parse the XML and extract text
                    try:
                        root_element = ET.fromstring(xml_content)
                        plain_text = ''.join(root_element.itertext())  # Get all text from the XML
                        text_files[filename] = plain_text  # Use filename as key and plain text as value
                    except ET.ParseError:
                        print(f"Error parsing {filename}. Skipping this file.")

    if existing_data is not None:
        # Combine the two dictionaries
        text_files.update(existing_data)

    return text_files

def _clean_dictionary_values(data_dict):
    cleaned_dict = {}
    for key, value in data_dict.items():
        cleaned_dict[key] = ac.clean_document(value)  # Clean the document for each value
    return cleaned_dict

"""
Public API Functions
"""

def _get_corpus(clean=True):
    """
    Internal function to get the text corpus dictionary.
    Not meant to be called directly - use search_corpus() instead.
    
    Args:
        clean: If True, return cleaned text; if False, return raw text
    
    Returns:
        dict: Dictionary of filename -> text content
    """
    global _text_corpus, _clean_corpus
    
    # Load data if not already cached
    if _text_corpus is None:
        print("📚 Loading corpus for the first time...")
        text_notes = _read_text_and_md_files(txt_notes_dir)
        cent_texts_prelim = _read_xml_files_as_text(parser_xml_dir, text_notes)
        _text_corpus = _read_xml_files_as_text(finished_xml_dir, cent_texts_prelim)
        print(f"✅ Loaded {len(_text_corpus)} documents")
    
    if clean and _clean_corpus is None:
        print("🧹 Cleaning Arabic script...")
        _clean_corpus = _clean_dictionary_values(_text_corpus)
        print("✅ Cleaning complete")
    
    return _clean_corpus if clean else _text_corpus

def search_corpus(regex_pattern, additional_chars=30, use_clean=True):
    """
    Search the corpus with regex and return matches with context.
    
    Args:
        regex_pattern: Regex pattern to search for
        additional_chars: Characters of context around matches
        use_clean: Whether to search cleaned or raw text
    
    Returns:
        dict: Search results with context
    """
    corpus = _get_corpus(clean=use_clean)
    return _reg_text_search(corpus, regex_pattern, additional_chars)

def corpus_info():
    """
    Display statistics about the corpus without printing all content.
    
    Returns:
        None (prints formatted statistics)
    """
    corpus = _get_corpus(clean=False)
    
    print("\n" + "=" * 70)
    print("📚 CORPUS STATISTICS")
    print("=" * 70)
    print(f"Total documents: {len(corpus)}")
    
    # Count file types
    txt_count = sum(1 for k in corpus.keys() if k.endswith('.txt'))
    md_count = sum(1 for k in corpus.keys() if k.endswith('.md'))
    xml_count = sum(1 for k in corpus.keys() if k.endswith('.xml'))
    
    print(f"\nDocument types:")
    print(f"  • .txt files: {txt_count}")
    print(f"  • .md files: {md_count}")
    print(f"  • .xml files: {xml_count}")
    
    # Calculate total character count
    total_chars = sum(len(text) for text in corpus.values())
    print(f"\nTotal characters: {total_chars:,}")
    print(f"Average document length: {total_chars // len(corpus):,} characters")
    
    # Show sample filenames
    print(f"\nSample documents (first 10):")
    for i, filename in enumerate(list(corpus.keys())[:10], 1):
        doc_length = len(corpus[filename])
        print(f"  {i:2d}. {filename:50s} ({doc_length:,} chars)")
    
    if len(corpus) > 10:
        print(f"  ... and {len(corpus) - 10} more documents")
    
    print("=" * 70)
    print("💡 Use search_corpus(pattern) to search within documents")
    print("=" * 70 + "\n")

"""
Utility Functions
"""

def _reg_text_search(data_dict, regex_pattern, additional_chars=30):
    """
    Find regex matches in dictionary values and return them with surrounding context.
    
    Args:
        data_dict: Dictionary with string values to search
        regex_pattern: Regex pattern to search for
        additional_chars: Number of characters to include before/after match (default: 30)
    
    Returns:
        Nested dict: {original_key: {reg_match_no1: context_text, ...}}
    """
    results = {}
    
    for key, value in data_dict.items():
        matches = re.finditer(regex_pattern, value)  # Find all matches of the regex pattern
        match_count = {}  # To keep track of how many times each match has occurred
        
        for match in matches:
            matched_value = match.group()  # Get the matched string
            match_count[matched_value] = match_count.get(matched_value, 0) + 1  # Increment count
            
            start_index = max(match.start() - additional_chars, 0)  # Ensure we don't go below 0
            end_index = match.end() + additional_chars  # Get the end index for slicing
            matched_text = value[start_index:end_index]  # Extract the matched text with context
            
            # Create a new key for the results
            ordinal = match_count[matched_value]  # Get the current count for this match
            result_key = f"reg_{matched_value}_no{ordinal}"
            
            # Initialize the nested dictionary for the original key if it doesn't exist
            if key not in results:
                results[key] = {}
            
            results[key][result_key] = matched_text  # Store the matched text in the nested dictionary

    return results

def md_report(matches_dict, inbox_path=None):
    """
    Generate a timestamped Markdown report from regex search results.
    
    Args:
        matches_dict: Nested dict from _reg_text_search with match results
        inbox_path: Directory to save report (default: current directory)
    
    Returns:
        str: Full path to the generated report file
    """
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create a standard filename with the timestamp
    report_file_name = f"regex_search_report[{timestamp}].md"
    
    # If inbox_path is not provided, use the current directory
    if inbox_path is None:
        inbox_path = os.getcwd()  # Default to the current working directory
    
    # Combine the directory and file name to create the full path
    full_path = os.path.join(inbox_path, report_file_name)

    # Create a Markdown string to hold the report
    markdown_content = ""

    # Iterate through the top-level keys in the dictionary
    for file_name, matches in matches_dict.items():
        # Add a section for each file
        markdown_content += f"### {file_name}\n\n"
        
        # Iterate through the matches in the inner dictionary
        for match_key, context in matches.items():
            # Extract the regex match from the match_key (handles underscores in regex)
            parts = match_key.split('_')
            regex_match = '_'.join(parts[1:-1])  # Get all parts except 'reg' and 'no{X}'
            
            # Add the regex match and context to the markdown content
            markdown_content += f"**Regex Match:** {regex_match}\n"
            markdown_content += f"**Context:** {context}\n\n"

    # Save the markdown content to a file
    with open(full_path, 'w', encoding='utf-8') as markdown_file:
        markdown_file.write(markdown_content)
    
    return full_path