#!/usr/bin/env python3
"""
Tajik Newspaper Corpus Analysis Tool
"""

import pandas as pd
import re
import os
from datetime import datetime

# Set home directory path
hdir = os.path.expanduser('~')

# Define paths
dh_path = '/Dropbox/Active_Directories/Digital_Humanities/'
corpus_path = os.path.join(hdir, dh_path.strip('/'), 'Corpora/tajik_newspaper_corpus/tajik_newspaper_corpus.csv')
inbox_path = os.path.join(hdir, 'Dropbox/Active_Directories/Inbox')

# Check if corpus file exists
if not os.path.exists(corpus_path):
    raise FileNotFoundError(f"Corpus file not found at: {corpus_path}")

# Load the corpus
print("📰 Loading Tajik newspaper corpus...")
df = pd.read_csv(corpus_path, encoding='utf-8')
# Rename sub_directory to newspaper
df.rename(columns={'sub_directory': 'newspaper'}, inplace=True)
print(f"✅ Loaded {len(df):,} articles")

# Display basic info
print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nAvailable newspapers:")
for newspaper, count in df['newspaper'].value_counts().items():
    print(f"  • {newspaper}: {count:,} articles")


def search_corpus(pattern, newspaper=None, case_sensitive=False):
    """
    Search the corpus using regex and return filtered dataframe.
    
    Args:
        pattern: Regex pattern to search for
        newspaper: Optional newspaper name to filter by (exact match)
        case_sensitive: Whether search should be case-sensitive (default: False)
    
    Returns:
        DataFrame: Filtered dataframe containing only rows where content matches the pattern
    """
    # Start with full dataframe or filter by newspaper
    working_df = df if newspaper is None else df[df['newspaper'] == newspaper]
    
    if newspaper:
        print(f"Searching in {newspaper}: {len(working_df):,} articles")
    
    # Set regex flags
    flags = 0 if case_sensitive else re.IGNORECASE
    
    # Filter rows where content matches the pattern
    mask = working_df['content'].str.contains(pattern, regex=True, na=False, flags=flags)
    filtered_df = working_df[mask].copy()
    
    print(f"Found {len(filtered_df):,} articles matching pattern (out of {len(working_df):,} total)")
    print(f"Match rate: {len(filtered_df)/len(working_df)*100:.2f}%")
    
    return filtered_df


def search_report(pattern, newspaper=None, case_sensitive=False, max_results=None):
    """
    Generate a markdown report of search results with highlighted matches.
    
    Args:
        pattern: Regex pattern to search for
        newspaper: Optional newspaper name to filter by
        case_sensitive: Whether search should be case-sensitive (default: False)
        max_results: Maximum number of results to include (default: None = all results)
    
    Returns:
        str: Path to the generated markdown file
    """
    # Get search results
    results_df = search_corpus(pattern, newspaper, case_sensitive)
    
    if len(results_df) == 0:
        print("No results found. No report generated.")
        return None
    
    # Sample if max_results specified
    if max_results and len(results_df) > max_results:
        print(f"📊 Randomly sampling {max_results} results from {len(results_df):,} total matches")
        results_df = results_df.sample(n=max_results, random_state=None)  # random_state=None for true randomness
    
    # Generate timestamp and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"tajik_search_report_{timestamp}.md"
    
    # Use global inbox_path
    full_path = os.path.join(inbox_path, report_filename)
    
    # Create markdown content
    markdown_content = f"# Tajik Newspaper Corpus Search Report\n\n"
    markdown_content += f"**Search Pattern:** `{pattern}`\n\n"
    if newspaper:
        markdown_content += f"**Newspaper:** {newspaper}\n\n"
    markdown_content += f"**Total Matches:** {len(results_df):,}"
    if max_results and len(results_df) == max_results:
        markdown_content += f" (randomly sampled)\n\n"
    else:
        markdown_content += "\n\n"
    markdown_content += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown_content += "---\n\n"
    
    # Set regex flags for highlighting
    flags = re.IGNORECASE if not case_sensitive else 0
    
    # Process each result
    for idx, row in results_df.iterrows():
        # Split content into sentences
        sentences = re.split(r'([.!?։]\s+)', row['content'])
        
        # Find sentences containing the pattern
        for i in range(0, len(sentences)-1, 2):  # Step by 2 to handle sentence + delimiter pairs
            sentence = sentences[i]
            delimiter = sentences[i+1] if i+1 < len(sentences) else ''
            
            if re.search(pattern, sentence, flags=flags):
                # Add context sentences if requested
                full_sentence = sentence + delimiter
                
                # Highlight all matches in the sentence with bold
                highlighted = re.sub(
                    f'({pattern})',
                    r'**\1**',
                    full_sentence,
                    flags=flags
                )
                
                # Add to markdown
                markdown_content += f"### {row['newspaper']} - {row['filename']}\n\n"
                markdown_content += f"{highlighted.strip()}\n\n"
                markdown_content += "---\n\n"
    
    # Write to file
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\n✅ Report saved to: {full_path}")
    return full_path


def get_newspapers():
    """
    Get list of all newspapers with article counts.
    
    Returns:
        Series: Newspaper names with article counts
    """
    return df['newspaper'].value_counts()