#!/usr/bin/env python3
"""
Combined Database & Corpus Research Tool
Loads both database query and corpus functions for integrated research
"""

import sys
import re
import sqlite3
import os
from pathlib import Path

# Add parent directory to path to access other libraries
projects_dir = Path(__file__).parent.parent
sys.path.insert(0, str(projects_dir / "database"))
sys.path.insert(0, str(projects_dir / "eurasia_corpus_tool"))

# Import the libraries
import database_query_functions as hdb
import historical_corpus_plain_text as hcorp

# Set database path (needed for combo_search to query database directly)
hdir = os.path.expanduser('~')
dh_path = '/Dropbox/Active_Directories/Digital_Humanities/'
database_path = os.path.join(hdir, dh_path.strip('/'), 'database_eurasia_7.0.db')

print("✅ Loaded Database Queries (hdb)")
print("✅ Loaded Eurasia Plain Text Corpus (hcorp)")
print("\n💡 Usage:")
print("  hdb.word_search('term')")
print("  hcorp.search_corpus('pattern')")



def _filter_corpus_by_serials(serial_list):
    """
    Internal function: Filter corpus dictionary to only include texts matching serial numbers.
    
    Args:
        serial_list (list): List of serial numbers to keep (e.g., [34, 45, 2037])
    
    Returns:
        dict: Filtered corpus dictionary with same structure as hcorp.search_corpus()
              Keys are filenames like 'ser2037.xml', values are match dictionaries
    
    Example:
        # If you have UIDs [2037, 2023] from bibliography
        filtered = _filter_corpus_by_serials([2037, 2023])
        # Returns only corpus entries for ser2037.xml and ser2023.xml
    """
    import re
    
    # Get full corpus dictionary
    full_corpus = hcorp._get_corpus()
    
    # Convert serial_list to set for faster lookup
    serial_set = set(serial_list)
    
    # Filter based on serial numbers in filenames
    filtered = {}
    for filename, data in full_corpus.items():
        # Extract serial number from filename (matches "ser" followed by digits)
        match = re.search(r'ser(\d+)', filename)
        if match:
            serial_num = int(match.group(1))
            if serial_num in serial_set:
                filtered[filename] = data
    
    return filtered


def search(corpus_term, database_term, database_filter=None, max_corpus_matches=50, save_markdown=False, return_data=False):
    """
    Search corpus for a term, filtered by bibliography criteria.
    Shows keyword-in-context results organized by source, with full bibliography metadata.
    
    Args:
        corpus_term (str): Regex pattern to search in corpus texts
        database_term (str or tuple): Search pattern(s) for bibliography (Author/Title/Gloss)
        database_filter (str or tuple, optional): Filter by repository attributes
        max_corpus_matches (int): Maximum corpus matches to display per document (default: 50)
        save_markdown (bool): If True, saves results as markdown to Inbox (default: False)
        return_data (bool): If True, returns combined_results dict (default: False)
    
    Returns:
        dict or None: Combined results if return_data=True, otherwise None
    
    Examples:
        combo_search('قاضی', 'Bukhara')
        # Searches for "qadi" in texts about Bukhara
        
        combo_search('تجارت', 'trade', 'manuscript', save_markdown=True)
        # Searches for "trade" in manuscript sources, saves to markdown
        
        results = combo_search('خان', 'Samarqand', return_data=True)
        # Returns data dict for further processing
    """
    print(f"🔍 COMBINED SEARCH")
    print("=" * 80)
    print(f"📜 Corpus term: '{corpus_term}'")
    print(f"📚 Database term: {database_term}")
    if database_filter:
        print(f"   Database filter: {database_filter}")
    print("=" * 80)
    
    # Step 1: Get matching bibliography UIDs
    print("\n📚 Step 1: Finding matching sources in database...")
    matched_uids = hdb._biblio_serials(database_term, database_filter)
    
    if not matched_uids:
        print("❌ No matching sources found in database")
        return None
    
    print(f"✅ Found {len(matched_uids)} matching sources")
    print(f"   Serial numbers: {matched_uids[:10]}{'...' if len(matched_uids) > 10 else ''}")
    
    # Step 2: Search corpus for the term
    print(f"\n📜 Step 2: Searching corpus for '{corpus_term}'...")
    corpus_results = hcorp.search_corpus(corpus_term)
    
    if not corpus_results:
        print(f"❌ No matches for '{corpus_term}' found in corpus")
        return None
    
    print(f"✅ Found matches in {len(corpus_results)} documents")
    
    # Step 3: Filter corpus results to only matched UIDs
    print(f"\n🔗 Step 3: Filtering corpus to matched sources...")
    filtered_corpus = _filter_corpus_by_serials(matched_uids)
    
    # Further filter to only documents that have the search term
    filtered_results = {k: v for k, v in corpus_results.items() if k in filtered_corpus}
    
    if not filtered_results:
        print(f"❌ No overlap: '{corpus_term}' not found in any of the {len(matched_uids)} matching sources")
        return None
    
    print(f"✅ Found '{corpus_term}' in {len(filtered_results)} of the matched sources")
    
    # Ask about markdown export if more than 10 results
    if len(filtered_results) > 10 and not save_markdown:
        response = input(f"\n💡 Found {len(filtered_results)} matching sources. Save as markdown? (y/n): ").strip().lower()
        if response == 'y':
            save_markdown = True
    
    # Step 4: Get bibliography metadata for matched documents
    print(f"\n📖 Step 4: Retrieving bibliography metadata...")
    
    # Extract serial numbers from filtered results
    result_serials = []
    for filename in filtered_results.keys():
        match = re.search(r'ser(\d+)', filename)
        if match:
            result_serials.append(int(match.group(1)))
    
    # Get bibliography details
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    try:
        placeholders = ','.join(['?' for _ in result_serials])
        cursor.execute(f"""
            SELECT b.UID, b.Author, b.Title, b.Gloss, b.Date_Pub_Greg, b.Date_Pub_Hij,
                   r.Acronym, r.Name_English, b.Catalog_No, b.Language, b.Status, b.Tags
            FROM bibliography b
            LEFT JOIN repositories r ON b.Repository_ID = r.UID
            WHERE b.UID IN ({placeholders})
            ORDER BY b.UID
        """, result_serials)
        
        bib_metadata = {row[0]: row for row in cursor.fetchall()}
        
    finally:
        cursor.close()
        conn.close()
    
    # Step 5: Format output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"📊 COMBINED SEARCH RESULTS")
    output_lines.append("=" * 80)
    output_lines.append(f"📜 Corpus term: '{corpus_term}'")
    output_lines.append(f"📚 Database term: {database_term}")
    if database_filter:
        output_lines.append(f"🔍 Database filter: {database_filter}")
    output_lines.append(f"✅ Found '{corpus_term}' in {len(filtered_results)} sources")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    total_matches = 0
    combined_results = {}
    
    for i, (filename, matches) in enumerate(filtered_results.items(), 1):
        # Extract serial number
        match = re.search(r'ser(\d+)', filename)
        if not match:
            continue
        serial = int(match.group(1))
        
        # Get bibliography metadata
        if serial in bib_metadata:
            uid, author, title, gloss, date_greg, date_hij, acronym, repo_name, catalog, language, status, tags = bib_metadata[serial]
            
            # Format source header
            output_lines.append(f"{i}. {author} - {title}")
            if gloss:
                output_lines.append(f"   📝 Gloss: {gloss}")
            output_lines.append(f"   🔑 UID: {uid}")
            if acronym:
                output_lines.append(f"   🏛️  Repository: {acronym}" + (f" ({repo_name})" if repo_name else ""))
            if catalog:
                output_lines.append(f"   📋 Catalog: {catalog}")
            if date_greg:
                date_str = f"   📅 Date: {date_greg}"
                if date_hij:
                    date_str += f" / {date_hij}"
                output_lines.append(date_str)
            if language:
                output_lines.append(f"   🌐 Language: {language}")
            if status:
                output_lines.append(f"   📊 Status: {status}")
            if tags:
                clean_tags = ', '.join(filter(None, tags.split()))
                output_lines.append(f"   🏷️  Tags: {clean_tags}")
            
            output_lines.append("")
            
            # Format matches
            match_count = 0
            for match_id, context in list(matches.items())[:max_corpus_matches]:
                match_count += 1
                total_matches += 1
                
                # Highlight the search term in context
                highlighted = re.sub(
                    f'({corpus_term})',
                    r'**\1**',
                    context,
                    flags=re.IGNORECASE
                )
                
                output_lines.append(f"   {highlighted}")
            
            if len(matches) > max_corpus_matches:
                output_lines.append(f"   ... and {len(matches) - max_corpus_matches} more matches")
            
            output_lines.append("")
            output_lines.append("─" * 80)
            output_lines.append("")
            
            # Store in results dict
            combined_results[serial] = {
                'filename': filename,
                'metadata': bib_metadata[serial],
                'matches': matches
            }
    
    # Final summary
    output_lines.append("=" * 80)
    output_lines.append(f"📊 FINAL SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append(f"✅ Found '{corpus_term}' {total_matches} times across {len(filtered_results)} sources")
    output_lines.append(f"📚 Database matched {len(matched_uids)} total sources")
    output_lines.append(f"📜 Corpus search found {len(corpus_results)} documents with term")
    output_lines.append(f"🔗 Overlap: {len(filtered_results)} sources")
    output_lines.append("=" * 80)
    
    # Output to console or file
    output_text = '\n'.join(output_lines)
    
    if save_markdown:
        # Generate filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create safe filename from search terms
        corpus_safe = re.sub(r'[^\w\s-]', '', corpus_term)[:30]
        db_safe = re.sub(r'[^\w\s-]', '', str(database_term))[:30]
        filename = f"combo_search_{corpus_safe}_{db_safe}_{timestamp}.md"
        
        # Use inbox path
        inbox_path = os.path.join(hdir, 'Dropbox/Active_Directories/Inbox')
        filepath = os.path.join(inbox_path, filename)
        
        # Convert to markdown format
        md_lines = []
        md_lines.append(f"# Combined Search Results\n")
        md_lines.append(f"**Corpus term:** `{corpus_term}`\n")
        md_lines.append(f"**Database term:** `{database_term}`\n")
        if database_filter:
            md_lines.append(f"**Database filter:** `{database_filter}`\n")
        md_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md_lines.append(f"**Results:** Found in {len(filtered_results)} sources\n")
        md_lines.append("---\n")
        
        for i, (filename, matches) in enumerate(filtered_results.items(), 1):
            match = re.search(r'ser(\d+)', filename)
            if not match:
                continue
            serial = int(match.group(1))
            
            if serial in bib_metadata:
                uid, author, title, gloss, date_greg, date_hij, acronym, repo_name, catalog, language, status, tags = bib_metadata[serial]
                
                md_lines.append(f"\n## {i}. {author} - {title}\n")
                if gloss:
                    md_lines.append(f"- **Gloss:** {gloss}\n")
                md_lines.append(f"- **UID:** {uid}\n")
                if acronym:
                    md_lines.append(f"- **Repository:** {acronym}" + (f" ({repo_name})" if repo_name else "") + "\n")
                if catalog:
                    md_lines.append(f"- **Catalog:** {catalog}\n")
                if date_greg:
                    date_str = f"- **Date:** {date_greg}"
                    if date_hij:
                        date_str += f" / {date_hij}"
                    md_lines.append(date_str + "\n")
                if language:
                    md_lines.append(f"- **Language:** {language}\n")
                if status:
                    md_lines.append(f"- **Status:** {status}\n")
                if tags:
                    clean_tags = ', '.join(filter(None, tags.split()))
                    md_lines.append(f"- **Tags:** {clean_tags}\n")
                
                md_lines.append("\n### Matches:\n")
                
                for match_id, context in list(matches.items())[:max_corpus_matches]:
                    highlighted = re.sub(
                        f'({corpus_term})',
                        r'**\1**',
                        context,
                        flags=re.IGNORECASE
                    )
                    md_lines.append(f"{highlighted}\n\n")
                
                if len(matches) > max_corpus_matches:
                    md_lines.append(f"*... and {len(matches) - max_corpus_matches} more matches*\n")
                
                md_lines.append("\n---\n")
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(md_lines)
        
        print(f"\n✅ Markdown report saved to: {filepath}")
    else:
        # Print to console
        print("\n" + output_text)
    
    # Return data only if requested
    if return_data:
        return combined_results