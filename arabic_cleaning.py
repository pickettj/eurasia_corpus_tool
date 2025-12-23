#!/usr/bin/env python3
"""
Arabic/Persian Script Cleaning
Normalizes text to standard Persian orthography and removes non-Arabic script.
"""

import re

def clean_document(doc):
    """
    Clean and normalize Arabic/Persian text.
    
    Steps:
        1. Normalize variant characters (Arabic → Persian standard)
        2. Remove all non-Arabic script characters (including punctuation)
        3. Collapse whitespace
    
    Args:
        doc (str): Raw text potentially containing mixed scripts
    
    Returns:
        str: Cleaned text with only Persian/Arabic letters and numerals
    
    Note:
        Non-Arabic script text (English, Russian, etc.) will be 
        completely removed, leaving only spaces.
    """
    # Step 1: Normalize variant characters to Persian standard
    doc = re.sub(r'ي', 'ی', doc)          # Arabic yaa → Persian ye
    doc = re.sub(r'ى', 'ی', doc)          # Alef maksura → Persian ye
    doc = re.sub(r'أ', 'ا', doc)          # Hamza on alef → plain alef
    doc = re.sub(r'إ', 'ا', doc)          # Hamza below alef → plain alef
    doc = re.sub(r'آ', 'ا', doc)          # Alef with madda → plain alef (optional - comment out to keep آ)
    doc = re.sub(r'ك', 'ک', doc)          # Arabic kaf → Persian kaf
    doc = re.sub(r'ة', 'ه', doc)          # Taa marbouta → haa
    doc = re.sub(r'ۀ', 'ه', doc)          # Haa with hamza → plain haa
    doc = re.sub(r'ؤ', 'و', doc)          # Hamza on waw → plain waw
    doc = re.sub(r'ئ', 'ی', doc)          # Hamza on ye → plain ye
    
    # Specific word normalizations (add more as needed)
    doc = re.sub(r'مسئله', 'مساله', doc)
    
    # Step 2: Keep ONLY Persian/Arabic script characters
    # Letters + numerals, NO punctuation
    allowed_pattern = (
        r'[^'
        # Persian/Arabic letters (base alphabet)
        r'آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
        # Persian/Arabic numerals
        r'۰۱۲۳۴۵۶۷۸۹'
        # Whitespace
        r' '
        r']'
    )
    doc = re.sub(allowed_pattern, ' ', doc)
    
    # Step 3: Collapse multiple spaces and trim
    doc = re.sub(r'\s+', ' ', doc)
    doc = doc.strip()
    
    return doc


def has_arabic_script(text):
    """
    Check if text contains any Arabic/Persian characters.
    
    Args:
        text (str): Text to check
    
    Returns:
        bool: True if contains Arabic script, False otherwise
    
    Example:
        >>> has_arabic_script("Hello")
        False
        >>> has_arabic_script("سلام")
        True
        >>> has_arabic_script("Hello سلام")
        True
    """
    # Unicode ranges for Arabic/Persian script
    # \u0600-\u06FF: Arabic
    # \u0750-\u077F: Arabic Supplement
    # \uFB50-\uFDFF: Arabic Presentation Forms-A
    # \uFE70-\uFEFF: Arabic Presentation Forms-B
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]'
    return bool(re.search(arabic_pattern, text))