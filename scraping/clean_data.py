"""
This module provides functions to normalize various date formats
into a consistent MM.YYYY format or 'NA' for unparseable dates.
"""

import re
import pandas as pd

# Month name to number mapping
MONTH_MAP = {
    'январь': '01', 'янв': '01',
    'февраль': '02', 'фев': '02',
    'март': '03', 'мар': '03',
    'апрель': '04', 'апр': '04',
    'май': '05', 'мая': '05',
    'июнь': '06', 'июн': '06',
    'июль': '07', 'июл': '07',
    'август': '08', 'авг': '08',
    'сентябрь': '09', 'сен': '09',
    'октябрь': '10', 'окт': '10',
    'ноябрь': '11', 'ноя': '11',
    'декабрь': '12', 'дек': '12'
}

# Patterns that indicate an unparseable date
INDETERMINATE_PATTERNS = re.compile(r'\?|\?\?|xx|хх|весна|лето|осень|зима')
YEAR_ONLY_PATTERN = re.compile(r'^20\d{2}$')
YEAR_RANGE_PATTERN = re.compile(r'^20\d{2}/20\d{2}$')

# Date patterns to try in order
DATE_PATTERNS = [
    r'(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>20\d{2})',  
    r'(?P<month>\d{1,2})\.(?P<year>20\d{2})',                     
    r'(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.\s*(?P<year>20\d{2})' 
]

def _clean_date_string(date_str: str):
    """Preprocesses a date returning a string in lowercase with prefixes removed."""
    date_str = date_str.strip().lower()
    return re.sub(r'^(не ранее|не позднее|до|не иранее)\s*', '', date_str)

def _is_indeterminate_date(date_str: str):
    """Checks if a date string represents an indeterminate date.    """
    if INDETERMINATE_PATTERNS.search(date_str):
        return not any(re.search(r'\d{1,2}\.\d{4}', p) for p in DATE_PATTERNS)
    return False

def _try_month_name_date(date_str: str):
    """Attempts to parse dates containing Russian month names.
    
    Args:
        date_str: Date string potentially containing month names
        
    Returns:
        Normalized date string or 'NA' if no match found
    """
    for month_name, month_num in MONTH_MAP.items():
        if month_name in date_str:
            year_match = re.search(r'20\d{2}', date_str)
            return f"{month_num}.{year_match.group()}" if year_match else 'NA'
    return 'NA'

def _try_pattern_matching(date_str: str) -> str:
    """Attempts to match date string against known patterns.
    
    Args:
        date_str: Cleaned date string to parse
        
    Returns:
        Normalized date string or 'NA' if no patterns match
    """
    date_str = re.sub(r'\s+', '', date_str)  # Remove all whitespace
    date_str = re.sub(r'(\d)\?', r'\1', date_str)  # Clean question marks in numbers
    
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, date_str)
        if match:
            month = match.group('month').zfill(2)
            year = match.group('year')
            return f"{month}.{year}"
    
    return 'NA'

def normalize_date(date_str: str):
    """Normalizes a Russian date string to MM.YYYY format or 'NA'.
    
    Handles various formats including:
    - Dates with prefixes ("не ранее 24.05.2024")
    - Month names ("март 2023")
    - Different separators (., /, -)
    - Incomplete dates ("??.04.2022")
    - Year-only ("2022")
    
    Args:
        date_str: Raw date string to normalize
        
    Returns:
        Normalized date in MM.YYYY format or 'NA' if unparseable
    """
    if not date_str:
        return 'NA'
    
    date_str = _clean_date_string(date_str)
    
    # Check for special cases first
    if _is_indeterminate_date(date_str):
        return 'NA'
    
    if YEAR_ONLY_PATTERN.fullmatch(date_str) or YEAR_RANGE_PATTERN.fullmatch(date_str):
        return 'NA'
    
    # Handle date ranges by taking first part
    if any(sep in date_str for sep in ['/', '-', '–']):
        date_str = re.split(r'[/–-]', date_str)[0]
    
    # Try month name parsing
    month_name_result = _try_month_name_date(date_str)
    if month_name_result != 'NA':
        return month_name_result
    
    # Try pattern matching
    return _try_pattern_matching(date_str)

if __name__ == "__main__":
    df = pd.read_csv("../data/daily.csv", encoding='utf-8-sig')
    if 'death_date' in df.columns:
        df['death_month'] = df['death_date'].apply(normalize_date)
        df.to_csv("../data/daily.csv", index=False)
        print("Successfully recoded date to MM.YYYY format")
