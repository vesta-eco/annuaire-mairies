#!/usr/bin/env python3
"""
Extract all internal links that are subdomains of geosphere.fr from CSV backlinks files.

This script processes CSV files containing backlink data and extracts all URLs 
that match the pattern: *.geosphere.fr

Usage:
    python extract_geosphere_subdomains.py
"""

import csv
import re
import os
from typing import Set, List
from urllib.parse import urlparse


def is_geosphere_subdomain(url: str) -> bool:
    """
    Check if a URL is a geosphere.fr subdomain.
    
    Args:
        url: The URL to check
        
    Returns:
        True if the URL is a geosphere.fr subdomain, False otherwise
    """
    if not url or not isinstance(url, str):
        return False
    
    try:
        # Handle URLs with or without protocol
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check if it's exactly geosphere.fr or a subdomain of geosphere.fr
        return domain == 'geosphere.fr' or domain.endswith('.geosphere.fr')
    
    except Exception:
        return False


def extract_urls_from_text(text: str) -> List[str]:
    """
    Extract URLs from text using regex patterns.
    
    Args:
        text: Text that may contain URLs
        
    Returns:
        List of URLs found in the text
    """
    if not text:
        return []
    
    # Pattern to match URLs (with or without protocol)
    url_pattern = r'(?:https?://)?(?:[\w-]+\.)*geosphere\.fr(?:/[^\s,]*)?'
    
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    return [url.rstrip(',') for url in urls if url]


def process_csv_file(file_path: str) -> Set[str]:
    """
    Process a single CSV file and extract geosphere.fr subdomain URLs.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Set of unique geosphere.fr subdomain URLs
    """
    geosphere_urls = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            # Try to detect if it's a proper CSV with headers
            sample = file.read(1024)
            file.seek(0)
            
            # Use csv.Sniffer to detect delimiter
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
                reader = csv.reader(file, dialect)
            except:
                # Fallback to comma delimiter
                reader = csv.reader(file, delimiter=',')
            
            for row_num, row in enumerate(reader, 1):
                # Process each cell in the row
                for cell in row:
                    if cell and isinstance(cell, str):
                        # Check if the cell itself is a geosphere URL
                        if is_geosphere_subdomain(cell):
                            geosphere_urls.add(cell.strip())
                        
                        # Also extract URLs from within the cell text
                        found_urls = extract_urls_from_text(cell)
                        for url in found_urls:
                            if is_geosphere_subdomain(url):
                                geosphere_urls.add(url.strip())
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return geosphere_urls


def find_csv_files(directory: str) -> List[str]:
    """
    Find all CSV files in a directory and its subdirectories.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of CSV file paths
    """
    csv_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    return csv_files


def clean_and_normalize_url(url: str) -> str:
    """
    Clean and normalize a URL.
    
    Args:
        url: Raw URL string
        
    Returns:
        Cleaned and normalized URL
    """
    if not url:
        return ""
    
    # Remove leading/trailing whitespace
    url = url.strip()
    
    # Add https:// if no protocol is specified
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Remove trailing slashes and fragments for consistency
    parsed = urlparse(url)
    
    # Reconstruct URL without fragment
    clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    
    # Add query parameters if they exist
    if parsed.query:
        clean_url += f"?{parsed.query}"
    
    # Remove trailing slash unless it's the root
    if clean_url.endswith('/') and len(parsed.path) > 1:
        clean_url = clean_url.rstrip('/')
    
    return clean_url


def main():
    """Main function to extract geosphere.fr subdomain links."""
    print("🔍 Extracting geosphere.fr subdomain links from CSV files...")
    
    # Find all CSV files in current directory and subdirectories
    current_dir = os.getcwd()
    csv_files = find_csv_files(current_dir)
    
    print(f"📁 Found {len(csv_files)} CSV files to process")
    
    all_geosphere_urls = set()
    
    # Process each CSV file
    for csv_file in csv_files:
        print(f"📄 Processing: {os.path.relpath(csv_file)}")
        urls = process_csv_file(csv_file)
        
        if urls:
            print(f"   ✅ Found {len(urls)} geosphere.fr URLs")
            all_geosphere_urls.update(urls)
        else:
            print(f"   ❌ No geosphere.fr URLs found")
    
    # Clean and normalize URLs
    cleaned_urls = set()
    for url in all_geosphere_urls:
        cleaned = clean_and_normalize_url(url)
        if cleaned:
            cleaned_urls.add(cleaned)
    
    # Sort URLs for better readability
    sorted_urls = sorted(cleaned_urls)
    
    # Write results to output file
    output_file = 'geosphere_subdomains.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Geosphere.fr Subdomain Links\n")
        f.write(f"# Extracted on {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total unique URLs found: {len(sorted_urls)}\n\n")
        
        for url in sorted_urls:
            f.write(f"{url}\n")
    
    print(f"\n✅ Extraction complete!")
    print(f"📊 Total unique geosphere.fr subdomain URLs found: {len(sorted_urls)}")
    print(f"💾 Results saved to: {output_file}")
    
    # Display some sample URLs
    if sorted_urls:
        print(f"\n📋 Sample URLs found:")
        for i, url in enumerate(sorted_urls[:10], 1):
            print(f"   {i}. {url}")
        
        if len(sorted_urls) > 10:
            print(f"   ... and {len(sorted_urls) - 10} more")


if __name__ == "__main__":
    main()