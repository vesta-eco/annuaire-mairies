#!/usr/bin/env python3
"""
Script to extract French cities from HTML select options and create a CSV file.

Usage:
    python extract_cities.py select-ads.txt
    python extract_cities.py input.html output.csv

Source for data : https://demarches.ideau.atreal.fr/urbanisme-nouvelle-demande/declaration-prealable-2025-01/?cancelurl=https%3A//ideau.atreal.fr/


"""

import re
import csv
import sys
import argparse
from pathlib import Path


def extract_cities_from_html(html_content):
    """
    Extract city names and INSEE codes from HTML select options.

    Args:
        html_content (str): HTML content containing select options

    Returns:
        list: List of dictionaries with 'commune' and 'code_insee' keys
    """
    # Pattern to match select option elements
    option_pattern = r'<li class="select2-results__option"[^>]*>(.*?)</li>'

    # Find all option elements
    option_matches = re.findall(option_pattern, html_content, re.DOTALL)

    cities = []

    for option_text in option_matches:
        # Skip the default "--" option
        if "--" in option_text:
            continue

        # Pattern to extract city name and INSEE code: "CityName (INSEE_CODE)"
        city_pattern = r"^(.+?)\s*\((\d+)\)$"
        city_match = re.search(city_pattern, option_text.strip())

        if city_match:
            commune = city_match.group(1).strip()
            code_insee = city_match.group(2)

            cities.append({"commune": commune, "code_insee": code_insee})

    return cities


def save_to_csv(cities, output_file):
    """
    Save cities data to CSV file.

    Args:
        cities (list): List of city dictionaries
        output_file (str): Path to output CSV file
    """
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["commune", "code_insee"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write city data
        for city in cities:
            writer.writerow(city)


def main():
    """Main function to handle command line arguments and process the file."""
    parser = argparse.ArgumentParser(
        description="Extract French cities from HTML select options and create CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python extract_cities.py select-ads.txt
    python extract_cities.py input.html output.csv
    python extract_cities.py data.html cities.csv
        """,
    )

    parser.add_argument("input_file", help="Input HTML file containing select options")
    parser.add_argument(
        "output_file",
        nargs="?",
        default="cities.csv",
        help="Output CSV file (default: cities.csv)",
    )

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    try:
        # Read the HTML file
        print(f"Reading HTML file: {args.input_file}")
        with open(input_path, "r", encoding="utf-8") as file:
            html_content = file.read()

        # Extract cities
        print("Extracting cities from HTML...")
        cities = extract_cities_from_html(html_content)

        if not cities:
            print("Warning: No cities found in the HTML content.")
            print("Please check if the file contains the expected HTML structure.")
            sys.exit(1)

        # Save to CSV
        print(f"Saving {len(cities)} cities to CSV: {args.output_file}")
        save_to_csv(cities, args.output_file)

        print(f"✅ Successfully extracted {len(cities)} cities!")
        print(f"📄 Output saved to: {args.output_file}")

        # Show first few entries as preview
        if cities:
            print("\n📋 Preview of first 5 cities:")
            for i, city in enumerate(cities[:5], 1):
                print(f"  {i}. {city['commune']} ({city['code_insee']})")

            if len(cities) > 5:
                print(f"  ... and {len(cities) - 5} more cities")

    except FileNotFoundError:
        print(f"Error: Could not read file '{args.input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
