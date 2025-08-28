#!/usr/bin/env python3

import pandas as pd
import json
import sys
from pathlib import Path


def load_epci_mapping():
    communes_path = "sources/communes-france-2025.csv"

    if not Path(communes_path).exists():
        print(f"Warning: {communes_path} not found. EPCI data will be empty.")
        return {}

    print(f"Loading EPCI data from {communes_path}...")
    df = pd.read_csv(communes_path, skiprows=2, low_memory=False)

    mapping = {}
    for _, row in df.iterrows():
        insee_code = row.get("code_insee")
        epci_code = row.get("epci_code")
        epci_name = row.get("epci_nom")

        if insee_code and epci_code:
            mapping[str(insee_code)] = {
                "epci_code": str(epci_code),
                "epci_name": str(epci_name) if epci_name else "",
            }

    print(f"Loaded EPCI mapping for {len(mapping)} communes")
    return mapping


def parse_pivot_data(pivot_field):
    try:
        return json.loads(pivot_field)
    except (json.JSONDecodeError, TypeError):
        return []


def get_matching_service(pivot_data, service_types):
    for service in pivot_data:
        if (
            isinstance(service, dict)
            and service.get("type_service_local") in service_types
        ):
            return service
    return None


def add_service_info(row, service, epci_mapping):
    enhanced_row = row.copy()
    enhanced_row["extracted_service_type"] = service.get("type_service_local")

    insee_codes = service.get("code_insee_commune", [])
    enhanced_row["insee_codes_served"] = ",".join(insee_codes)
    enhanced_row["num_communes_served"] = len(insee_codes)

    if insee_codes and epci_mapping:
        first_insee = str(insee_codes[0])
        epci_info = epci_mapping.get(first_insee, {})
        enhanced_row["epci_code"] = epci_info.get("epci_code", "")
        enhanced_row["epci_name"] = epci_info.get("epci_name", "")
    else:
        enhanced_row["epci_code"] = ""
        enhanced_row["epci_name"] = ""

    return enhanced_row


def extract_local_services(
    csv_path, output_path=None, service_types=["mairie", "epci"]
):
    print(f"Loading CSV file: {csv_path}")

    epci_mapping = load_epci_mapping()

    try:
        df = pd.read_csv(csv_path, sep=";", dtype=str, na_filter=False)
        print(f"Loaded {len(df)} total records")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    filtered_records = []
    print(f"Filtering for service types: {service_types}")

    for record_num, (idx, row) in enumerate(df.iterrows(), 1):
        if record_num % 10000 == 0:
            print(f"Processing record {record_num}...")

        pivot_field = row.get("pivot", "")
        pivot_data = parse_pivot_data(pivot_field)

        if not pivot_data:
            continue

        matching_service = get_matching_service(pivot_data, service_types)
        if matching_service:
            enhanced_row = add_service_info(row, matching_service, epci_mapping)
            filtered_records.append(enhanced_row)

    if not filtered_records:
        print("No matching records found!")
        return pd.DataFrame()

    filtered_df = pd.DataFrame(filtered_records)

    if output_path:
        filtered_df.to_csv(output_path, sep=";", index=False)
        print(f"Saved {len(filtered_df)} records to {output_path}")

    return filtered_df


def parse_json_field(field_value):
    try:
        return json.loads(field_value)
    except (json.JSONDecodeError, TypeError):
        return []


def extract_websites(site_internet_field):
    sites = parse_json_field(site_internet_field)
    return [
        site["valeur"]
        for site in sites
        if isinstance(site, dict) and site.get("valeur")
    ]


def extract_phones(telephone_field):
    phones = parse_json_field(telephone_field)
    return [
        phone["valeur"]
        for phone in phones
        if isinstance(phone, dict) and phone.get("valeur")
    ]


def extract_addresses(adresse_field):
    addresses = parse_json_field(adresse_field)
    return [
        addr["valeur"]
        for addr in addresses
        if isinstance(addr, dict) and addr.get("valeur")
    ]


def create_contact_record(row):
    return {
        "nom": row.get("nom", ""),
        "service_type": row.get("extracted_service_type", ""),
        "insee_codes": row.get("insee_codes_served", ""),
        "num_communes": row.get("num_communes_served", 0),
        "epci_code": row.get("epci_code", ""),
        "epci_name": row.get("epci_name", ""),
        "email": row.get("adresse_courriel", ""),
        "websites": extract_websites(row.get("site_internet")),
        "phones": extract_phones(row.get("telephone")),
        "addresses": extract_addresses(row.get("adresse")),
    }


def extract_contact_info(df):
    contact_info = []

    for idx, row in df.iterrows():
        contact_record = create_contact_record(row)

        contact_record["websites"] = ";".join(contact_record["websites"])
        contact_record["phones"] = ";".join(contact_record["phones"])
        contact_record["addresses"] = ";".join(contact_record["addresses"])

        contact_info.append(contact_record)

    return pd.DataFrame(contact_info)


def print_statistics(filtered_df):
    print(f"Total records extracted: {len(filtered_df)}")

    service_stats = filtered_df["extracted_service_type"].value_counts()
    for service_type, count in service_stats.items():
        print(f"{service_type.upper()}: {count} records")


def print_sample_records(filtered_df):
    for service_type in ["mairie", "epci"]:
        sample = filtered_df[
            filtered_df["extracted_service_type"] == service_type
        ].head(3)
        if len(sample) > 0:
            print(f"\nSample {service_type.upper()}s:")
            for idx, row in sample.iterrows():
                print(
                    f"  - {row['nom']} (serves {row['num_communes_served']} communes)"
                )


def main():
    csv_path = "sources/annuaire_administrations.csv"
    output_full_path = "mairies_epcis_full.csv"
    output_contact_path = "mairies_epcis_contacts.csv"

    if not Path(csv_path).exists():
        print(f"Error: Input file not found: {csv_path}")
        sys.exit(1)

    print("=== Extracting Mairies and EPCIs ===")
    filtered_df = extract_local_services(csv_path, output_full_path, ["mairie", "epci"])

    if filtered_df is None or len(filtered_df) == 0:
        print("No records extracted. Exiting.")
        sys.exit(1)

    print("\n=== Creating Contact Information Summary ===")
    contact_df = extract_contact_info(filtered_df)
    contact_df.to_csv(output_contact_path, sep=";", index=False)
    print(f"Contact summary saved to: {output_contact_path}")

    print("\n=== Summary Statistics ===")
    print_statistics(filtered_df)

    print("\n=== Sample Records ===")
    print_sample_records(filtered_df)

    print(f"\nFiles created:")
    print(f"  - {output_full_path} (complete data)")
    print(f"  - {output_contact_path} (contact information)")


if __name__ == "__main__":
    main()
