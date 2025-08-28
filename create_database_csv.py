#!/usr/bin/env python3

import pandas as pd
import re
import logging
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def normalize_insee_code(code):
    if pd.isna(code):
        return ""
    cleaned = re.sub(r"\D", "", str(code).strip().rstrip(".0"))
    return cleaned.lstrip("0") or "0"


def extract_primary_website(websites_str):
    if pd.isna(websites_str):
        return ""
    urls = re.split(r"[,;|\n]", str(websites_str))
    for url in urls:
        url = url.strip()
        if url and not any(
            kw in url.lower() for kw in ["urbanisme", "demarches", "guichet", "portail"]
        ):
            return url if url.startswith(("http://", "https://")) else f"https://{url}"
    return ""


def filter_valid_phone_numbers(phones_str):
    """Filter phone numbers to keep only those with 10 or more digits"""
    if pd.isna(phones_str) or not phones_str:
        return ""

    # Split by semicolon to handle multiple phone numbers
    phones = [phone.strip() for phone in str(phones_str).split(";")]
    valid_phones = []

    for phone in phones:
        if phone:
            # Count only digits in the phone number
            digit_count = len(re.sub(r"\D", "", phone))
            if digit_count >= 10:
                valid_phones.append(phone)

    return ";".join(valid_phones) if valid_phones else ""


def normalize_commune_name(name):
    if pd.isna(name):
        return ""
    return str(name).replace("Mairie - ", "").strip()


def has_valid_admin_content(url):
    if pd.isna(url) or not url:
        return False

    url_lower = url.lower()

    if any(keyword in url_lower for keyword in ["plan", "plu"]):
        return False

    valid_keywords = ["urbanisme", "guichet", "droit", "declaration", "demande", "sve"]
    return any(keyword in url_lower for keyword in valid_keywords)


def has_meaningful_url_structure(url):
    if pd.isna(url) or not url:
        return False

    try:
        parsed = urlparse(url)
        domain_parts = parsed.netloc.split(".")
        has_subdomain = len(domain_parts) > 2
        has_folder = len(parsed.path.strip("/")) > 0
        return has_subdomain or has_folder
    except:
        return False


def load_source_data():
    urban_portals = pd.read_csv("urban_planning_portals.csv")
    municipal_contacts = pd.read_csv("mairies_epcis_contacts.csv", sep=";", dtype=str)
    return urban_portals, municipal_contacts


def extract_validated_portals(urban_portals):
    validated_portals = urban_portals[urban_portals["ai_selected"] == True].copy()
    validated_portals["normalized_entity_code"] = validated_portals[
        "entity_code"
    ].apply(normalize_insee_code)

    mairie_portals = validated_portals[
        validated_portals["service_type"] == "mairie"
    ].copy()
    epci_portals = validated_portals[validated_portals["service_type"] == "epci"].copy()

    if not mairie_portals.empty:
        best_mairie_portals = mairie_portals.loc[
            mairie_portals.groupby("normalized_entity_code")["ai_confidence"].idxmax()
        ][["normalized_entity_code", "url"]].rename(
            columns={"url": "admin_website_url"}
        )
    else:
        best_mairie_portals = pd.DataFrame(
            columns=["normalized_entity_code", "admin_website_url"]
        )

    if not epci_portals.empty:
        best_epci_portals = epci_portals.loc[
            epci_portals.groupby("normalized_entity_code")["ai_confidence"].idxmax()
        ][["normalized_entity_code", "url"]].rename(
            columns={"url": "epci_admin_website_url"}
        )
    else:
        best_epci_portals = pd.DataFrame(
            columns=["normalized_entity_code", "epci_admin_website_url"]
        )

    return best_mairie_portals, best_epci_portals


def prepare_mairie_records(municipal_contacts):
    mairies = municipal_contacts[municipal_contacts["service_type"] == "mairie"].copy()
    mairies["normalized_insee_code"] = mairies["insee_codes"].apply(
        normalize_insee_code
    )
    mairies["commune"] = mairies["nom"].apply(normalize_commune_name)
    mairies["url"] = mairies["websites"].apply(extract_primary_website)
    mairies["phones"] = mairies["phones"].apply(filter_valid_phone_numbers)
    # Use the dedicated postal_code column instead of extracting from address
    mairies["code_postal"] = mairies["postal_code"]
    mairies["normalized_epci_code"] = mairies["epci_code"].apply(normalize_insee_code)

    return mairies[
        [
            "normalized_insee_code",
            "commune",
            "epci_name",
            "normalized_epci_code",
            "url",
            "email",
            "phones",
            "address",
            "code_postal",
        ]
    ].rename(
        columns={
            "normalized_insee_code": "code_insee",
            "epci_name": "epci",
            "address": "adresse",
        }
    )


def prepare_epci_websites(municipal_contacts):
    epcis = municipal_contacts[municipal_contacts["service_type"] == "epci"].copy()
    epcis["normalized_epci_code"] = epcis["epci_code"].apply(normalize_insee_code)
    epcis["epci_url"] = epcis["websites"].apply(extract_primary_website)

    # Deduplicate by EPCI code, keeping the first occurrence
    result = epcis[["normalized_epci_code", "epci_url"]].drop_duplicates(
        subset=["normalized_epci_code"], keep="first"
    )
    logging.info(
        f"Deduplicated EPCI websites from {len(epcis)} to {len(result)} records"
    )
    return result


def build_final_dataset():
    logging.info("Loading source data...")
    urban_portals, municipal_contacts = load_source_data()
    logging.info(
        f"Loaded {len(urban_portals):,} urban portals and {len(municipal_contacts):,} municipal contacts"
    )

    logging.info("Extracting validated portals...")
    mairie_portals, epci_portals = extract_validated_portals(urban_portals)
    logging.info(
        f"Extracted {len(mairie_portals):,} mairie portals and {len(epci_portals):,} EPCI portals"
    )

    logging.info("Preparing mairie records...")
    mairies = prepare_mairie_records(municipal_contacts)
    logging.info(f"Prepared {len(mairies):,} mairie records")

    logging.info("Preparing EPCI websites...")
    epci_websites = prepare_epci_websites(municipal_contacts)
    logging.info(f"Prepared {len(epci_websites):,} EPCI websites")

    logging.info("Merging mairie portals...")
    result = mairies.merge(
        mairie_portals,
        left_on="code_insee",
        right_on="normalized_entity_code",
        how="left",
    )
    logging.info(f"After mairie portals merge: {len(result):,} records")

    logging.info("Merging EPCI websites...")
    result = result.merge(epci_websites, on="normalized_epci_code", how="left")
    logging.info(f"After EPCI websites merge: {len(result):,} records")

    logging.info("Merging EPCI portals...")
    result = result.merge(
        epci_portals,
        left_on="normalized_epci_code",
        right_on="normalized_entity_code",
        how="left",
        suffixes=("", "_epci"),
    )
    logging.info(f"After EPCI portals merge: {len(result):,} records")

    logging.info("Selecting final columns...")
    result = result[
        [
            "code_insee",
            "commune",
            "epci",
            "epci_url",
            "epci_admin_website_url",
            "admin_website_url",
            "url",
            "email",
            "phones",
            "adresse",
            "code_postal",
        ]
    ].copy()
    logging.info(f"Selected columns, final dataset: {len(result):,} records")

    logging.info("Validating admin website URLs...")
    result.loc[
        ~result["admin_website_url"].apply(has_valid_admin_content), "admin_website_url"
    ] = None
    logging.info("Validating EPCI admin website URLs...")
    result.loc[
        ~result["epci_admin_website_url"].apply(has_meaningful_url_structure),
        "epci_admin_website_url",
    ] = None

    nullable_fields = [
        "epci",
        "epci_url",
        "epci_admin_website_url",
        "admin_website_url",
        "url",
        "email",
        "phones",
        "adresse",
        "code_postal",
    ]
    for field in nullable_fields:
        result[field] = result[field].replace("", None)

    return result


def main():
    print("Creating database CSV...")
    dataset = build_final_dataset()
    dataset.to_csv("mairies_database_ready.csv", index=False, encoding="utf-8")

    print(f"✅ Created {len(dataset):,} records")
    print(f"Records with email: {dataset['email'].notna().sum():,}")
    print(f"Records with phone: {dataset['phones'].notna().sum():,}")
    print(f"Records with website: {dataset['url'].notna().sum():,}")
    print(
        f"Records with valid admin portal: {dataset['admin_website_url'].notna().sum():,}"
    )
    print(f"Records with EPCI: {dataset['epci'].notna().sum():,}")
    print(f"Records with EPCI URL: {dataset['epci_url'].notna().sum():,}")
    print(
        f"Records with valid EPCI admin portal: {dataset['epci_admin_website_url'].notna().sum():,}"
    )


if __name__ == "__main__":
    main()
