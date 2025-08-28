#!/usr/bin/env python3

import pandas as pd
import re
from urllib.parse import urlparse


def clean_insee_code(code):
    if pd.isna(code):
        return ""
    cleaned = re.sub(r"\D", "", str(code).strip().rstrip(".0"))
    return cleaned.lstrip("0") or "0"


def extract_main_website(websites_str):
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


def extract_postal_code(address):
    if pd.isna(address):
        return ""
    match = re.search(r"\b(\d{5})\b", str(address))
    return match.group(1) if match else ""


def clean_commune_name(name):
    if pd.isna(name):
        return ""
    return str(name).replace("Mairie - ", "").strip()


def is_valid_admin_url(url):
    """Validate admin URLs for mairies based on content keywords"""
    if pd.isna(url) or not url:
        return False

    url_lower = url.lower()

    # Invalid if contains plan/plu keywords
    if any(keyword in url_lower for keyword in ["plan", "plu"]):
        return False

    # Valid if contains urbanisme-related keywords
    valid_keywords = ["urbanisme", "guichet", "droit", "declaration", "demande", "sve"]
    return any(keyword in url_lower for keyword in valid_keywords)


def is_valid_epci_url(url):
    """Validate EPCI URLs - must have subdomain OR at least one folder"""
    if pd.isna(url) or not url:
        return False

    try:
        parsed = urlparse(url)
        domain_parts = parsed.netloc.split(".")

        # Has subdomain (more than 2 parts like www.example.com)
        has_subdomain = len(domain_parts) > 2

        # Has folder path (more than just '/')
        has_folder = len(parsed.path.strip("/")) > 0

        return has_subdomain or has_folder
    except:
        return False


def load_data():
    urban = pd.read_csv("urban_planning_portals.csv")
    contacts = pd.read_csv("mairies_epcis_contacts.csv", sep=";", dtype=str)
    return urban, contacts


def prepare_urban_portals(urban):
    urban = urban[urban["ai_selected"] == True].copy()
    urban["clean_entity_code"] = urban["entity_code"].apply(clean_insee_code)
    best_portals = urban.loc[
        urban.groupby("clean_entity_code")["ai_confidence"].idxmax()
    ]
    return best_portals[["clean_entity_code", "url"]].rename(
        columns={"url": "admin_website_url"}
    )


def prepare_mairies(contacts):
    mairies = contacts[contacts["service_type"] == "mairie"].copy()
    mairies["clean_insee_code"] = mairies["insee_codes"].apply(clean_insee_code)
    mairies["commune"] = mairies["nom"].apply(clean_commune_name)
    mairies["url"] = mairies["websites"].apply(extract_main_website)
    mairies["code_postal"] = mairies["addresses"].apply(extract_postal_code)
    mairies["clean_epci_code"] = mairies["epci_code"].apply(clean_insee_code)

    return mairies[
        [
            "clean_insee_code",
            "commune",
            "epci_name",
            "clean_epci_code",
            "url",
            "email",
            "addresses",
            "code_postal",
        ]
    ].rename(
        columns={
            "clean_insee_code": "code_insee",
            "epci_name": "epci",
            "addresses": "adresse",
        }
    )


def prepare_epcis(contacts):
    epcis = contacts[contacts["service_type"] == "epci"].copy()
    epcis["clean_epci_code"] = epcis["epci_code"].apply(clean_insee_code)
    epcis["epci_url"] = epcis["websites"].apply(extract_main_website)

    return epcis[["clean_epci_code", "epci_url"]]


def create_final_dataset():
    urban, contacts = load_data()

    portals = prepare_urban_portals(urban)
    mairies = prepare_mairies(contacts)
    epcis = prepare_epcis(contacts)

    result = mairies.merge(
        portals, left_on="code_insee", right_on="clean_entity_code", how="left"
    )
    result = result.merge(epcis, on="clean_epci_code", how="left")

    result = result[
        [
            "code_insee",
            "commune",
            "epci",
            "epci_url",
            "admin_website_url",
            "url",
            "email",
            "adresse",
            "code_postal",
        ]
    ].copy()

    result["epci_admin_website_url"] = ""
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
            "adresse",
            "code_postal",
        ]
    ]

    # Apply URL validations
    result.loc[
        ~result["admin_website_url"].apply(is_valid_admin_url), "admin_website_url"
    ] = None
    result.loc[~result["epci_url"].apply(is_valid_epci_url), "epci_url"] = None

    nullable_cols = [
        "epci",
        "epci_url",
        "epci_admin_website_url",
        "admin_website_url",
        "url",
        "email",
        "adresse",
        "code_postal",
    ]
    for col in nullable_cols:
        result[col] = result[col].replace("", None)

    return result


def main():
    print("Creating database CSV...")
    df = create_final_dataset()
    df.to_csv("mairies_database_ready.csv", index=False, encoding="utf-8")

    print(f"✅ Created {len(df):,} records")
    print(f"Records with email: {df['email'].notna().sum():,}")
    print(f"Records with website: {df['url'].notna().sum():,}")
    print(f"Records with valid admin portal: {df['admin_website_url'].notna().sum():,}")
    print(f"Records with EPCI: {df['epci'].notna().sum():,}")
    print(f"Records with valid EPCI URL: {df['epci_url'].notna().sum():,}")


if __name__ == "__main__":
    main()
