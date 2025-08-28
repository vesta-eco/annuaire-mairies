import pandas as pd
import requests
import json
import tldextract
import argparse
import time
from tqdm import tqdm


def extract_json_value(json_field):
    if pd.isnull(json_field) or json_field == "":
        return ""

    try:
        data = json.loads(json_field)
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("valeur", "")
    except (json.JSONDecodeError, AttributeError, KeyError):
        return ""

    return ""


def get_domain(url):
    if pd.isnull(url):
        return ""

    parts = tldextract.extract(url)
    if parts.suffix:
        return f"{parts.domain.lower()}.{parts.suffix.lower()}"
    return parts.domain.lower()


def clean_commune_name(mairie_name):
    if pd.isnull(mairie_name):
        return ""

    prefixes = ["Mairie déléguée - ", "Mairie - ", "Mairie déléguée ", "Mairie "]

    for prefix in prefixes:
        if mairie_name.startswith(prefix):
            return mairie_name[len(prefix) :]

    return mairie_name


def fetch_sve_communes():
    api_url = "https://sve-api.sirap.fr/api/v1/communes"
    sve_url = "https://sve.sirap.fr"
    communes = []
    page = 1

    try:
        response = requests.get(f"{api_url}?pageIndex=1&pageSize=1", timeout=10)
        response.raise_for_status()
        total_count = response.json().get("count", 0)

        print(f"Fetching {total_count} SVE SIRAP communes...")

        while len(communes) < total_count:
            response = requests.get(
                f"{api_url}?pageIndex={page}&pageSize=50", timeout=10
            )
            response.raise_for_status()

            page_communes = response.json().get("list", [])
            if not page_communes:
                break

            for commune in page_communes:
                insee_code = commune.get("insee")
                if insee_code:
                    communes.append(
                        {
                            "code_insee_commune": insee_code,
                            "admin_website_url": sve_url,
                            "method": "sve_sirap_api",
                        }
                    )

            page += 1
            time.sleep(0.1)

        return pd.DataFrame(communes)

    except Exception as e:
        print(f"SVE SIRAP API error: {e}")
        return pd.DataFrame(
            columns=["code_insee_commune", "admin_website_url", "method"]
        )


def assign_sve_websites(cities_df):
    sve_communes = fetch_sve_communes()

    if sve_communes.empty:
        return cities_df

    sve_communes["normalized_insee"] = sve_communes["code_insee_commune"].str[1:]
    sve_codes = set(sve_communes["normalized_insee"])

    mask = cities_df["code_insee_commune"].isin(sve_codes)
    cities_df.loc[mask, "admin_website_url"] = "https://sve.sirap.fr"
    cities_df.loc[mask, "method"] = "sve_sirap_api"

    matches = mask.sum()
    print(f"SVE SIRAP: {matches} cities matched")

    return cities_df


def load_cities():
    admin_data = pd.read_csv(
        "sources/annuaire_administrations.csv", sep=";", encoding="utf-8", comment="#"
    )
    mairies = admin_data[admin_data["nom"].str.contains("Mairie", na=False)].copy()

    mairies["website_url"] = mairies["site_internet"].apply(extract_json_value)
    mairies["email"] = mairies["adresse_courriel"].apply(extract_json_value)
    mairies["phone"] = mairies["telephone"].apply(extract_json_value)
    mairies["city_domain"] = mairies["website_url"].apply(get_domain)
    mairies["commune_name"] = mairies["nom"].apply(clean_commune_name)

    print(f"Loaded {len(mairies)} cities")
    return mairies


def load_groupings():
    communes = pd.read_csv("sources/communes-france-2025.csv", dtype=str, comment="#")

    columns = [
        "code_insee",
        "nom_standard",
        "epci_code",
        "epci_nom",
        "code_postal",
        "population",
    ]
    groupings = communes[columns].copy()

    groupings = groupings.rename(
        columns={
            "code_insee": "code_insee_commune",
            "nom_standard": "commune_name",
            "epci_code": "EPCI_code",
            "epci_nom": "EPCI",
        }
    )

    print(f"Loaded {len(groupings)} groupings")
    return groupings


def load_known_sites():
    sites = pd.read_csv("sources/known_admin_websites.csv", dtype=str)
    sites["epci_code"] = sites["epci_code"].str.replace(".0", "", regex=False)
    print(f"Loaded {len(sites)} known sites")
    return sites


def step2_manual_collection(known_sites_df):
    print("\n=== STEP 2: Manual Collection ===")

    results = []

    # Add all manually collected sites
    print(f"\n📋 Processing {len(known_sites_df)} manually collected admin websites...")

    for i, (_, row) in enumerate(known_sites_df.iterrows()):
        epci_name = row.get("EPCI", "Unknown EPCI")
        epci_code = row["epci_code"]
        admin_url = row["admin_website_url"]

        results.append(
            {
                "EPCI_code": epci_code,
                "admin_website_url": admin_url,
                "method": "manual_collection",
            }
        )

        if i < 5:  # Show first 5 as examples
            print(f"  ✅ {epci_name} (code: {epci_code}) -> {admin_url}")
        elif i == 5:
            print(f"  ... and {len(known_sites_df) - 5} more sites")

    print(f"\n🎯 Added {len(known_sites_df)} manually collected sites")

    # Check for duplicate EPCI codes in known sites
    duplicates = known_sites_df["epci_code"].duplicated()
    if duplicates.any():
        duplicate_count = duplicates.sum()
        print(
            f"⚠️  Warning: {duplicate_count} duplicate EPCI codes found in known admin sites"
        )
        print("   These EPCIs have multiple admin websites - keeping first occurrence:")

    manual_df = pd.DataFrame(results)
    return manual_df


def step2_epci_inference(cities_df, admin_results_df):
    print("\n=== STEP 2: EPCI-Level Inference ===")

    # Count cities that already have admin websites assigned (e.g., from SVE SIRAP)
    pre_existing_admin_cities = cities_df["admin_website_url"].notna().sum()
    if pre_existing_admin_cities > 0:
        print(
            f"\n📋 Cities with pre-assigned admin websites: {pre_existing_admin_cities}"
        )
        pre_existing_methods = cities_df[cities_df["admin_website_url"].notna()][
            "method"
        ].value_counts()
        for method, count in pre_existing_methods.items():
            print(f"  - {method}: {count} cities")

    epci_code_to_admin = admin_results_df.set_index("EPCI_code")[
        "admin_website_url"
    ].to_dict()
    epci_code_to_method = admin_results_df.set_index("EPCI_code")["method"].to_dict()

    print(
        f"\n🔗 Mapping admin websites to remaining cities via {len(epci_code_to_admin)} EPCIs..."
    )

    # Show breakdown by method
    if not admin_results_df.empty:
        method_counts = admin_results_df["method"].value_counts()
        for method, count in method_counts.items():
            print(f"  - {method}: {count} EPCIs")

    # Only assign EPCI-level admin websites to cities that don't already have one
    # (preserves direct assignments like SVE SIRAP)
    no_admin_mask = cities_df["admin_website_url"].isna()
    cities_df.loc[no_admin_mask, "admin_website_url"] = cities_df.loc[
        no_admin_mask, "EPCI_code"
    ].map(epci_code_to_admin)
    cities_df.loc[no_admin_mask, "method"] = cities_df.loc[
        no_admin_mask, "EPCI_code"
    ].map(epci_code_to_method)

    cities_df.loc[cities_df["admin_website_url"].isna(), "method"] = pd.NA

    matched_cities = cities_df["admin_website_url"].notna().sum()
    matched_epcis = len(epci_code_to_admin)
    total_cities = len(cities_df)

    print(f"\n🎯 EPCI inference results:")
    print(
        f"  - Cities with admin websites: {matched_cities:,} / {total_cities:,} ({matched_cities/total_cities*100:.1f}%)"
    )
    print(f"  - EPCIs with admin websites: {matched_epcis}")
    print(
        f"  - Coverage: {matched_cities} cities across {matched_epcis} EPCIs have admin websites"
    )

    return cities_df


def run_admin_website_pipeline(
    run_sve_sirap=True,
    run_manual_collection=True,
):
    print("Starting Clean Admin Website Pipeline")
    print("=" * 50)

    all_cities = load_cities_from_annuaire()
    city_groupings = load_city_groupings()
    known_sites = load_known_admin_sites()

    cities_df = pd.merge(
        all_cities, city_groupings, on="code_insee_commune", how="left"
    )

    print(f"\nTotal cities processed: {len(cities_df)}")
    cities_with_domains = len(cities_df[cities_df["city_domain"] != ""])
    print(f"Cities with domains: {cities_with_domains}")

    all_admin_results = []

    # Step 0: SVE SIRAP API Check (highest priority) - directly modifies cities_df
    if run_sve_sirap:
        cities_df = step0_sve_sirap_check(cities_df)
    else:
        print("\n=== STEP 0: SVE SIRAP API Check === SKIPPED")

    # Step 1: Manual Collection (renumbered from Step 2)
    if run_manual_collection:
        manual_results = step2_manual_collection(known_sites)
        if not manual_results.empty:
            all_admin_results.append(manual_results)
    else:
        print("\n=== STEP 1: Manual Collection === SKIPPED")

    if all_admin_results:
        combined_admin_results = pd.concat(all_admin_results, ignore_index=True)

        # Handle duplicates properly - don't lose EPCIs!
        before_dedup = len(combined_admin_results)
        combined_admin_results = combined_admin_results.drop_duplicates(
            subset=["EPCI_code"], keep="first"
        )
        after_dedup = len(combined_admin_results)
        if before_dedup != after_dedup:
            print(
                f"⚠️  Removed {before_dedup - after_dedup} duplicate EPCI codes (kept first occurrence)"
            )

    else:
        combined_admin_results = pd.DataFrame(
            columns=["EPCI_code", "admin_website_url", "method"]
        )

    final_cities_df = step2_epci_inference(cities_df, combined_admin_results)

    output_columns = [
        "code_insee_commune",
        "commune_name_x",
        "code_postal",
        "population",
        "EPCI_code",
        "EPCI",
        "city_domain",
        "website_url",
        "email",
        "phone",
        "admin_website_url",
        "method",
    ]

    final_output = final_cities_df[output_columns].copy()
    final_output = final_output.rename(columns={"commune_name_x": "commune_name"})

    final_output.to_csv(
        "cities_with_admin_websites_clean.csv", index=False, encoding="utf-8"
    )

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    total_cities = len(final_output)
    cities_with_admin = final_output["admin_website_url"].notna().sum()

    print(f"Total cities processed: {total_cities}")
    print(f"Cities with admin websites: {cities_with_admin}")
    print(f"Coverage: {cities_with_admin/total_cities*100:.1f}%")

    print("\nMethods used:")
    method_counts = final_output["method"].value_counts()
    for method, count in method_counts.items():
        print(f"  {method}: {count} cities")

    print(f"\nResults saved to 'cities_with_admin_websites_clean.csv'")
    return final_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover admin websites for French cities"
    )
    parser.add_argument(
        "--no-sve-sirap",
        action="store_true",
        help="Skip SVE SIRAP API check (runs by default)",
    )

    parser.add_argument(
        "--no-manual",
        action="store_true",
        help="Skip manual collection",
    )

    args = parser.parse_args()

    run_admin_website_pipeline(
        run_sve_sirap=not args.no_sve_sirap,
        run_manual_collection=not args.no_manual,
    )
