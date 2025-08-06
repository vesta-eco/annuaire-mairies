import pandas as pd
import requests
import json
import tldextract
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import warnings
from functools import lru_cache
import argparse
import os
import time

warnings.filterwarnings("ignore")


def extract_value_from_json_field(json_field):
    if pd.isnull(json_field) or json_field == "":
        return ""

    try:
        data = json.loads(json_field)
        if isinstance(data, list) and len(data) > 0:
            return data[0].get("valeur", "")
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass

    return ""


def get_domain_from_url(url):
    if pd.isnull(url):
        return ""

    domain_parts = tldextract.extract(url)
    return (
        domain_parts.domain.lower() + "." + domain_parts.suffix.lower()
        if domain_parts.suffix
        else domain_parts.domain.lower()
    )


def extract_commune_name(mairie_name):
    if pd.isnull(mairie_name):
        return ""

    prefixes_to_remove = [
        "Mairie déléguée - ",
        "Mairie - ",
        "Mairie déléguée ",
        "Mairie ",
    ]

    commune_name = mairie_name
    for prefix in prefixes_to_remove:
        if commune_name.startswith(prefix):
            commune_name = commune_name[len(prefix) :]
            break

    return commune_name


def fetch_sve_sirap_communes():
    """
    Fetch all communes from SVE SIRAP API with pagination.
    Returns a DataFrame with INSEE codes and the SVE SIRAP admin website URL.
    """
    print("Fetching communes from SVE SIRAP API...")

    base_url = "https://sve-api.sirap.fr/api/v1/communes"
    sve_sirap_url = "https://sve.sirap.fr"
    all_communes = []
    page_index = 1
    page_size = 50  # Reasonable page size to avoid timeouts

    try:
        # First request to get total count
        response = requests.get(f"{base_url}?pageIndex=1&pageSize=1", timeout=10)
        response.raise_for_status()
        data = response.json()
        total_count = data.get("count", 0)

        print(f"Found {total_count} communes using SVE SIRAP")

        # Calculate total pages needed
        total_pages = (total_count + page_size - 1) // page_size

        # Fetch all pages
        with tqdm(total=total_pages, desc="Fetching SVE SIRAP pages") as pbar:
            while True:
                try:
                    response = requests.get(
                        f"{base_url}?pageIndex={page_index}&pageSize={page_size}",
                        timeout=10,
                    )
                    response.raise_for_status()
                    data = response.json()

                    communes_list = data.get("list", [])
                    if not communes_list:
                        break

                    # Process each commune in the page
                    for commune in communes_list:
                        insee_code = commune.get("insee")
                        if insee_code:
                            all_communes.append(
                                {
                                    "code_insee_commune": insee_code,
                                    "admin_website_url": sve_sirap_url,
                                    "method": "sve_sirap_api",
                                }
                            )

                    page_index += 1
                    pbar.update(1)

                    # Check if we've got all data
                    if len(all_communes) >= total_count:
                        break

                    # Small delay to be respectful to the API
                    time.sleep(0.1)

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching page {page_index}: {e}")
                    # Try a few more times before giving up
                    if page_index <= 3:  # Only retry for first few pages
                        time.sleep(1)
                        continue
                    else:
                        break

        sve_sirap_df = pd.DataFrame(all_communes)
        print(f"Successfully fetched {len(sve_sirap_df)} communes from SVE SIRAP API")
        return sve_sirap_df

    except Exception as e:
        print(f"Error fetching SVE SIRAP data: {e}")
        return pd.DataFrame(
            columns=["code_insee_commune", "admin_website_url", "method"]
        )


def step0_sve_sirap_check(cities_df):
    """
    Step 0: SVE SIRAP API Check - highest priority method
    Directly assigns SVE SIRAP as admin website for matching cities
    """
    print("\n=== STEP 0: SVE SIRAP API Check ===")

    sve_sirap_communes = fetch_sve_sirap_communes()

    if sve_sirap_communes.empty:
        print("No communes found via SVE SIRAP API")
        return cities_df

    # Normalize INSEE codes - SVE SIRAP uses 6-digit codes, communes CSV uses 5-digit codes WITH leading zeros
    # Convert SVE SIRAP codes from 6-digit (059010) to 5-digit format (59010) to match communes data
    sve_sirap_communes["normalized_insee"] = sve_sirap_communes[
        "code_insee_commune"
    ].str[1:]
    sve_sirap_insee_codes = set(sve_sirap_communes["normalized_insee"])

    # Create a mask for cities that use SVE SIRAP
    cities_df["uses_sve_sirap"] = cities_df["code_insee_commune"].isin(
        sve_sirap_insee_codes
    )

    # Assign SVE SIRAP as admin website for matching cities
    cities_df.loc[cities_df["uses_sve_sirap"], "admin_website_url"] = (
        "https://sve.sirap.fr"
    )
    cities_df.loc[cities_df["uses_sve_sirap"], "method"] = "sve_sirap_api"

    # Count matches
    sve_sirap_matches = cities_df["uses_sve_sirap"].sum()
    overlap_insee_codes = sve_sirap_insee_codes.intersection(
        set(cities_df["code_insee_commune"])
    )

    print(f"Found {len(sve_sirap_communes)} communes in SVE SIRAP API")
    print(f"Matched {sve_sirap_matches} cities in your dataset to SVE SIRAP")
    print(
        f"INSEE code overlap: {len(overlap_insee_codes)} / {len(sve_sirap_insee_codes)} (after normalization)"
    )

    # Clean up temporary column
    cities_df = cities_df.drop("uses_sve_sirap", axis=1)

    return cities_df


def load_cities_from_annuaire():
    print("Loading cities from annuaire...")

    admin_data = pd.read_csv(
        "sources/annuaire_administrations.csv", sep=";", encoding="utf-8"
    )
    mairies_data = admin_data[admin_data["nom"].str.contains("Mairie", na=False)].copy()

    mairies_data["website_url"] = mairies_data["site_internet"].apply(
        extract_value_from_json_field
    )
    mairies_data["email"] = mairies_data["adresse_courriel"].apply(
        extract_value_from_json_field
    )
    mairies_data["phone"] = mairies_data["telephone"].apply(
        extract_value_from_json_field
    )
    mairies_data["city_domain"] = mairies_data["website_url"].apply(get_domain_from_url)
    mairies_data["commune_name"] = mairies_data["nom"].apply(extract_commune_name)

    cities_with_domains_count = len(mairies_data[mairies_data["city_domain"] != ""])
    print(
        f"Found {len(mairies_data)} cities total, {cities_with_domains_count} with domains"
    )
    return mairies_data


def load_city_groupings():
    print("Loading city groupings...")
    communes_data = pd.read_csv("sources/communes-france-2025.csv", dtype=str)

    groupings = communes_data[
        [
            "code_insee",
            "nom_standard",
            "epci_code",
            "epci_nom",
            "code_postal",
            "population",
        ]
    ].copy()
    groupings = groupings.rename(
        columns={
            "code_insee": "code_insee_commune",
            "nom_standard": "commune_name",
            "epci_code": "EPCI_code",
            "epci_nom": "EPCI",
        }
    )

    print(f"Loaded {len(groupings)} cities with groupings")
    return groupings


def load_known_admin_sites():
    print("Loading known admin websites...")
    known_sites = pd.read_csv("sources/known_admin_websites.csv", dtype=str)

    # Clean EPCI codes - remove .0 suffix if present
    known_sites["epci_code"] = known_sites["epci_code"].str.replace(
        ".0", "", regex=False
    )

    print(f"Found {len(known_sites)} known admin websites")
    return known_sites


def load_backlinks():
    print("Loading backlinks...")
    backlinks_folder = "backlinks"
    backlinks_files = [f for f in os.listdir(backlinks_folder) if f.endswith(".csv")]

    backlinks_list = []
    for f in backlinks_files:
        df = pd.read_csv(os.path.join(backlinks_folder, f), dtype=str)
        backlinks_list.append(df)

    backlinks_data = pd.concat(backlinks_list, ignore_index=True)
    backlinks_data["source_domain"] = backlinks_data["Source url"].apply(
        get_domain_from_url
    )

    print(f"Loaded {len(backlinks_data)} backlinks")
    return backlinks_data


@lru_cache(maxsize=1000)
def check_service_public_api(code_insee):
    url_daua = f"https://demarches.service-public.fr/mademarche/DAUA/ajax/referentiel/rest/getCommuneRaccordee?codeInsee={code_insee}"
    url_urbanisme = f"https://demarches.service-public.fr/mademarche/DAUA/ajax/referentiel/rest/getURLTeleserviceUrbanismeAmont?codeInsee={code_insee}"

    try:
        response_daua = requests.get(url_daua, timeout=5)
        if response_daua.status_code == 200:
            data_daua = response_daua.json()
            if data_daua.get("communeRaccordeeDAUA", False):
                return "https://demarches.service-public.fr/mademarche/DAUA/demarche"

        response_urbanisme = requests.get(url_urbanisme, timeout=5)
        if response_urbanisme.status_code == 200:
            data_urbanisme = response_urbanisme.json()
            url_service_urbanisme = data_urbanisme.get("urlServiceUrbanisme")
            if url_service_urbanisme:
                return url_service_urbanisme

    except Exception:
        pass

    return None


def step1_service_public_check(cities_df):
    print("\n=== STEP 1: Service Public API Check ===")

    results = []
    cities_to_check = cities_df["code_insee_commune"].unique()

    def check_city(code_insee):
        admin_url = check_service_public_api(code_insee)
        if admin_url:
            return (code_insee, admin_url)
        return None

    print(f"Checking {len(cities_to_check)} cities...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        api_results = list(
            tqdm(
                executor.map(check_city, cities_to_check),
                total=len(cities_to_check),
                desc="Service Public API",
            )
        )

    for result in api_results:
        if result:
            code_insee, admin_url = result
            results.append(
                {
                    "code_insee_commune": code_insee,
                    "admin_website_url": admin_url,
                    "method": "service_public_api",
                }
            )

    service_public_df = pd.DataFrame(results)
    print(f"Found {len(service_public_df)} cities via Service Public API")
    return service_public_df


def extract_admin_prefixes(known_sites_df, all_cities_df):
    discovered_prefixes = set()

    for _, known_row in known_sites_df.iterrows():
        admin_url = known_row["admin_website_url"]
        admin_domain = get_domain_from_url(admin_url)
        known_epci_code = known_row["epci_code"]

        if admin_domain:
            epci_cities = all_cities_df[all_cities_df["EPCI_code"] == known_epci_code]

            for _, city_row in epci_cities.iterrows():
                city_domain = city_row["city_domain"]

                if city_domain and admin_domain.endswith("." + city_domain):
                    prefix = admin_domain.replace("." + city_domain, "")
                    if len(prefix) >= 2 and prefix != "www":
                        discovered_prefixes.add(prefix)

            admin_keywords = [
                "urbanisme",
                "demarche",
                "permis",
                "gnau",
                "guichet",
                "sve",
                "cartads",
            ]
            domain_parts = admin_domain.split(".")
            if len(domain_parts) >= 2:
                first_part = domain_parts[0]
                if any(keyword in first_part.lower() for keyword in admin_keywords):
                    discovered_prefixes.add(first_part)

    common_admin_prefixes = [
        "urbanisme",
        "demarches",
        "permis",
        "guichet",
        "e-permis",
        "admin",
        "services",
        "mairie",
        "gnau",
        "sve",
    ]
    discovered_prefixes.update(common_admin_prefixes)

    return list(discovered_prefixes)


def step2_manual_and_prefix_expansion(
    known_sites_df, all_cities_df, existing_results_df, skip_prefix_expansion=False
):
    print("\n=== STEP 2: Manual Collection + Prefix Expansion ===")

    results = []

    # First, add all manually collected sites
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

    # Get existing EPCI codes that already have admin websites
    existing_epci_codes = (
        set(existing_results_df["EPCI_code"].dropna())
        if not existing_results_df.empty
        else set()
    )
    known_epci_codes = set(known_sites_df["epci_code"])
    all_existing_epci_codes = existing_epci_codes.union(known_epci_codes)

    # Try prefix expansion for EPCIs that don't have admin websites yet (if not skipped)
    prefix_results = []

    if skip_prefix_expansion:
        print("\n🚫 Prefix expansion skipped as requested")
    else:
        cities_without_admin = all_cities_df[
            (~all_cities_df["EPCI_code"].isin(all_existing_epci_codes))
            & (all_cities_df["city_domain"] != "")
        ]

        prefixes = extract_admin_prefixes(known_sites_df, all_cities_df)
        print(f"Found {len(prefixes)} common prefixes: {prefixes[:10]}...")

        successful_prefix_epcis = set()

        print(
            f"\nTesting prefixes on {len(cities_without_admin)} cities in EPCIs without admin websites..."
        )

        for i, prefix in enumerate(prefixes):
            print(f"\n--- Testing prefix '{prefix}' ({i+1}/{len(prefixes)}) ---")
            prefix_hits = 0

            for _, city_row in cities_without_admin.iterrows():
                epci_code = city_row["EPCI_code"]
                if epci_code in successful_prefix_epcis:
                    continue  # Already found a working prefix for this EPCI

                city_domain = city_row["city_domain"]
                commune_name = city_row.get("commune_name", "Unknown")
                test_url = f"https://{prefix}.{city_domain}"

                try:
                    print(f"  Testing: {test_url} (for {commune_name})", end=" -> ")
                    response = requests.head(test_url, timeout=2, verify=False)
                    if response.status_code < 400:
                        print(f"✅ SUCCESS ({response.status_code})")
                        prefix_results.append(
                            {
                                "EPCI_code": epci_code,
                                "admin_website_url": test_url,
                                "method": "prefix_expansion",
                            }
                        )
                        successful_prefix_epcis.add(epci_code)
                        prefix_hits += 1
                        break
                    else:
                        print(f"❌ Failed ({response.status_code})")
                except Exception as e:
                    print(f"❌ Error ({type(e).__name__})")
                    continue

            print(f"  → Found {prefix_hits} working sites with prefix '{prefix}'")

            if prefix_hits == 0:
                print(
                    f"  → No working sites found for prefix '{prefix}', trying next prefix..."
                )

        print(f"\n🎯 Prefix expansion summary:")
        print(f"  - Total EPCIs tested: {len(set(cities_without_admin['EPCI_code']))}")
        print(f"  - EPCIs with working prefixes: {len(successful_prefix_epcis)}")
        print(f"  - Working URLs found: {len(prefix_results)}")

    results.extend(prefix_results)
    print(f"Found {len(prefix_results)} additional sites via prefix expansion")

    manual_df = pd.DataFrame(results)
    return manual_df


def identify_main_urbanism_platforms(backlinks_df):
    urbanism_keywords = [
        "urbanisme",
        "demarche",
        "permis",
        "declaration",
        "travaux",
        "gnau",
        "guichet",
        "sve",
        "operis",
        "numerian",
        "cartads",
    ]

    platform_domains = set()

    for _, row in backlinks_df.iterrows():
        target_url = row["Target url"]
        target_domain = get_domain_from_url(target_url)

        if any(keyword in target_domain.lower() for keyword in urbanism_keywords):
            platform_domains.add(target_domain)

    return list(platform_domains)


def step3_backlinks_to_platforms(all_cities_df, backlinks_df, existing_results_df):
    print("\n=== STEP 3: Backlinks to Main Platforms ===")

    platform_domains = identify_main_urbanism_platforms(backlinks_df)
    print(f"Identified {len(platform_domains)} main urbanism platforms")

    results = []
    cities_with_domains = all_cities_df[all_cities_df["city_domain"] != ""]
    city_domain_to_epci_code = cities_with_domains.set_index("city_domain")[
        "EPCI_code"
    ].to_dict()
    existing_epci_codes = (
        set(existing_results_df["EPCI_code"].dropna())
        if not existing_results_df.empty
        else set()
    )

    for _, backlink_row in tqdm(
        backlinks_df.iterrows(), total=len(backlinks_df), desc="Processing backlinks"
    ):
        source_domain = backlink_row["source_domain"]
        target_url = backlink_row["Target url"]
        target_domain = get_domain_from_url(target_url)

        if (
            target_domain in platform_domains
            and source_domain in city_domain_to_epci_code
        ):
            epci_code = city_domain_to_epci_code[source_domain]

            if epci_code not in existing_epci_codes:
                results.append(
                    {
                        "EPCI_code": epci_code,
                        "admin_website_url": target_url,
                        "method": "backlinks_to_platform",
                    }
                )
                existing_epci_codes.add(epci_code)

    backlinks_df_result = pd.DataFrame(results)
    print(f"Found {len(backlinks_df_result)} EPCIs via backlinks to platforms")
    return backlinks_df_result


def step4_epci_inference(cities_df, admin_results_df):
    print("\n=== STEP 4: EPCI-Level Inference ===")

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
    run_service_public=False,
    run_manual_expansion=True,
    run_backlinks=True,
    skip_prefix_expansion=False,
):
    print("Starting Clean Admin Website Pipeline")
    print("=" * 50)

    all_cities = load_cities_from_annuaire()
    city_groupings = load_city_groupings()
    known_sites = load_known_admin_sites()
    backlinks = load_backlinks()

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

    # Step 1: Service Public API Check
    if run_service_public:
        service_public_results = step1_service_public_check(cities_df)
        if not service_public_results.empty:
            sp_epci_results = pd.merge(
                service_public_results,
                cities_df[["code_insee_commune", "EPCI_code"]],
                on="code_insee_commune",
            )
            sp_epci_results = sp_epci_results[
                ["EPCI_code", "admin_website_url", "method"]
            ].drop_duplicates()
            all_admin_results.append(sp_epci_results)
    else:
        print("\n=== STEP 1: Service Public API Check === SKIPPED")

    # Step 2: Manual Collection + Prefix Expansion
    if run_manual_expansion:
        existing_df = (
            pd.concat(all_admin_results, ignore_index=True)
            if all_admin_results
            else pd.DataFrame()
        )
        manual_results = step2_manual_and_prefix_expansion(
            known_sites, cities_df, existing_df, skip_prefix_expansion
        )
        if not manual_results.empty:
            all_admin_results.append(manual_results)
    else:
        print("\n=== STEP 2: Manual + Prefix Expansion === SKIPPED")

    # Step 3: Backlinks to Main Platforms
    if run_backlinks:
        existing_df = (
            pd.concat(all_admin_results, ignore_index=True)
            if all_admin_results
            else pd.DataFrame()
        )
        backlinks_results = step3_backlinks_to_platforms(
            cities_df, backlinks, existing_df
        )
        if not backlinks_results.empty:
            all_admin_results.append(backlinks_results)
    else:
        print("\n=== STEP 3: Backlinks to Platforms === SKIPPED")

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

    final_cities_df = step4_epci_inference(cities_df, combined_admin_results)

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
        "--service-public", action="store_true", help="Run Service Public API check"
    )
    parser.add_argument(
        "--no-manual",
        action="store_true",
        help="Skip manual collection and prefix expansion",
    )
    parser.add_argument(
        "--no-backlinks",
        action="store_true",
        help="Skip backlinks to platforms analysis",
    )
    parser.add_argument(
        "--no-prefix-expansion",
        action="store_true",
        help="Skip prefix expansion but keep manual collection",
    )

    args = parser.parse_args()

    run_admin_website_pipeline(
        run_sve_sirap=not args.no_sve_sirap,
        run_service_public=args.service_public,
        run_manual_expansion=not args.no_manual,
        run_backlinks=not args.no_backlinks,
        skip_prefix_expansion=args.no_prefix_expansion,
    )
