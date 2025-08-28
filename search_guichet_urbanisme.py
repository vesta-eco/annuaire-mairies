#!/usr/bin/env python3

import argparse
import json
import os
import pandas as pd
import re
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class UrbanPlanningSearcher:
    def __init__(self, serp_key: str, openai_key: str = None):
        self.serp_key = serp_key
        self.openai_client = (
            self._init_openai(openai_key) if openai_key and OpenAI else None
        )
        self.confidence_threshold = 0.7

    def _init_openai(self, api_key: str) -> Optional[OpenAI]:
        try:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            return OpenAI()
        except Exception:
            return None

    def _clean_entity_name(self, name: str) -> str:
        if not name:
            return ""
        return re.sub(r"^\s*Mairie\s*-\s*", "", name.strip(), flags=re.IGNORECASE)

    def _clean_code(self, code: str) -> str:
        if not code:
            return ""
        cleaned = re.sub(r"\D", "", str(code).strip().rstrip(".0"))
        return cleaned.lstrip("0") or "0"

    def _parse_epci_filter(self, epci_filter_input: str) -> Set[str]:
        if not epci_filter_input:
            return set()

        if os.path.isfile(epci_filter_input):
            try:
                with open(epci_filter_input, "r", encoding="utf-8") as f:
                    epci_codes = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"❌ Error reading EPCI filter file: {e}")
                return set()
        else:
            epci_codes = [code.strip() for code in epci_filter_input.split(",")]

        return {self._clean_code(code) for code in epci_codes if code.strip()}

    def search_google(self, municipality: str, code: str) -> List[Dict]:
        dept_code = code[:2] if len(code) >= 2 else ""
        query = f"guichet unique urbanisme {municipality} {dept_code}"

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serp_key,
            "num": 10,
            "hl": "fr",
            "gl": "fr",
        }

        try:
            print(f"🔍 Searching: {query}")
            response = requests.get(
                "https://serpapi.com/search.json", params=params, timeout=30
            )
            response.raise_for_status()

            results = []
            for rank, item in enumerate(
                response.json().get("organic_results", [])[:10], 1
            ):
                result = {
                    "entity_name": municipality,
                    "entity_code": code,
                    "search_query": query,
                    "rank": rank,
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "domain": item.get("displayed_link", ""),
                    "found_at": datetime.now().isoformat(),
                }
                results.append(result)

            print(f"✅ Found {len(results)} results")
            return results

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def evaluate_with_ai(
        self,
        municipality: str,
        code: str,
        results: List[Dict],
        service_type: str,
        epci_info: Dict = None,
    ) -> Optional[Dict]:
        if not self.openai_client or not results:
            return None

        context = {
            "collectivite": {
                "name": municipality,
                "code": code,
                "type": service_type,
                "epci_info": epci_info or {},
            },
            "candidates": [
                {
                    "index": idx,
                    "title": r["title"][:200],
                    "url": r["url"][:500],
                    "domain": r["domain"][:200],
                    "snippet": r["snippet"][:500],
                }
                for idx, r in enumerate(results, 1)
            ],
        }

        system_prompt = self._build_system_prompt(service_type)
        user_prompt = self._build_evaluation_prompt(context)

        try:
            completion = self.openai_client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            ai_response = json.loads(completion.choices[0].message.content or "{}")
            return self._process_ai_response(results, ai_response)

        except Exception as e:
            print(f"⚠️ AI evaluation failed: {e}")
            return None

    def _build_system_prompt(self, service_type: str) -> str:
        return f"""You are an expert in French urban planning portals. Your task is to identify the SPECIFIC local portal for urban planning declarations ("déclaration préalable") for this {service_type}.

CRITICAL RULES:
1. ONLY return results with HIGH CONFIDENCE (>= 0.7/1.0)
2. LOCAL PORTALS ONLY - reject national portals (service-public.fr, gouvernement.fr)
3. For EPCIs: ensure the portal is specifically for THIS community, not another one
4. For municipalities: if portal belongs to an EPCI, verify it's the RIGHT EPCI
5. Prefer direct access to declaration forms over general information pages

TRUSTED PLATFORMS (even with generic domains):
- operis.fr, geosphere.fr, sictiam.fr, sirap.fr, atreal.fr, netagis.fr
- e-permis.fr, urbanisme17.fr, siea-sig.fr, adacl40.fr

RETURN FORMAT: JSON with confidence_score (0-1), best_index (1-based), and detailed reasoning."""

    def _build_evaluation_prompt(self, context: Dict) -> str:
        return json.dumps(
            {
                "task": "Find the specific local urban planning portal",
                "context": context,
                "requirements": {
                    "minimum_confidence": 0.7,
                    "prefer_local_platforms": True,
                    "avoid_national_sites": True,
                    "verify_epci_match": True,
                },
            },
            ensure_ascii=False,
        )

    def _process_ai_response(
        self, results: List[Dict], ai_response: Dict
    ) -> Optional[Dict]:
        confidence = ai_response.get("confidence_score", 0)
        best_index = ai_response.get("best_index")
        reasoning = ai_response.get("reasoning", "")

        print(f"🤖 AI confidence: {confidence:.2f}")
        print(f"🤖 Reasoning: {reasoning}")

        if confidence < self.confidence_threshold:
            print(
                f"⚠️ Confidence too low ({confidence:.2f} < {self.confidence_threshold}), skipping"
            )
            return None

        if not best_index or best_index < 1 or best_index > len(results):
            print("⚠️ Invalid best_index from AI, skipping")
            return None

        best_result = results[best_index - 1].copy()
        best_result.update(
            {
                "ai_confidence": confidence,
                "ai_reasoning": reasoning,
                "ai_timestamp": datetime.now().isoformat(),
            }
        )

        return best_result

    def load_entities(
        self,
        csv_path: str,
        service_types: List[str] = None,
        target_epci_codes: Set[str] = None,
    ) -> pd.DataFrame:
        try:
            df = pd.read_csv(csv_path, sep=";", dtype=str, comment="#")

            if service_types:
                df = df[df["service_type"].isin(service_types)]

            df["code"] = df.apply(
                lambda row: (
                    row.get("epci_code", "")
                    if row.get("service_type") == "epci"
                    else row.get("insee_codes", "")
                ),
                axis=1,
            )

            df["clean_name"] = df["nom"].apply(self._clean_entity_name)

            if target_epci_codes:
                df["clean_epci_code"] = df["epci_code"].apply(self._clean_code)
                initial_count = len(df)
                df = df[
                    (df["service_type"] != "mairie")
                    | (df["clean_epci_code"].isin(target_epci_codes))
                ]
                filtered_count = len(df)
                print(
                    f"🎯 EPCI filter applied: {initial_count} → {filtered_count} entities"
                )

            columns = [
                "nom",
                "clean_name",
                "code",
                "service_type",
                "epci_code",
                "epci_name",
            ]
            available_columns = [col for col in columns if col in df.columns]

            return df[available_columns].dropna(subset=["code"])

        except Exception as e:
            print(f"❌ Error loading entities: {e}")
            return pd.DataFrame()

    def get_processed_codes(self, output_csv: str) -> set:
        if not os.path.exists(output_csv):
            return set()

        try:
            df = pd.read_csv(output_csv)
            if "entity_code" in df.columns:
                return set(df["entity_code"].apply(self._clean_code))
        except Exception:
            pass
        return set()

    def save_result(self, result: Dict, output_csv: str):
        if not result:
            return

        try:
            df = pd.DataFrame([result])
            file_exists = os.path.exists(output_csv)

            df.to_csv(
                output_csv,
                index=False,
                encoding="utf-8",
                mode="a" if file_exists else "w",
                header=not file_exists,
            )

        except Exception as e:
            print(f"❌ Error saving result: {e}")

    def process_entities(
        self,
        input_csv: str,
        output_csv: str,
        max_searches: int = None,
        service_types: List[str] = None,
        target_epci_codes: Set[str] = None,
        skip_existing: bool = True,
        delay: float = 1.0,
    ):
        entities = self.load_entities(input_csv, service_types, target_epci_codes)
        if entities.empty:
            print("❌ No entities found to process")
            return

        if skip_existing:
            processed_codes = self.get_processed_codes(output_csv)
            if processed_codes:
                entities["clean_code"] = entities["code"].apply(self._clean_code)
                entities = entities[~entities["clean_code"].isin(processed_codes)]
                print(f"⏭️ Skipping {len(processed_codes)} already processed entities")

        if max_searches:
            entities = entities.head(max_searches)

        print(f"🚀 Processing {len(entities)} entities")

        successful_count = 0
        failed_attempts = []

        for idx, (_, row) in enumerate(entities.iterrows(), 1):
            name = row["clean_name"]
            code = self._clean_code(row["code"])
            service_type = row["service_type"]

            epci_info = {}
            if "epci_code" in row and "epci_name" in row:
                epci_info = {
                    "code": row.get("epci_code", ""),
                    "name": row.get("epci_name", ""),
                }

            print(
                f"\n[{idx}/{len(entities)}] Processing: {row['nom']} ({service_type})"
            )

            search_results = self.search_google(name, code)
            if not search_results:
                print("❌ No search results found")
                failed_attempts.append(
                    {
                        "name": name,
                        "code": code,
                        "service_type": service_type,
                        "reason": "no_search_results",
                    }
                )
                continue

            ai_evaluation = self.evaluate_with_ai(
                name, code, search_results, service_type, epci_info
            )

            results_saved = 0
            for result in search_results:
                result["service_type"] = service_type
                result["entity_code"] = code
                result["epci_name"] = row.get("epci_name", "")
                result["epci_code"] = row.get("epci_code", "")

                if ai_evaluation and ai_evaluation.get("url") == result["url"]:
                    result["ai_confidence"] = ai_evaluation.get("ai_confidence", 0)
                    result["ai_reasoning"] = ai_evaluation.get("ai_reasoning", "")
                    result["ai_selected"] = True
                    result["ai_timestamp"] = ai_evaluation.get("ai_timestamp", "")
                else:
                    result["ai_confidence"] = 0
                    result["ai_reasoning"] = ""
                    result["ai_selected"] = False
                    result["ai_timestamp"] = ""

                self.save_result(result, output_csv)
                results_saved += 1

            successful_count += 1
            ai_status = (
                f"(AI selected: {bool(ai_evaluation)})"
                if ai_evaluation
                else "(No AI selection)"
            )
            print(f"✅ Saved {results_saved} results {ai_status}")

            if delay > 0:
                time.sleep(delay)

        print(
            f"\n🎉 Processing complete: {successful_count}/{len(entities)} entities processed"
        )
        if failed_attempts:
            print(f"⚠️ {len(failed_attempts)} entities had no search results")

    def show_summary(self, results_csv: str):
        try:
            if not os.path.exists(results_csv):
                print("❌ Results file not found")
                return

            df = pd.read_csv(results_csv)
            print(f"\n📊 Results Summary:")
            print(f"Total results saved: {len(df):,}")
            print(f"Unique entities: {df['entity_name'].nunique():,}")

            if "ai_selected" in df.columns:
                ai_selected_count = df["ai_selected"].sum()
                print(f"AI-selected results: {ai_selected_count:,}")

            if "ai_confidence" in df.columns:
                confident_results = df[df["ai_confidence"] >= self.confidence_threshold]
                if len(confident_results) > 0:
                    avg_confidence = confident_results["ai_confidence"].mean()
                    print(f"Average confidence (selected): {avg_confidence:.2f}")

            if "domain" in df.columns:
                top_domains = df["domain"].value_counts().head(5)
                print(f"Top domains: {', '.join(top_domains.index.tolist())}")

        except Exception as e:
            print(f"❌ Summary error: {e}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Search for urban planning portals")
    parser.add_argument(
        "--input-file", default="mairies_epcis_contacts.csv", help="Input CSV file"
    )
    parser.add_argument(
        "--output-file", default="urban_planning_portals.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--service-type",
        action="append",
        choices=["mairie", "epci"],
        help="Service types to process (can be used multiple times)",
    )
    parser.add_argument(
        "--target-epcis",
        help="Comma-separated EPCI SIREN codes or path to file containing codes (filters mairies only)",
    )
    parser.add_argument(
        "--max-searches", type=int, default=40, help="Maximum entities to process"
    )
    parser.add_argument(
        "--reprocess", action="store_true", help="Reprocess existing entities"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Delay between requests (seconds)"
    )

    args = parser.parse_args()

    serp_key = os.getenv("SERP_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not serp_key:
        print("❌ Missing SERP_API_KEY environment variable")
        return

    if not openai_key:
        print("⚠️ Missing OPENAI_API_KEY - will skip AI evaluation")

    searcher = UrbanPlanningSearcher(serp_key, openai_key)

    target_epci_codes = (
        searcher._parse_epci_filter(args.target_epcis) if args.target_epcis else None
    )

    if target_epci_codes:
        print(f"🎯 Filtering mairies by {len(target_epci_codes)} target EPCIs")

    print("🔍 Urban Planning Portal Search")
    print(f"📁 Input: {args.input_file}")
    print(f"💾 Output: {args.output_file}")
    print(f"🎯 Confidence threshold: {searcher.confidence_threshold}")
    print("💾 Save all results: ENABLED")

    try:
        searcher.process_entities(
            input_csv=args.input_file,
            output_csv=args.output_file,
            max_searches=args.max_searches,
            service_types=args.service_type,
            target_epci_codes=target_epci_codes,
            skip_existing=not args.reprocess,
            delay=args.delay,
        )

        searcher.show_summary(args.output_file)

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
