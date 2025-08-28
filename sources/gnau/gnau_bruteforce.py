#!/usr/bin/env python3
"""
gnau_collect.py   –   rerunnable, deduplicating collector
---------------------------------------------------------
Examples
--------

# Full crawl (160) – default:
./gnau_collect.py

# Only retry the troublemakers (2,3,4,24) – slow polite mode:
./gnau_collect.py --hosts 2 3 4 24 --throttle 3

# Fast one-off against gnau24, single try, no back-off:
./gnau_collect.py --hosts 24 --retries 1 --throttle 0
"""
import argparse, pathlib, urllib.request, urllib.parse, ssl, time, random, sys
from collections import defaultdict

# ----- CDX settings --------------------------------------------------------
CDX_URL   = ("https://web.archive.org/cdx/search/cdx?"
             "url={host}/*&output=text&fl=original&filter=statuscode:200")
SKIP      = {"Application", "css", "fonts", "lib"}       # noisy first-level dirs

# ---------------------------------------------------------------------------
def fetch_host(host: str,
               retries: int,
               base_back: float,
               throttle: float,
               jitter: float,
               timeout: int) -> set[str]:
    """Return a set of city folders for one host (empty on failure)."""
    url  = CDX_URL.format(host=urllib.parse.quote(host, safe=""))
    ctx  = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=timeout) as r:
                cities = set()
                for line in r.read().decode().splitlines():
                    p = urllib.parse.urlparse(line.strip())
                    if p.netloc != host:
                        continue
                    segs = p.path.split("/")
                    if len(segs) > 1 and segs[1] and segs[1] not in SKIP:
                        cities.add(segs[1])
            return cities                    # ← success
        except Exception as e:
            if attempt == retries:
                print(f"    ERROR – {e}")
            else:
                back = base_back * (2**(attempt-1)) + random.uniform(0, jitter)
                print(f"    attempt {attempt}/{retries} failed "
                      f"({e}) – retrying in {back:.1f}s")
                time.sleep(back)
    return set()                             # all attempts failed


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Collect GN-AU city folders.")
    ap.add_argument("--hosts", metavar="N", type=int, nargs="*",
                    help="Only run for these gnauN numbers (160). "
                         "If omitted, run the full range.")
    ap.add_argument("--outfile", default="gnau_subfolders.csv",
                    help="CSV file to create / update (default: %(default)s)")
    ap.add_argument("--retries",  type=int, default=3,
                    help="Max attempts per host (default: %(default)s)")
    ap.add_argument("--throttle", type=float, default=3.0,
                    help="Seconds to sleep after *every* host (default: %(default)s)")
    ap.add_argument("--base-back", type=float, default=2.0,
                    help="Initial back-off seconds (doubles each retry)")
    ap.add_argument("--jitter",    type=float, default=0.75,
                    help="± seconds random jitter (default: %(default)s)")
    ap.add_argument("--timeout",   type=int, default=20,
                    help="HTTP timeout per request (s) (default: %(default)s)")
    ns = ap.parse_args(argv)

    # -------- host list ----------------------------------------------------
    targets = ns.hosts if ns.hosts else list(range(1, 61))
    targets = [h for h in targets if 1 <= h <=60]
    if not targets:
        print("No valid hosts specified (must be 160).")
        sys.exit(1)

    # -------- load existing CSV (dedupe) -----------------------------------
    out_path = pathlib.Path(ns.outfile)
    all_urls = set()
    if out_path.exists():
        all_urls.update(p.strip() for p in out_path.read_text().splitlines() if p.strip())

    folders: dict[str, set[str]] = defaultdict(set)

    print(f"=== GN-AU collector – {len(targets)} host(s) ===")
    for i, n in enumerate(targets, 1):
        host = f"gnau{n}.operis.fr"
        print(f"[{i:02d}/{len(targets)}] {host}")
        cities = fetch_host(host, ns.retries, ns.base_back,
                            ns.throttle, ns.jitter, ns.timeout)
        folders[host] = cities
        print(f"    {len(cities)} folder(s) found")
        pause = ns.throttle + random.uniform(0, ns.jitter)
        time.sleep(max(0, pause))

    # -------- merge & write ------------------------------------------------
    for host, cities in folders.items():
        for city in cities:
            all_urls.add(f"https://{host}/{city}/")

    with out_path.open("w", encoding="utf-8") as f:
        for url in sorted(all_urls):
            f.write(url + "\n")

    print(f"\n=== Done. {len(all_urls)} unique URLs written to {out_path} ===")


if __name__ == "__main__":
    main()
