"""
One-shot ARSO archive backfill → existing sensors (metadata.sifra).

Not used by the continuous hydro scraper. Default mode is dry-run
(no writes) unless --ingest is passed.

Examples:
  python arso_historic_import.py --sifra 9275 --from 1994 --to 1994
  python arso_historic_import.py --ingest --sifra 9275 --from 1994 --to 1994
  python arso_historic_import.py --all-stations --from 2000 --to 2000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from arso_historic.runner import run

# Inclusive bounds for --from / --to
_YEAR_MIN = 1900


def main() -> int:
    current_year = datetime.now(timezone.utc).year

    parser = argparse.ArgumentParser(
        description="ARSO historic daily CSV → known sensors (sifra)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch/parse/report only (default; same as omitting --ingest)",
    )
    mode.add_argument(
        "--ingest",
        action="store_true",
        help="Write via POST /dataIngest (requires source column)",
    )
    parser.add_argument(
        "--sifra",
        default=None,
        help="Single ARSO station id (required unless --all-stations)",
    )
    parser.add_argument(
        "--all-stations",
        action="store_true",
        help="Process every sensor that has sifra (explicit bulk)",
    )
    parser.add_argument(
        "--from",
        dest="year_from",
        type=int,
        default=None,
        help=f"Start year inclusive (default: {_YEAR_MIN}; required with --all-stations)",
    )
    parser.add_argument(
        "--to",
        dest="year_to",
        type=int,
        default=current_year,
        help=f"End year inclusive (default: {current_year})",
    )
    parser.add_argument(
        "--api",
        default=None,
        help="Middleware API base URL (default: MIDDLEWARE_API or localhost:5006)",
    )
    parser.add_argument(
        "--all-statuses",
        action="store_true",
        help="Include inactive sensors (default: active only)",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report to stdout",
    )
    args = parser.parse_args()

    # --- CLI validation ---
    sifra = (args.sifra or "").strip() or None
    if not sifra and not args.all_stations:
        print(
            "error: pass --sifra <id> or --all-stations for bulk",
            file=sys.stderr,
        )
        return 2
    if sifra and args.all_stations:
        print("error: use either --sifra or --all-stations, not both", file=sys.stderr)
        return 2
    # Bulk without an explicit start year is too easy to over-fetch
    if args.all_stations and args.year_from is None:
        print("error: --all-stations requires explicit --from", file=sys.stderr)
        return 2

    year_from = args.year_from if args.year_from is not None else _YEAR_MIN
    year_to = args.year_to

    if year_from > year_to:
        print("error: --from must be <= --to", file=sys.stderr)
        return 2
    if year_from < _YEAR_MIN or year_to > current_year + 1:
        print(
            f"error: years must be in [{_YEAR_MIN}, {current_year + 1}]",
            file=sys.stderr,
        )
        return 2
    if args.chunk_size < 1:
        print("error: --chunk-size must be >= 1", file=sys.stderr)
        return 2

    # Writes only with --ingest; --dry-run is optional explicit no-op
    dry_run = not args.ingest

    try:
        result = run(
            sifra=sifra,
            year_from=year_from,
            year_to=year_to,
            dry_run=dry_run,
            active_only=not args.all_statuses,
            api_url=args.api,
            chunk_size=args.chunk_size,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return _exit_code(result, dry_run)

    mode_label = "DRY-RUN" if result["dry_run"] else "INGEST"
    print(
        f"[{mode_label}] stations={result['station_count']} "
        f"years={result['year_from']}-{result['year_to']} "
        f"ready={result['totals']['ready']} "
        f"ingested={result['totals']['ingested']} "
        f"failed={result['totals']['failed_stations']}"
    )
    for report in result["reports"]:
        if "error" in report:
            print(f"  sifra={report['sifra']} ERROR: {report['error']}")
            continue
        stats = report["stats"]
        print(
            f"  sifra={report['sifra']} "
            f"kept={stats['kept']} "
            f"ready={stats['ready_to_ingest']} "
            f"empty={stats['skipped_empty']} "
            f"out_of_range={stats['skipped_out_of_range']} "
            f"blocked={stats.get('skipped_blocked_unit', 0)} "
            f"bad_ts={stats.get('skipped_bad_ts', 0)} "
            f"bad_value={stats.get('skipped_bad_value', 0)}"
        )
        for st, meta in report.get("sensors", {}).items():
            print(
                f"    {st}: id={meta.get('sensor_id')} "
                f"unit={meta.get('unit')} "
                f"label={meta.get('sensor_label')}"
            )
        for note in report.get("conversions", {}).values():
            print(f"    convert: {note}")
        for warning in report.get("warnings", []):
            print(f"    WARN: {warning}")
        if report.get("sample"):
            print(f"    sample: {report['sample'][:2]}")

    return _exit_code(result, dry_run)


def _exit_code(result: dict, dry_run: bool) -> int:
    """0 = ok work done; 1 = station failures or nothing processed."""
    if result["totals"]["failed_stations"]:
        return 1
    if result["station_count"] == 0:
        return 1
    # Ingest with nothing written and nothing ready is a soft failure
    if (
        not dry_run
        and result["totals"]["ingested"] == 0
        and result["totals"]["ready"] == 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
