from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


VALID_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_BASE_URL = "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/resolve/main"


def _download_file(url: str, out_path: Path, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"Keeping existing file: {out_path}")
        return

    print(f"Downloading: {url}")
    with urllib.request.urlopen(url) as response, out_path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"Wrote: {out_path}")


def _remote_filename(kind: str, subset: str) -> str:
    if kind == "rul":
        return f"RUL_{subset}.txt"
    return f"{kind}_{subset}.txt"


def _prepare_subset(base_url: str, subset: str, data_dir: Path, force: bool) -> None:
    mapping = {
        "train": data_dir / f"train_{subset}.txt",
        "test": data_dir / f"test_{subset}.txt",
        "rul": data_dir / f"RUL_{subset}.txt",
    }
    for kind, out_path in mapping.items():
        remote_name = _remote_filename(kind, subset)
        url = f"{base_url.rstrip('/')}/{remote_name}"
        _download_file(url, out_path, force=force)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare NASA C-MAPSS files for this project."
    )
    parser.add_argument(
        "--subset",
        default="FD001",
        help="Subset to prepare (FD001/FD002/FD003/FD004) or 'all'. Default: FD001",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for prepared files. Default: data",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL containing train_FDxxx.txt, test_FDxxx.txt, RUL_FDxxx.txt files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subset_arg = args.subset.upper()

    if subset_arg == "ALL":
        subsets = list(VALID_SUBSETS)
    elif subset_arg in VALID_SUBSETS:
        subsets = [subset_arg]
    else:
        print(f"Invalid subset: {args.subset}. Use one of {VALID_SUBSETS} or 'all'.", file=sys.stderr)
        return 2

    try:
        for subset in subsets:
            print(f"\nPreparing subset: {subset}")
            _prepare_subset(args.base_url, subset, args.data_dir, force=args.force)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to prepare dataset: {exc}", file=sys.stderr)
        return 1

    print("\nDone. Prepared files:")
    for subset in subsets:
        print(f"- {args.data_dir / f'train_{subset}.txt'}")
        print(f"- {args.data_dir / f'test_{subset}.txt'}")
        print(f"- {args.data_dir / f'RUL_{subset}.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
