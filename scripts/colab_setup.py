"""
Colab setup: copy training data from Google Drive into finetuning/output.

Expects on Google Drive either:
  - A folder named glm_ocr_output (inside My Drive) containing:
    labels.json, train.txt, test.txt, and *.png images
  - Or a zip file glm_ocr_output.zip with the same contents

Run from repo root (e.g. after git clone):
  python scripts/colab_setup.py
  python scripts/colab_setup.py --drive-path "My Drive/glm_ocr_output"
"""
import argparse
import shutil
import sys
import zipfile
from pathlib import Path

# Repo root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "finetuning" / "output"

# Default path inside mounted Drive (Colab mounts at /content/drive)
DEFAULT_DRIVE_PATH = Path("/content/drive/MyDrive/glm_ocr_output")


def _flatten_single_subdir(out_dir: Path):
    """If out_dir has a single subdir and no labels.json at top level, move subdir contents up."""
    subdirs = [d for d in out_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and not (out_dir / "labels.json").is_file():
        single = subdirs[0]
        for item in single.iterdir():
            shutil.move(str(item), str(out_dir / item.name))
        single.rmdir()


def is_colab():
    return "google.colab" in sys.modules


def mount_drive():
    if not Path("/content/drive").exists() or not list(Path("/content/drive").iterdir()):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:
            print("Mount failed. In Colab, run the cell that mounts Drive first.", file=sys.stderr)
            raise SystemExit(1) from e


def main():
    parser = argparse.ArgumentParser(description="Copy glm_ocr training data from Google Drive to finetuning/output.")
    parser.add_argument(
        "--drive-path",
        type=Path,
        default=DEFAULT_DRIVE_PATH,
        help="Path to folder or zip on Drive (default: /content/drive/MyDrive/glm_ocr_output).",
    )
    parser.add_argument(
        "--no-mount",
        action="store_true",
        help="Do not attempt to mount Drive (already mounted).",
    )
    args = parser.parse_args()

    drive_path = args.drive_path.resolve()

    if is_colab() and not args.no_mount:
        mount_drive()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer folder, then zip
    if drive_path.is_dir():
        # Copy contents (not the folder itself) so labels.json, train.txt, *.png land in output/
        for item in drive_path.iterdir():
            dest = OUTPUT_DIR / item.name
            if item.is_file():
                shutil.copy2(item, dest)
            else:
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
        print(f"Copied contents of {drive_path} -> {OUTPUT_DIR}")
    elif Path(str(drive_path) + ".zip").is_file():
        zip_path = Path(str(drive_path) + ".zip")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(OUTPUT_DIR)
        _flatten_single_subdir(OUTPUT_DIR)
        print(f"Unzipped {zip_path} -> {OUTPUT_DIR}")
    elif drive_path.with_suffix(".zip").is_file():
        zip_path = drive_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(OUTPUT_DIR)
        _flatten_single_subdir(OUTPUT_DIR)
        print(f"Unzipped {zip_path} -> {OUTPUT_DIR}")
    else:
        print("Training data not found on Drive.", file=sys.stderr)
        print("Upload your finetuning/output folder to Google Drive:", file=sys.stderr)
        print("  1. Zip the folder: zip -r glm_ocr_output.zip labels.json train.txt test.txt *.png", file=sys.stderr)
        print("  2. Upload glm_ocr_output.zip to My Drive (or create folder glm_ocr_output and upload contents).", file=sys.stderr)
        print(f"  3. Expected path: {DEFAULT_DRIVE_PATH} (folder) or {DEFAULT_DRIVE_PATH}.zip", file=sys.stderr)
        raise SystemExit(1)

    # Sanity check
    required = ["labels.json", "train.txt", "test.txt"]
    for name in required:
        if not (OUTPUT_DIR / name).is_file():
            print(f"Warning: missing {name} in {OUTPUT_DIR}", file=sys.stderr)
    n_png = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"Done. {OUTPUT_DIR} has {n_png} PNGs and {[f for f in required if (OUTPUT_DIR / f).is_file()]}")


if __name__ == "__main__":
    main()
