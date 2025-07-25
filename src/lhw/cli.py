# src/lhw/cli.py

from __future__ import annotations

import random
import kagglehub
import argparse
import shutil
from pathlib import Path

from llama_index.readers.file import PDFReader

DATASET = "snehaanbhawal/resume-dataset"
DATA = "data"
SAMPLE_DIR = Path(DATA) / "sample"

def _download_impl(sample: int) -> None:
    """Your real download logic lives here so it's testable."""
    # Load the latest version
    download_path =  Path(
        kagglehub.dataset_download(
        "snehaanbhawal/resume-dataset",
        )
    )

    print(f"Downloaded to: {download_path}")

    SAMPLE_DIR.mkdir(parents=True)

    # save samples
    all_files = [
        p for p in download_path.rglob("*")
        if p.is_file()
    ]
    chosen = random.sample(all_files, min(sample, len(all_files)))

    for src in chosen:
        dst = SAMPLE_DIR / src.name
        shutil.copy2(src, dst)

    print(f"Sampled {len(chosen)} files into: {SAMPLE_DIR.resolve()}")


def _pdf2txt_impl() -> None:
    """Convert all PDFs in src_dir to .txt files in dst_dir using LlamaIndex readers."""
    reader = PDFReader()

    for pdf in SAMPLE_DIR.glob("*.pdf"):
        docs = reader.load_data(file=pdf)
        text = "\n\n".join(d.text for d in docs)
        (SAMPLE_DIR / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
        print(f"{pdf.name} -> {pdf.stem}.txt [OK]")

def _cmd_download(args: argparse.Namespace) -> int:
    _download_impl(sample=args.sample)
    return 0

def _cmd_pdf2txt(args: argparse.Namespace) -> int:
    _pdf2txt_impl()
    return 0


def download() -> int:
    """
    Entry point for downloading resumes dataset.
    """
    parser = argparse.ArgumentParser(prog="download-resumes", description="Download resumes dataset")
    parser.add_argument("--sample", type=int, default=40)
    args = parser.parse_args()
    return _cmd_download(args)

def main(argv: list[str] | None = None) -> int:
    """Entry point for the multi-command CLI (lhw)."""
    parser = argparse.ArgumentParser(prog="lhw", description="Resume tools CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- download subcommand ---
    p_dl = sub.add_parser("download", help="Download resumes dataset")
    p_dl.add_argument("--sample", type=int, default=50, help="How many items to download (demo)")
    p_dl.set_defaults(func=_cmd_download)

    # --- download subcommand ---
    p_p2t = sub.add_parser("pdf2txt", help="PDF to text conversion")
    p_p2t.set_defaults(func=_cmd_pdf2txt)

    args = parser.parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())