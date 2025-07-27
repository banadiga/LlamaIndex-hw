# src/lhw/cli.py

from __future__ import annotations

import random
import kagglehub
import argparse
import shutil
import psycopg
import os

from pathlib import Path

from llama_index.readers.file import PDFReader
from llama_index.core import VectorStoreIndex, Settings, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

DATASET = "snehaanbhawal/resume-dataset"
DATA = "data"
SAMPLE_DIR = Path(DATA) / "sample"
PERSIST_DIR = Path(DATA) / "lhw-index"

def _download_impl(sample: int) -> None:
    """Your real download logic lives here so it's testable."""
    # Load the latest version
    download_path = Path(
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


def _chunks_impl(chunk_size: int, chunk_overlap: int, paragraph_separator: str, debug: bool) -> None:
    """Sentence splitter."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator=paragraph_separator,
    )
    for txt in SAMPLE_DIR.glob("*.txt"):
        cv_text = open(txt, encoding="utf-8").read()
        chunks = splitter.split_text(cv_text)
        print(f"{txt.name} -> {len(chunks)} [OK]")
        if debug:
            # write one file per chunk
            for i, ch in enumerate(chunks, 1):
                (SAMPLE_DIR / f"{txt.stem}_chunk_{i:03d}.txt").write_text(ch, encoding="utf-8")


def _embeddings_impl() -> None:
    """Embeddings"""
    embedder = OpenAIEmbedding(model="text-embedding-3-small")

    POSTGRES_USER = os.getenv("POSTGRES_USER", "")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "embeddings")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    DSN = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    with psycopg.connect(DSN, row_factory=dict_row) as conn:
        register_vector(conn)  # teach psycopg how to send/receive vector
        with conn.cursor() as cur:
            for txt in SAMPLE_DIR.glob("*_chunk_*.txt"):
                cv_text = open(txt, encoding="utf-8").read()
                embedding = embedder.get_text_embedding(cv_text)
                chunk_no = int(txt.stem.split("_chunk_")[-1])
                print(f"{txt.name} [OK]")
                cur.execute(
                    """
                    INSERT INTO cv_chunks (file_name, chunk_no, content, embedding)
                    VALUES (%s, %s, %s, %s) ON CONFLICT (file_name, chunk_no)
                                    DO
                    UPDATE SET content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        created_at = now();
                    """,
                    (txt.name, chunk_no, cv_text, embedding),
                )
        conn.commit()


def _llamaindex_impl() -> None:
    reader = SimpleDirectoryReader(
        input_dir=SAMPLE_DIR,
        required_exts=[".pdf"],
        recursive=False,
        filename_as_id=True,
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} PDFs from {Path(SAMPLE_DIR).resolve()}")

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    vector_store = PGVectorStore.from_params(
        database=os.environ["POSTGRES_DB"],
        host="localhost",
        port=5432,
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        table_name="lhw_index"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)],
        show_progress=True,
    )

    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index created and persisted to ./{PERSIST_DIR}")


def _cmd_download(args: argparse.Namespace) -> int:
    _download_impl(sample=args.sample)
    return 0


def _cmd_pdf2txt(args: argparse.Namespace) -> int:
    _pdf2txt_impl()
    return 0


def _cmd_chunks(args: argparse.Namespace) -> int:
    _chunks_impl(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                 paragraph_separator=args.paragraph_separator, debug=args.debug)
    return 0


def _cmd_embeddings(args: argparse.Namespace) -> int:
    _embeddings_impl()
    return 0


def _cmd_llamaindex(args: argparse.Namespace) -> int:
    _llamaindex_impl()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the multi-command CLI (lhw)."""
    parser = argparse.ArgumentParser(prog="lhw", description="Resume tools CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- download subcommand ---
    p_dl = sub.add_parser("download", help="Download resumes dataset")
    p_dl.add_argument("--sample", type=int, default=50, help="How many items to download")
    p_dl.set_defaults(func=_cmd_download)

    # --- download subcommand ---
    p_p2t = sub.add_parser("pdf2txt", help="PDF to text conversion")
    p_p2t.set_defaults(func=_cmd_pdf2txt)

    # --- chunks subcommand ---
    p_chunks = sub.add_parser("chunks", help="Sentence splitter")
    p_chunks.add_argument("--chunk_size", type=int, default=512, help="Size of chunks")
    p_chunks.add_argument("--chunk_overlap", type=int, default=64, help="Overlap of chunks")
    p_chunks.add_argument("--paragraph_separator", type=str, default="\n\n", help="Separator between paragraphs")
    p_chunks.add_argument("--debug", type=bool, default=False, help="Debug mode")
    p_chunks.set_defaults(func=_cmd_chunks)

    # --- embeddings subcommand ---
    p_embeddings = sub.add_parser("embeddings", help="embeddings")
    p_embeddings.set_defaults(func=_cmd_embeddings)

    # --- embeddings subcommand ---
    p_llamaindex = sub.add_parser("llama-index", help="Create llama-index")
    p_llamaindex.set_defaults(func=_cmd_llamaindex)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
