# src/lhw/cli.py

from __future__ import annotations

import random
import kagglehub
import argparse
import shutil
import psycopg
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List
import json
from sqlalchemy import create_engine, text as sa_text, bindparam, String, Numeric
from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.core import Document, SummaryIndex, VectorStoreIndex, Settings, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

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

    print(f"[DEBUG] Downloaded to: {download_path}")

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

    print(f"[INFO] Sampled {len(chosen)} files into: {SAMPLE_DIR.resolve()}")


def _pdf2txt_impl() -> None:
    """Convert all PDFs in src_dir to .txt files in dst_dir using LlamaIndex readers."""
    reader = PDFReader()

    for pdf in SAMPLE_DIR.glob("*.pdf"):
        docs = reader.load_data(file=pdf)
        text = "\n\n".join(d.text for d in docs)
        (SAMPLE_DIR / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")
        print(f"[INFO] {pdf.name} -> {pdf.stem}.txt [OK]")


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
        print(f"[INFO] {txt.name} -> {len(chunks)} [OK]")
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
                print(f"[INFO] {txt.name} [OK]")
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
    print(f"[DEBUG] Loaded {len(documents)} PDFs from {Path(SAMPLE_DIR).resolve()}")

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
    print(f"[INFO] Index created and persisted to ./{PERSIST_DIR}")


@dataclass
class CandidateInfo:
    doc_id: str
    name: str | None
    profession: str | None
    years_experience: float | None


def _prompt_llamauserinfo(text: str) -> str:
    return f"""
You extract structured data from a resume/CV.

Return a JSON object with EXACT keys:
- "name": full candidate name if present, else null.
- "profession": primary profession/role (concise), else null.
- "years_experience": number of years of commercial experience as a number (e.g., 5 or 5.5), else null.

Rules:
- Prefer explicit statements like "X years of experience".
- If not explicit but work history is clear, estimate from earliest professional start year to latest role; otherwise null.
- Answer with JSON ONLY. No prose.

Output format:
{{
name: <name>,
profession: <profession>,
years_experience: <number of years of experience>
}}

Resume:
<<<
{text}
>>>

""".strip()


def _llamauserinfo_impl() -> None:
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    vector_store = PGVectorStore.from_params(
        database=os.environ["POSTGRES_DB"],
        host="localhost",
        port="5432",
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        table_name="data_lhw_index"
    )

    engine = create_engine(vector_store.connection_string)
    schema = getattr(vector_store, "schema_name", "public")
    table = vector_store.table_name

    with engine.connect() as conn:
        rows = conn.execute(
            sa_text(f'SELECT node_id, text, metadata_ FROM "{schema}"."{table}" ORDER BY node_id')
        ).fetchall()

    print(f"[DEBUG] Total node_ids: {len(rows)}")

    docs: dict[str, list] = defaultdict(list)
    for node_id, text_value, metadata in rows:
        print(f"[DEBUG] {node_id} | meta={metadata}\n{'-' * 60}")
        file_name = metadata.get("file_name")
        docs[file_name].append(text_value)

    print(f"[DEBUG] Loaded docs:{len(docs)}")
    results: List[CandidateInfo] = []

    for file_name, nodes in docs.items():
        print(f"[DEBUG] Processing doc: {file_name} chunks {len(nodes)}...")
        text_parts = [n for n in nodes]
        text = "\n\n".join(text_parts)
        prompt = _prompt_llamauserinfo(text)
        resp = Settings.llm.complete(prompt)
        data = json.loads(resp.text)
        print(f"[DEBUG] Completed {data}")
        results.append(CandidateInfo(
            doc_id=file_name,
            name=data["name"],
            profession=data["profession"],
            years_experience=data["years_experience"]
        ))
        print("[DEBUG] ------------------------------------------------------------------------------------")

    with engine.begin() as conn:  # transaction w/ auto-commit on success
        for r in results:
            sql = sa_text(f"""
                UPDATE "{schema}"."{table}"
                SET metadata_ = (
                    COALESCE(metadata_::jsonb, '{{}}'::jsonb)
                    || jsonb_build_object(
                        'candidate_info',
                        jsonb_build_object(
                            'name', :name,
                            'profession', :profession,
                            'years_experience', :years_experience
                        )
                    )
                    || jsonb_build_object(
                        'name', :name,
                        'profession', :profession,
                        'years_experience', :years_experience
                    )
                )::json
                WHERE metadata_->>'file_name' = :file_name
                """).bindparams(
                bindparam("name", type_=String()),
                bindparam("profession", type_=String()),
                bindparam("years_experience", type_=Numeric(asdecimal=False)),
                bindparam("file_name", type_=String()),
            )

            params = {
                "file_name": r.doc_id,
                "name": (r.name or "").strip(),
                "profession": (r.profession or "").strip(),
                "years_experience": r.years_experience,
            }

            res = conn.execute(sql, params)
            print(f'[DEBUG] Updated {res.rowcount} rows for file "{r.doc_id}".')

    if results:
        header = f"{'FILE':40} | {'NAME':25} | {'PROFESSION':25} | YEARS"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{str(r.doc_id or '')[:40]:40} | "
                f"{str(r.name or '')[:25]:25} | "
                f"{str(r.profession or '')[:25]:25} | "
                f"{str(r.years_experience or '')}"
            )

    return results


def _prompt_llamafullinfo() -> str:
    return f"""You are summarizing a single candidate CV composed of multiple text chunks.

Return a concise JSON object with this schema:
{{
    "candidate_name": string|null,
    "file_name": string|null,
  "years_experience": number|null,
  "strongest_skills": [string],   // top 5 max, most distinctive/impactful first
  "professional_highlights": [    // 3–6 bullets, concrete achievements, impact, scope
    string
  ],
  "recent_roles": [               // up to 3: "Title — Company (Years or Dates)"
    string
  ]
}}

Rules:
- Derive only from the provided text. If a field is unknown, use null or [].
- Prefer explicit numbers for experience; if inferred, be conservative.
- Be specific and avoid generic buzzwords.
- Keep total under ~1300 characters.
"""


def _llamafullinfo_impl() -> None:
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
    vector_store = PGVectorStore.from_params(
        database=os.environ["POSTGRES_DB"],
        host="localhost",
        port="5432",
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        table_name="data_lhw_index"
    )

    engine = create_engine(vector_store.connection_string)
    schema = getattr(vector_store, "schema_name", "public")
    table = vector_store.table_name

    with engine.connect() as conn:
        rows = conn.execute(
            sa_text(f'SELECT node_id, text, metadata_ FROM "{schema}"."{table}" ORDER BY node_id')
        ).fetchall()

    print(f"[DEBUG] Total node_ids: {len(rows)}")

    chunks = []
    for node_id, text_value, metadata in rows:
        print(f"[DEBUG] {node_id} | meta={metadata}\n{'-' * 60}")
        chunks.append({"text_": text_value, "metadata_": metadata})

    docs = [Document(text=c["text_"], metadata=c["metadata_"]) for c in chunks]

    print(f"[DEBUG] Loaded chunks:{len(docs)}")

    sidx = SummaryIndex.from_documents(docs)
    qe = sidx.as_query_engine(streaming=True, response_mode="tree_summarize")

    stream = qe.query(_prompt_llamafullinfo())
    parts = []
    for token in stream.response_gen:
        parts.append(token)
        print(f"{token}", end="", flush=True)

    print("\n", flush=True)  # newline after the streamed block

    return "".join(parts)


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


def _cmd_llamauserinfo(args: argparse.Namespace) -> int:
    _llamauserinfo_impl()
    return 0


def _cmd_llamafullinfo(args: argparse.Namespace) -> int:
    _llamafullinfo_impl()
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

    # --- llama-index subcommand ---
    p_llamaindex = sub.add_parser("llama-index", help="Create llama-index")
    p_llamaindex.set_defaults(func=_cmd_llamaindex)

    # --- llama-user-info subcommand ---
    p_lllamauserinfo = sub.add_parser("llama-user-info", help="user info")
    p_lllamauserinfo.set_defaults(func=_cmd_llamauserinfo)

    # --- llama-full-info subcommand ---
    p_llamafullinfo = sub.add_parser("llama-full-info", help="full info")
    p_llamafullinfo.set_defaults(func=_cmd_llamafullinfo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
