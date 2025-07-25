CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cv_chunks (
    id         BIGSERIAL PRIMARY KEY,
    file_name  TEXT NOT NULL,
    chunk_no   INT  NOT NULL,
    content    TEXT NOT NULL,
    embedding  VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (file_name, chunk_no)
);

CREATE INDEX IF NOT EXISTS cv_chunks_embedding_idx
    ON cv_chunks
 USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

ANALYZE cv_chunks;
