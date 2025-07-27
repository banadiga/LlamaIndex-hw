CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS lhw_index (
    id         BIGSERIAL PRIMARY KEY,
    file_name  TEXT NOT NULL,
    chunk_no   INT  NOT NULL,
    content    TEXT NOT NULL,
    embedding  VECTOR(1536),
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (file_name, chunk_no)
);

CREATE INDEX IF NOT EXISTS lhw_index_embedding_idx
    ON lhw_index
 USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

ANALYZE lhw_index;
