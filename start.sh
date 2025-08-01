#!/bin/zsh

echo "Stop docker" && docker-compose down -v && \
echo "Start docker" && docker-compose up -d && \
echo "Clean data" && rm -rf data/ && \
echo "Create venv" && python3 -m venv .venv && \
echo "Use venv" && source .venv/bin/activate && \
echo "Install dependency" && pip install . && \
echo "Download pdf" && lhw download --sample 30 && \
echo "llama-index" && lhw llama-index && \
echo "llama-user-info" && lhw llama-user-info && \
echo "llama-full-info" && lhw llama-full-info && \
echo "Done"
