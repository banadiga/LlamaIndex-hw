#!/bin/zsh

echo "Clean data" && rm -rf data/ && \
echo "Create venv" && python3 -m venv .venv && \
echo "Use venv" && source .venv/bin/activate && \
echo "Install dependency" &&  pip install . && \
echo "Download 30 pdf" &&  lhw download --sample 30 && \
echo "Convert 30 pdf to text" &&  lhw pdf2txt && \
echo "Create chunks" &&  lhw chunks --debug True && \
echo "Stop docker" && docker-compose down -v && \
echo "Start docker" && docker-compose up -d && \
echo "Done"