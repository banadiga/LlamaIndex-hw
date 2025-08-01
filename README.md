# LlamaIndex hw

This involves reading CV files from

- [x] Loading CV Files: 
This involves reading CV files from https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset 
You can choose up to 20-30 CVs.
- [x] Split CV into small meaningful chunks.
- [x] Generating Embeddings: Convert the parsed data into numerical representations (embeddings) that can be easily 
processed by machine learning algorithms. This typically involves using techniques like word embeddings or sentence 
embeddings.
- [x] Storing Embeddings in a Vector Database: Save the generated embeddings into a vector database. As a vector store, 
you can choose **PostgreSQL**~~, ChromaDB, FAISS, etc,~~
- [x] Storing Embeddings in a Vector Database: Save the generated embeddings into a vector database. As a vector store, 
you can choose **PostgreSQL** with llama index.
- [x] Retrieving Candidate Details: Extract and display specific information about each  candidate, such as name, 
profession, and years of commercial experience.
- [x] Generating Experience Summary: Based on the parsed data and embeddings, generate a summary of each candidate’s 
strongest skills and professional highlights.
Important note:
The task should be done using LlamaIndex.

Expected outcome:
The repository contains a straightforward web application that lists candidates. Users can click on any candidate to 
view detailed information and a summary of their profile.


# One script
To run HW as one script use [start.sh](start.sh)

```bash
sh start.sh
```

# Step by step

## PostgreSQL
Use docker compose to start up *PostgreSQL* as a vector store.

```bash
docker-compose up
```

## python env

Create python virtual env

```bash
python3 -m venv .venv
```

activate it
```bash
source .venv/bin/activate
```

install dependency and recreate scripts
```bash
pip install .
```

## Download 
To download CV files from
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

```bash
lhw download --sample 30
```

## Convert PDF to TXT

```bash
lhw pdf2txt
```

## TXT to chunks

```bash
lhw chunks --debug True
```

## Embeddings

```bash
lhw embeddings
```

## Create llama-index

```bash
lhw llama-index
```

## Create llama-group-info
Retrieving Candidate Details: Extract and display specific information about each  candidate, such as name, 
profession, and years of commercial experience.

```bash
lhw llama-group-info
```