# Pocket Lawyer, 2016 technique

A rebuild of the 2016 retrieval pipeline: classify a lay question into an area of law, then rank New York or California statutes and Nolo articles inside that area with TF-IDF and cosine similarity. The classifier is trained on r/legaladvice flairs (Li et al. 2022, `jonathanli/legal-advice-reddit`) because FindLaw Answers no longer exists. Federal statutes are out of scope.

The notebook and README at the repository root are the 2016 artifact. Nothing in this directory is allowed to modify them.

## Run

```bash
cd 2016
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
docker compose up -d
pytest
```

MongoDB listens on `127.0.0.1:27017` only. Scraped pages and trained models go to `data/` and `artifacts/`, which are gitignored.

Serving does not use Mongo. After the artifacts exist:

```bash
python -m pl2016.app
```

`POST /ask` takes `{"question": "...", "state": "NY"}`. `GET /health` and `GET /manifest` are open. The image is `docker build -t pl2016 .` from this directory, which copies `artifacts/`.
