FROM python:3.12-slim

WORKDIR /srv
COPY pyproject.toml README.md ./
COPY pl2016 ./pl2016
RUN pip install --no-cache-dir . \
    && python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"
COPY artifacts ./artifacts
ENV PL2016_ARTIFACTS=/srv/artifacts
EXPOSE 8080
CMD ["python", "-m", "pl2016.app"]
