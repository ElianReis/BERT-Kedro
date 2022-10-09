# transformers-match

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation and `src/environment.yml` for `conda` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How it works

Using the Data Catalog orchestration we can separate search and base data to create our sentence pairing.


> Queries: data/01_raw/queries.csv
> 
> Corpus: data/01_raw/corpus.csv

They are adapted to create match between two csv simple files. Just put the data you want
and try to run the kedro pipeline with CUDA + sentece-transformers.

## Configuration of the model

> conf/base/parameters.yml

## For more info:
> Check **sentence-transformers** and **kedro** documentation:
> 
> https://www.sbert.net/
> 
> https://kedro.readthedocs.io/en/stable/