from unidecode import unidecode
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    losses,
    util,
)
import logging
from sentence_transformers.readers import InputExample
from .constants import dic

logging = logging.getLogger(__name__)


def model_load(parameters):
    """
    Load the model based on parameters model-name
    Args:
        parameters: yaml file
    """
    logging.info("Loading the pre-trained model")
    return SentenceTransformer(parameters['url'])


def _pseudo_train(model, parameters):
    """
    Simple training using tuples and labels
    Args:
        model: model that you choose to use as pre-trained embeddings
        parameters: yaml file
    """
    train_examples = [
        InputExample(
            texts=[
                "homem",
                "mulher",
            ],
            label=0.97,
        ),
        InputExample(
            texts=[
                "carro",
                "veículo",
            ],
            label=0.99,
        ),
    ]
    logging.info("Pseudo-training the model only for test purposes")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)
    logging.info("Training the model")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=parameters['epochs'],
        warmup_steps=parameters['warmup_steps'],
    )

    return model


def _real_training(model, parameters):
    # We didn't create any dataframe for training because depends exactly on which data you are working.
    # If you want to understand better how we can train, check sentence-transformers documentation.
    pass


def cleaner(df: pd.DataFrame, col: str):
    """
    Clean any dataframe quickly without using any default library.
    I didn't use any nltk corpus to remove specific lexicons, lemmatizations and so on
    But feel free to use.
    Args:
        df: corpus and queries dataframe
        col: name of the column that you want to clean-up
    """
    logging.info(f"Cleaning {col} column")
    for v, j in dic.items():
        df[col] = df[col].str.lower().replace(v, j, regex=True)
    df[col] = df[col].apply(unidecode)
    df[col] = (
        df[col]
        .str.split()
        .apply(lambda x: OrderedDict.fromkeys(x).keys())
        .str.join(" ")
    )
    df[col] = df[col].str.replace(r"\s+$", "", regex=True)

    return df[col].tolist()


def _encoder_normalizer(model, data):
    """
    This will encode your data and create every embedding using
    pre-trained model. We're going to use the CUDA too.
    Be carefull for perfomance issues
    Args:
        model: pre-trained model
        data: dataframe
    """
    corpus_embeddings = model.encode(data, convert_to_tensor=True).to("cuda")

    return util.normalize_embeddings(corpus_embeddings)


def _progress_worker(model, dataset_search, dataset_corpus, parameters):
    """
    This function will use semantic search function to create a cosine similarity score
    between embeddings on both datasets. And will return everything without change other values (even
    if we clean the column)
    Args:
        model: pre-trained model
        dataset_search: search csv
        dataset_corpus: corpus csv
        parameters: yaml config
    """
    logging.info("Converting for-match columns to list")
    queries = cleaner(dataset_search, parameters['dataset_from_column'])
    corpus = cleaner(dataset_corpus, parameters['dataset_to_column'])

    logging.info("Encoder normalizing queries and corpus")
    hits = util.semantic_search(
        _encoder_normalizer(model, queries),
        _encoder_normalizer(model, corpus),
        score_function=util.dot_score,
        top_k=1,
    )
    logging.info("Rollback to dataframe structure")
    data = pd.DataFrame(hits).stack().apply(pd.Series)
    data["Query"] = queries
    data["Corpus"] = [corpus[i] for i in data.corpus_id.astype(int).tolist()]

    return data
