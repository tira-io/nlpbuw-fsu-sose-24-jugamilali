from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    model = SentenceTransformer(str(Path(__file__).parent / "bert_model"))
    
    threshold = 0.9287000000000001

    # Compute embeddings for each sentence
    embeddings1 = model.encode(df['sentence1'].tolist())
    embeddings2 = model.encode(df['sentence2'].tolist())

    # Calculate cosine similarities
    cosine_similarities = [cosine_similarity([e1], [e2])[0][0] for e1, e2 in zip(embeddings1, embeddings2)]
    # Add cosine similarities to the DataFrame
    df['cosine_similarity'] = cosine_similarities

    # Calculate Pearson correlations
    pearson_correlations = [np.corrcoef(e1, e2)[0, 1] for e1, e2 in zip(embeddings1, embeddings2)]
    # Add distances to the DataFrame
    df['pearson_correlation'] = pearson_correlations

    # Define weights for each metric
    weights = {
        'cosine_similarity': 0.4,
        'pearson_correlation': 1,
    }

    # Calculate the weighted sum of the metrics
    df['weighted_sum'] = (
        df['cosine_similarity'] * weights['cosine_similarity'] +
        df['pearson_correlation'] * weights['pearson_correlation']
    )

    df["label"] = (df["weighted_sum"] >= threshold).astype(int)
    df = df.drop(columns=["weighted_sum", "sentence1", "sentence2", "pearson_correlation", "cosine_similarity"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
