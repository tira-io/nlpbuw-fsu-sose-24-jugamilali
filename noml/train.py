from pathlib import Path

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import os


def find_best_threshold(df, true_labels, weighted_sum_column):
    best_threshold = 0
    best_accuracy = 0

    for threshold in np.arange(0, 1.0001, 0.0001):
        df['predicted_label'] = (df[weighted_sum_column] >= threshold).astype(int)
        accuracy = accuracy_score(true_labels, df['predicted_label'])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold

if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    df = text.join(labels)


    # Load the pre-trained Sentence-BERT model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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
    
    best_threshold = find_best_threshold(df, df['label'], 'weighted_sum')

    output_dir = get_output_directory(str(Path(__file__).parent))

    model_path = os.path.join(output_dir, "bert_model")
    model.save(model_path)

    threshold_path = os.path.join(output_dir, "threshold.txt")
    with open(threshold_path, 'w') as f:
        f.write(str(best_threshold))