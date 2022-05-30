from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Maximal Marginal Relevance 적용
# 문서와 가장 유사한 키워드/키프레이즈를 선택하는 것으로 시작합니다. 그런 다음 문서와 유사하고 이미 선택한 키워드/키프레이즈와 유사하지 않은 새 후보를 반복적으로 선택

def extractKeywords(doc):

    n_gram_range = (1, 1)
    stop_words = "english"


    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names()



    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-distilroberta-v1')
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)

    diversity = 0.4

    top_n = 5
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)
    word_similarity = cosine_similarity(candidate_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(candidates)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [candidates[idx] for idx in keywords_idx]


doc = """ Supervised learning is the machine learning task of learning a function that maps an input to an output basedon example input-output pairs.[1] It infers a function from labeled training data consisting of a set of training examples.[2] In supervised learning, each example is a pair consisting of an input object (typically a vector) and a desired output value (also called the supervisory signal). A supervised learning  algorithm analyzes the training data and produces an  inferred function, which can be used for mapping new examples. An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances. This requires the learning algorithm to  generalize from the training data to unseen situations in a 'reasonable' way (see inductive bias)."""

result = extractKeywords(doc)

print(result)
