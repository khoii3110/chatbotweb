from underthesea import ner, sent_tokenize
import numpy as np
import pickle
import nltk
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans


def tom_tat_van_ban(content):
    content_parsed = content.lower().strip()
    sentences = sent_tokenize(content_parsed)

    num_sent = len(sentences)

    # Sử dụng NER để lấy từ được gắn nhãn
    word_labels = []
    for sentence in sentences:
        sentence_labels = ner(sentence)
        word_labels.extend([label[0] for label in sentence_labels])

    filtered_words = [word for word in word_labels if word not in ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']]
    # Biểu diễn vector cho mỗi câu
    X = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        sentence_vec = np.zeros((len(filtered_words)))
        for idx, word in enumerate(filtered_words):
            if word in words:
                sentence_vec[idx] = 1
        X.append(sentence_vec)

    # Tính số cụm cho K-means
    n_clusters = int(num_sent * (35/100))
    if n_clusters >= (num_sent - 1):
        return 'Văn bản quá ngắn! nhập văn bản dài hơn tôi sẽ tóm tắt giúp bạn'
    else:
        # Áp dụng K-means
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans = kmeans.fit(X)

        #tinh tb
        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))

        # Tìm câu gần nhất với trung tâm của mỗi cụm
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])
        summary = ' '.join([sentences[closest[idx]] for idx in ordering])
        return f'Văn bản đã được tóm tắt của bạn:\n{summary}'

# content = input("Nhập vào văn bản: ")
# print(tom_tat_van_ban(content))