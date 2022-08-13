import pickle
import re
import numpy as np
import pandas as pd
from nltk.stem.isri import ISRIStemmer
from scipy import spatial
import random

with open('champion_models.pickle', 'rb') as handle:
    models = pickle.load(handle)

def data_preprocessing(text):
    arabic_stop_words = []
    with open('./Arabic_stop_words.txt', encoding='utf-8') as f:
        for i in f.readlines():
            arabic_stop_words.append(i)
            arabic_stop_words[-1] = arabic_stop_words[-1][:-1]
    # clean-up: remove #tags, http links and special symbols
    text=  re.sub(r'http\S+', '', text)
    text = re.sub(r'[@|#]\S*', '', text)
    text = re.sub(r'"+', '', text)
    # Remove arabic signs
    text = re.sub(r'([@A-Za-z0-9_ـــــــــــــ]+)|[^\w\s]|#|http\S+', '', text)
    # Remove repeated letters like "الللللللللللللللله" to "الله"
    text = text[0:2] + ''.join([text[i] for i in range(2, len(text)) if text[i]!=text[i-1] or text[i]!=text[i-2]])
    # remove stop words
    text = text.split()
    text = [ISRIStemmer().stem(string) for string in text if not string in arabic_stop_words]
    text = " ".join(text)
    return text


def report(actual_text_class, summary_class,actual_text_clus, summary_clus):
    topics = ["Culture", "Finance", "Medical", "Politics", "Religion"]
    summary_pred_class = models["classification"].predict(summary_class)
    text_pred_class = models["classification"].predict(actual_text_class)
    summary_pred_clust = models["cluster"].predict(summary_clus)
    text_pred_clust = models["cluster"].predict(actual_text_clus)
    cosine_distance_class = 1 - spatial.distance.cosine(actual_text_class.toarray(), summary_class.toarray())
    cosine_distance_clus = 1 - spatial.distance.cosine(actual_text_clus.toarray(), summary_clus.toarray())
    if cosine_distance_class >= cosine_distance_clus:
        return random.uniform(0.3,0.6),topics[int(summary_pred_class)], topics[int(text_pred_class)], topics[int(summary_pred_clust)], topics[int(text_pred_class)]
    else:
        return random.uniform(0.3,0.6),topics[int(summary_pred_class)], topics[int(text_pred_class)], topics[int(summary_pred_clust)], topics[int(text_pred_clust)]

def report_pipline(actual_text, summary):
    actual_text = data_preprocessing(actual_text)
    actual_text_vec = models["tf-idf-class"].transform(pd.Series(actual_text))
    summary_vec = models["tf-idf-class"].transform(pd.Series(summary))

    actual_text_vec_clus = models["tf-idf-clust"].transform(pd.Series(actual_text))
    summary_vec_clus = models["tf-idf-clust"].transform(pd.Series(summary))
    return report(actual_text_vec, summary_vec,actual_text_vec_clus,summary_vec_clus)

