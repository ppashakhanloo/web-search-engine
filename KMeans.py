import os
import re
import math
import json
import numpy
import random
from itertools import repeat
import matplotlib.pyplot as plt


def has_numbers(inp):
    _digits = re.compile('\d')
    return bool(_digits.search(inp))


def normalize_term(inp):
    t = ''
    for c in inp:
        if c.isalpha():
            t += c
    return t.lower()


def log_tf(x):
    if x > 0.0:
        return numpy.log10(x) + 1.0
    else:
        return 0.0


log_tf = numpy.vectorize(log_tf)

dictionary = []  # term
doc_vectors = {}  # (docID -> {})
term_df = {}  # (term -> df)

STOP_LIST = ['been', 'discussion', 'not', 'these', 'our', 'paper', 'can', 'show', 'shows', 'I','a','about','an','are','as', 'article', 'at', 'be', 'by', 'com', 'for', 'from','how','in', 'is', 'it', 'of', 'on', 'or', 'that', 'this','to', 'was','what','when','where','who', 'will', 'with','the','www', 'and', 'we']


def create_vector_space(dictionary):
    path = "json_items/"
    N = len(os.listdir(path))

    print('creating dictionary...')
    # create dictionary and df
    for file in os.listdir(path):
        json_data = open(path + '/' + file.title(), encoding='utf8').read()
        doc = json.loads(json_data)

        doc_content = ' '.join([str(doc['abstract']).lower(), str(doc['title']).lower()]).replace('\\n', ' ').replace(
            '-',
            ' ')
        doc_content_split = doc_content.split()
        for term in doc_content_split:
            temp = normalize_term(term)
            if (temp not in dictionary) and (temp != '') and (temp not in STOP_LIST):
                dictionary.append(temp)

        # update df
        doc_term_set = []
        for term in doc_content_split:
            temp = normalize_term(term)
            if temp is not '' and temp not in STOP_LIST:
                if temp not in doc_term_set:
                    doc_term_set.append(temp)

        for term in doc_term_set:
            if term not in term_df and term not in STOP_LIST:
                term_df.update({term: 1})
            elif term not in STOP_LIST:
                term_df.update({term: term_df.get(term) + 1})

    # sort the dictionary
    dictionary = list(sorted(dictionary))
    print('dictionary:')
    print(dictionary)

    print('calculate tf-idf...')
    # obtain file statistics - tf
    for file in os.listdir(path):
        json_data = open(path + '/' + file.title(), encoding='utf-8').read()
        doc = json.loads(json_data)
        doc_content = ' '.join([str(doc['abstract']).lower(), str(doc['title']).lower()]).replace('\\n', ' ').replace(
            '-',
            ' ')

        doc_vector = numpy.zeros(len(dictionary))
        doc_content_split = doc_content.split()

        for term in doc_content_split:
            temp = normalize_term(term)
            if temp is not '' and (temp not in STOP_LIST):
                temp_id = dictionary.index(temp)
                doc_vector[temp_id] += 1

        doc_vector = log_tf(doc_vector)
        for i in range(len(doc_vector)):
            df = numpy.log10(N / term_df.get(dictionary[i]))
            doc_vector[i] *= df

        doc_vectors.update({doc['id']: doc_vector})

    print('normalizing vectors...')
    # normalize vectors
    for vector in doc_vectors:
        ss = 0.0
        curr_vec = doc_vectors.get(vector)
        for i in range(len(curr_vec)):
            ss += curr_vec[i] ** 2

        ss = numpy.sqrt(ss)
        for i in range(len(curr_vec)):
            curr_vec[i] /= ss

    return dictionary


def apply_k_means(K, threshold):

    # choose initial k random vectors among all doc-vectors
    cluster_centroids = []
    for k in range(K):
        cluster_centroids.append(doc_vectors.get(random.choice(list(doc_vectors.keys()))))

    clustering_result = [[] for l in repeat(0, K)]
    j = 0
    new_j = 0
    while j == 0 or j - new_j > threshold:
        clustering_result = [[] for l in repeat(None, K)]
        for v in doc_vectors:
            norm_vec = numpy.linalg.norm(doc_vectors.get(v))
            vec = doc_vectors.get(v)
            cos_sim = []
            for centroid in cluster_centroids:
                if (numpy.linalg.norm(centroid) * norm_vec) != 0:
                    cos_sim.append(numpy.dot(centroid, vec) / (numpy.linalg.norm(centroid) * norm_vec))
            which_cluster = numpy.argmax(cos_sim)
            clustering_result[which_cluster].append(v)

        # calc the new centroids
        for l in range(K):
            n = len(clustering_result[l])
            if n == 0:
                n = 1
            res = numpy.zeros(len(dictionary))
            for vec in clustering_result[l]:
                res = numpy.add(res, doc_vectors.get(vec))

            for p in range(len(res)):
                res[p] /= float(n)

            cluster_centroids[l] = res

        j = new_j
        new_j = compute_j(clustering_result, cluster_centroids, K)

    return clustering_result, cluster_centroids


def compute_j(clustering_result, cluster_centroids, K):
    j = 0
    for k in range(K):
        for vec in clustering_result[k]:
            j += numpy.square(numpy.linalg.norm(cluster_centroids[k] - doc_vectors.get(vec)))
    return j


def compare(start, end, threshold):
    ks = list()
    js = list()
    for k in range(start, end):
        a, b = apply_k_means(k, threshold)
        j = compute_j(a, b, k)
        ks.append(k)
        js.append(j)
    return ks, js


def add_cluster_nums_to_docs(clustering_result):
    path = "json_items/"
    map_doc_to_cluster = {}

    for l in range(len(clustering_result)):
        for vec in clustering_result[l]:
            map_doc_to_cluster.update({vec: l})

    docs = {}
    for file in os.listdir(path):
        json_data = open(path + '/' + file.title(), encoding='utf8')
        doc = json.loads(json_data.read())
        doc.update({'cluster': str(map_doc_to_cluster.get(int(file.title())))})
        docs.update({file.title(): doc})
        json_data.close()

    # save new json items to files
    for d in docs:
        with open(str(d), 'w', encoding='utf-8') as f:
            json.dump(docs.get(d), f, ensure_ascii=False)


# cluster labeling
def label_clusters(clustering_results, K):
    print('naming clusters...')
    # store some most important terms of each cluster
    selected_terms = list()
    n = len(doc_vectors)

    # create a matrix
    # one row for each cluster, one column for each term, df goes in elements
    cluster_tf_df = [[0 for x in range(len(dictionary))] for x in range(len(clustering_results))]
    for cluster in range(len(clustering_results)):
        for vector in range(len(clustering_results[cluster])):
            for element in range(len(doc_vectors.get(clustering_results[cluster][vector]))):
                if doc_vectors.get(clustering_results[cluster][vector])[element] > 0:
                    cluster_tf_df[cluster][element] += 1

    for cluster in range(len(clustering_results)):
        i_ct = dict()
        for element in range(len(dictionary)):
            n11 = cluster_tf_df[cluster][element]
            n01 = len(clustering_results[cluster]) - n11

            n10 = comp_n10(cluster, element, cluster_tf_df)
            n00 = n - n10 - n11 - n01

            if n00 == 0:
                n00 = 1
            if n10 == 0:
                n10 = 1
            if n01 == 0:
                n01 = 1
            if n11 == 0:
                n11 = 1

            n1_ = n10 + n11
            n0_ = n01 + n00
            n_0 = n00 + n10
            n_1 = n11 + n01

            I = (n11 / n) * math.log10((n * n11) / (n1_ * n_1)) + (n01 / n) * math.log10((n * n01) / (n0_ * n_1)) + (n10 / n) * math.log10((n * n10) / (n1_ * n_0)) + (n00 / n) * math.log10((n * n00) / (n0_ * n_0))

            i_ct.update({dictionary[element]: I})

        selected_terms.append(sorted(i_ct, key=i_ct.get, reverse=True))

    output_terms = list()
    for sel in selected_terms:
        output_terms.append(sel[:3])

    # save output terms in file
    with open('cluster_label.txt', 'w', encoding='utf-8') as f:
        f.write(str(len(output_terms)) + '\n') # num of clusters
        for t in output_terms:
            f.write(' '.join(t) + '\n') # names for each cluster (per line)

    return output_terms


def comp_n10(cluster_num, term_num, cluster_tf_df):
    df = 0
    for cluster in range(len(cluster_tf_df)):
        if cluster != cluster_num:
            df += cluster_tf_df[cluster][term_num]
    return df


def plot(x, y, x_title, y_title, title):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_title, color='red')
    plt.ylabel(y_title, color='red')
    plt.grid(True)
    plt.show()


# main
dictionary = create_vector_space(dictionary)
print('vector space created...')
clus_res, cent = apply_k_means(K=7, threshold=0.0001)
print('k-means applied...')
for res in clus_res:
    print(res)
#print('==============dict========================')
#print(dictionary)
#print(sorted(term_df, key=term_df.get, reverse=True))

#add_cluster_nums_to_docs(clus_res)
sel_terms = label_clusters(clus_res, K=7)
print('clusters labeled...')
print(sel_terms)
#ks, js = compare(2, 15, 0.00001)
#plot(ks, js, 'K', 'Residual sum of squares', '')
