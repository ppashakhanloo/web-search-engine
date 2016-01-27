import os
import json
import re
import random
import numpy
from itertools import repeat


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

dictionary = [] # term
doc_vectors = {} # (docID -> {})
term_df = {} # (term -> df)


def create_vector_space(dictionary):
    path = "json_items/"
    N = len(os.listdir(path))

    print('creating dictionary...')
    # create dictionary and df
    for file in os.listdir(path):
        json_data = open(path + '/' + file.title(), encoding='utf8').read()
        doc = json.loads(json_data)

        doc_content = ' '.join([str(doc['abstract']).lower(), str(doc['title']).lower()]).replace('\\n', ' ').replace('-',
                                                                                                                      ' ')
        doc_content_split = doc_content.split()
        for term in doc_content_split:
            temp = normalize_term(term)
            if (temp not in dictionary) and (temp != ''):
                dictionary.append(temp)

        # update df
        doc_term_set = []
        for term in doc_content_split:
            temp = normalize_term(term)
            if temp is not '':
                if temp not in doc_term_set:
                    doc_term_set.append(temp)

        for term in doc_term_set:
            if term not in term_df:
                term_df.update({term: 1})
            else:
                term_df.update({term: term_df.get(term) + 1})
    # sort the dictionary
    dictionary = list(sorted(dictionary))

    print('calc tf-idf...')
    # obtain file statistics - tf
    for file in os.listdir(path):
        json_data = open(path + '/' + file.title(), encoding='utf-8').read()
        doc = json.loads(json_data)
        doc_content = ' '.join([str(doc['abstract']).lower(), str(doc['title']).lower()]).replace('\\n', ' ').replace('-',
                                                                                                                      ' ')

        doc_vector = numpy.zeros(len(dictionary))
        doc_content_split = doc_content.split()

        for term in doc_content_split:
            temp = normalize_term(term)
            if temp is not '':
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
        for i in range(len(doc_vectors.get(vector))):
            ss += doc_vectors.get(vector)[i] ** 2

        ss = numpy.sqrt(ss)
        for i in range(len(doc_vectors.get(vector))):
            doc_vectors.get(vector)[i] /= ss


def apply_kmeans(K):
    # choose initial k random vectors among all doc-vectors
    cluster_centroids = []
    for k in range(K):
        cluster_centroids.append(doc_vectors.get(random.choice(list(doc_vectors.keys()))))

    clustering_result = [[] for l in repeat(None, K)]
    counter = 0
    for x in range(25):
        counter += 1
        clustering_result = [[] for l in repeat(None, K)]
        for v in doc_vectors:
            norm_vec = numpy.linalg.norm(doc_vectors.get(v))
            vec = doc_vectors.get(v)
            cos_sim = []
            for centroid in cluster_centroids:
                cos_sim.append(numpy.dot(centroid, vec) / (numpy.linalg.norm(centroid) * norm_vec))
            which_cluster = numpy.argmax(cos_sim)
            clustering_result[which_cluster].append(v)

        # calc the new centroids
        for l in range(K):
            n = len(clustering_result[l])
            res = numpy.zeros(len(dictionary))
            for vec in clustering_result[l]:
                res = numpy.add(res, doc_vectors.get(vec))

            for j in range(len(res)):
                res[j] /= n

            cluster_centroids[l] = res

    return clustering_result, cluster_centroids


def compute_j(clustering_result, cluster_centroids, K):
    j = 0
    for k in range(K):
        for vec in clustering_result[k]:
            inter = numpy.square(numpy.linalg.norm(cluster_centroids[k] - doc_vectors.get(vec)))
            j += inter
    return j


def compare():
    for k in range(2, 11):
        a, b = apply_kmeans(k)
        j = compute_j(a, b, k)
        print('k=' + str(k) + '=>' + 'j=' + str(j))


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


# main
create_vector_space(dictionary)
print('vector space created...')
clus_res, cent = apply_kmeans(K=7)
print('k-means applied...')
for res in clus_res:
    print(res)
add_cluster_nums_to_docs(clus_res)
