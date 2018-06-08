import nltk, numpy
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from itertools import product
from nltk.corpus import wordnet as wn


def vector_cluster(lista):
    collection = nltk.TextCollection(lista)
    vocabulary = list(set(collection))
    vector = [numpy.array(TF(f, vocabulary, collection)) for f in lista]
    print('Vector creado')
    return vector

def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        word_tf.append(collection.tf(word, document))
    return word_tf



def cluster_texts(vectores, clustersNumber):
    clusterer = AgglomerativeClustering(n_clusters=clustersNumber,
                                        linkage="average", affinity="cosine")
    clusters = clusterer.fit_predict(vectores)

    print("test: ", clusters)
    # Gold Standard
    reference =[0, 5, 0, 0, 0, 2, 2, 2, 3, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 0, 2, 5]
    print("reference: ", reference)


    return adjusted_rand_score(reference, clusters)


def get_similarity_score_1(word, given_list):
    max_similarity = 0

    if len(given_list) > 1:
        if word.lower() in given_list:
            max_similarity = 1
        else:
            current_verb_list = wn.synsets(word.lower())
            for verb in given_list:
                related_verbs = wn.synsets(verb.lower())
                for a, b in product(related_verbs, current_verb_list):
                    d = wn.wup_similarity(a, b)
                    try:
                        if d > max_similarity:
                            max_similarity = d
                    except:
                        continue
    else:
        if word.lower() == given_list[0].lower():
            max_similarity = 1
        else:
            current_verb_list = wn.synsets(word.lower())
            related_verbs = wn.synsets(given_list[0].lower())
            for a, b in product(related_verbs, current_verb_list):
                d = wn.wup_similarity(a, b)
                try:
                    if d > max_similarity:
                        max_similarity = d
                except:
                    continue
    return max_similarity

def comparation(sent1, sent2):
    similary = [get_similarity_score_1(word, sent2) for word in sent1]
    #print(similary)
    mean = numpy.mean(similary)
    #print(mean)
    return mean

def cluster_title(titles):
    print('Clusterizando.......')
    indices = list(range(len(titles)))
    for i in range(1,len(titles)):
        print('.  ', end=' ')
        comp = []
        ind = []
        for j in range(0, i):
            d = comparation(titles[i], titles[j])
            print('relacion title ', i, ' con title ', j, ' es de: ', d)
            if d > 0.6:
                #Guardo la comparacion y el indice del titulo
                comp.append(d)
                ind.append(j)
        if comp != []:
            #Si es no vacio es que ha entrado algun title, en caso contrario no modifico el indice
            maximo = max(comp)
            print('El maximo coincide con: ', ind[comp.index(maximo)])
            indice_relacionado = ind[comp.index(maximo)]
            indices[i] = indices[indice_relacionado]
            #el .index del maximo te da la posicion de donde esta, asi que coge el indice de que esta en esa posicion
            #ese indice es el del titulo que mas coincidencia ha tenido, asi que cambias el i por ese j

    print('Devolviendo índices:')
    return indices