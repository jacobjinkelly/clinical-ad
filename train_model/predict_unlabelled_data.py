from localglobalembed import AbbrRep
import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.manifold import TSNE

def predict_gaussian(gaussians, x_test):
    gl = np.zeros((len(x_test), max(gaussians) + 1))
    for u in range(len(x_test)):
        for v in range(max(gaussians) + 1):
            gl[u][v] = np.log(0)
    predicted_labels = []

    d = len(x_test[0])
    for u in range(len(x_test)):
        for v in gaussians:
            norm_factor = ((2 * np.pi) ** (-d / 2)) * np.linalg.det(gaussians[v]["cov"]) ** (-1 / 2)
            md = np.exp(np.dot(np.dot((x_test[u][0] - gaussians[v]["mean"]).T,
                                      np.linalg.inv(gaussians[v]["cov"])), (x_test[u][0] - gaussians[v]["mean"]))*(-1/2))

            gl[u][v] = np.log(norm_factor * md)


    for i in range(len(x_test)):
        predicted_labels.append(np.argmax(gl[i]))
    return predicted_labels

def compute_mean_mles(train_data):
    mean = np.sum(train_data, axis=0)/len(train_data)
    return mean

def compute_sigma_mles(train_data):
    num_samples = len(train_data)
    d = len(train_data[0])
    means = compute_mean_mles(train_data)
    I = np.identity(d) * 0.1
    covariance = np.add(np.dot(np.array(train_data - means).T,
                               np.array(train_data - means)*(1/num_samples)), I)
    return means, covariance

def gda(X, labels, clusters):
    g_id = []
    g_id_to_label = {}
    seen = {}
    count = {}
    idx = 0
    for cluster in range(len(clusters)):
        try:
            g_id.append(seen[(clusters[cluster], labels[cluster])])
            count[seen[(clusters[cluster], labels[cluster])]] += 1
        except KeyError:
            seen[(clusters[cluster], labels[cluster])] = idx
            count[seen[(clusters[cluster], labels[cluster])]] = 1
            g_id.append(seen[(clusters[cluster], labels[cluster])])
            g_id_to_label[idx] = labels[cluster]
            idx += 1

    X_train = {}


    for i in range(len(X)):
        if count[g_id[i]] > 1:
            try:
                X_train[g_id[i]].append(X[i])
            except KeyError:
                X_train[g_id[i]] = [X[i]]


    gaussians = {}
    for key in X_train:
        gaussians[key] = {}
        # mean = np.mean(X_train[key], axis=0)
        mean, cov = compute_sigma_mles(X_train[key])
        # cov = np.cov(X_train[key])

        gaussians[key]["mean"] = mean
        gaussians[key]["cov"] = cov

    return gaussians, g_id_to_label

def cluster_rs(data, unlabelled_x, sim=False):
    X = []
    labels = []
    key = "mimic_rs"
    key2 = "mimic_rs_sim"
    for subkey in data[key]:
        for i in data[key][subkey]:
            X.append(i.embedding[0])
            labels.append(i.label)
    if(sim == True):
        for subkey in data[key2]:
            for i in data[key2][subkey]:
                X.append(i.embedding[0])
                labels.append(i.label)

    split = len(X)
    for i in unlabelled_x:
        X.append(i)

    X = np.array(X).reshape((-1,len(X[0])))

    reduced_dim = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000, random_state=42)
    results = reduced_dim.fit_transform(X)
    vis_x = results[:, 0]
    vis_y = results[:, 1]
    X = []
    unlabelled_x_red = []
    for i in range (len(vis_x)):
        if i < split:
            X.append([vis_x[i], vis_y[i]])
        else:
            unlabelled_x_red.append([vis_x[i], vis_y[i]])


    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    gaussians, g_id_to_label= gda(X, labels, clustering.labels_)
    return gaussians, g_id_to_label, unlabelled_x_red

def predict_labels(data, unlabelled_x, sim=False):
    gaussians, g_id_to_label, unlabelled_x_red = cluster_rs(data, unlabelled_x, sim=True)

    predictions = predict_gaussian(gaussians, unlabelled_x_red)
    prediction_labels = []
    for i in predictions:
        prediction_labels.append(g_id_to_label[i])

    return prediction_labels

def knn(data, unlabelled_x, sim=False):
    X = []
    labels = []
    key = "mimic_rs"
    key2 = "mimic_rs_sim"
    for subkey in data[key]:
        for i in data[key][subkey]:
            X.append(i.embedding[0])
            labels.append(i.label)
    if(sim == True):
        for subkey in data[key2]:
            for i in data[key2][subkey]:
                X.append(i.embedding[0])
                labels.append(i.label)
    pred = []
    for i in unlabelled_x:
        dist = np.sum((np.array(X) - np.array(i))**2, axis=1)
        ind = np.argpartition(dist, -10)[-10:]
        poss_labels = []
        for i in ind:
            poss_labels.append(labels[i])
        x = Counter(poss_labels)
        maj_vote = x.most_common()[0][0]
        pred.append(maj_vote)

    return pred