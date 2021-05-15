# Ryan Cavanagh
import os
import numpy as np
import pandas as pd 
import glob
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.manifold import TSNE

directory = os.path.join(os.path.join(os.path.join(os.path.join(os.getcwd(), '**'), '**'), '**'), '**')
dataSetList = []
for dir in glob.glob(directory):
    try:
        data = []
        x = dir.split(os.sep)
        data.append(x[-3] + " " + x[-2] + " " + x[-1])

        df = pd.read_csv(os.path.join(dir, "WL.CSV"), skiprows=1)
        x = np.transpose(df.iloc[:,2:4].to_numpy())
        data.append(x)
        start = x[0][0]
        end = x[1][-1]

        df = pd.read_csv(os.path.join(dir, "WR.CSV"), skiprows=1)
        x = np.transpose(df.iloc[:,2:4].to_numpy())
        data.append(x)
        start = min(start, x[0][0])
        end = max(end, x[1][-1])

        df = pd.read_csv(os.path.join(dir, "PS.CSV"), skiprows=1)
        x_transpose = df.iloc[:,2:14][start - 1:end].to_numpy()
        x = np.transpose(x_transpose)
        data.append(x)
        data.append(x_transpose)

        # sumGroups = []
        # x = df.iloc[:,2:5][start - 1:end]
        # sumGroups.append(x.sum(axis=1).to_numpy())
        # x = df.iloc[:,5:8][start - 1:end]
        # sumGroups.append(x.sum(axis=1).to_numpy())
        # x = df.iloc[:,8:11][start - 1:end]
        # sumGroups.append(x.sum(axis=1).to_numpy())
        # x = df.iloc[:,11:14][start - 1:end]
        # sumGroups.append(x.sum(axis=1).to_numpy())

        # data.append(sumGroups)

        data.append(start)
        data.append(end)

        dataSetList.append(data)
        # break
    except:
        print(dir)


def percentage(true, k):
    count = 0
    total = 0
    for i in (true == k):
        if i:
            count += 1
        total += 1
    print(count/total)


saveLocation = 'FiguresTSNE_kmeans'
tsne = TSNE(n_components=2)

if not os.path.exists(saveLocation):
    os.makedirs(saveLocation)

def show_tsne(data, k_label, true_label, title, trial):
    #Grayson
    dataPoints = tsne.fit_transform(data)
    #plot training figures
    plt.clf()
    plt.scatter(dataPoints[k_label == 0][:,0],dataPoints[k_label == 0][:,1],label="Centroid 0",s=13)
    plt.scatter(dataPoints[k_label == 1][:,0],dataPoints[k_label == 1][:,1],label="Centroid 1",s=13)
    plt.legend(loc='best', title='Legend', frameon=True, fontsize='medium')
    plt.title(trial[0] + " " + title + " K-Means Label")
    try:
        plt.savefig(os.path.join(saveLocation, trial[0] + " " + title + " K-Means Label" + ".png"))
    except:
        pass
    # plt.show()

    plt.clf()
    plt.scatter(dataPoints[true_label == 0][:,0],dataPoints[true_label == 0][:,1],label="Swing",s=13)
    plt.scatter(dataPoints[true_label == 1][:,0],dataPoints[true_label == 1][:,1],label="Stance",s=13)
    plt.legend(loc='best', title='Legend', frameon=True, fontsize='medium')
    plt.title(trial[0] + " " + title + " true label")
    try:
        plt.savefig(os.path.join(saveLocation, trial[0] + " " + title + " true label" + ".png"))
    except:
        pass
    # plt.show()


def true_labels_12D(k, trial):
    if k == 2:
        three_clusters = False
    else:
        three_clusters = True

    l_swing = False
    r_swing = False
    l_end = False
    r_end = False
    left = np.transpose(trial[1]).flatten().tolist()
    right = np.transpose(trial[2]).flatten().tolist()
    true_labels = []
    l_next = left.pop(0)
    r_next = right.pop(0)

    # 00 l_swing = 0, r_swing = 0
    # 01 l_swing = 0, r_swing = 1
    # 10 l_swing = 1, r_swing = 0
    # 11 l_swing = 1, r_swing = 1

    for i in range(trial[5], trial[6] + 1):
            if not l_end:
                if l_swing:
                    if i > l_next:
                        l_swing = False
                        if len(left) == 0:
                            l_end = True
                        else:
                            l_next = left.pop(0)
                else:
                    if i >= l_next:
                        l_swing = True
                        if len(left) == 0:
                            l_end = True
                        else:
                            l_next = left.pop(0)

            if not r_end:
                if r_swing:
                    if i > r_next:
                        r_swing = False
                        if len(right) == 0:
                            r_end = True
                        else:
                            r_next = right.pop(0)
                else:
                    if i >= r_next:
                        r_swing = True
                        if len(right) == 0:
                            r_end = True
                        else:
                            r_next = right.pop(0)
                        

            if l_swing and r_swing:
                # 11 = 3
                if three_clusters:
                    true_labels.append(3)
                else:
                    # true_labels.append(1)
                    true_labels.append(0)
            elif l_swing:
                # 10 = 2
                if three_clusters:
                    true_labels.append(2)
                else:
                    # true_labels.append(1)
                    true_labels.append(0)
            elif r_swing:
                # 01 = 1
                # true_labels.append(1)
                true_labels.append(0)
            else:
                # 00 = 0
                # true_labels.append(0)
                true_labels.append(1)
    return np.array(true_labels)

def kmeans_12D(k, time, trial):
    if time:
        array = []
        for i, row in enumerate(trial[4]):
            row = row.tolist()
            row.append(i + trial[5])
            array.append(np.array(row))
        X = np.array(array)
    else:
        X = np.array(trial[4])

    kmeans = KMeans(n_clusters = k, random_state=0).fit(X)
    kmeans_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # print(X)
    # print(kmeans_labels)
    # print(true_labels_12D(k))
    # print()

    return X, kmeans_labels, true_labels_12D(k, trial)







def kmeans_12D_prediction(k, trial):
    X = np.array(trial[4])

    true_labels = true_labels_12D(k, trial)

    shuffle(X, true_labels)
    x_train, x_test, train_true_labels, test_true_labels = train_test_split(X, true_labels, test_size=0.25, random_state=1, shuffle=False)

    kmeans = KMeans(n_clusters = k, random_state=0).fit(x_train)
    kmeans_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    predict_labels = kmeans.predict(x_test)

    # print(x_train)
    # print(kmeans_labels)
    # print(train_true_labels)
    # print()
    
    # print(x_test)
    # print(predict_labels)
    # print(test_true_labels)
    # print()

    return x_train, kmeans_labels, train_true_labels, x_test, predict_labels, test_true_labels






def true_labels_6D(left, trial):
    swing = False
    end = False
    if left:
        array = np.transpose(trial[1]).flatten().tolist()
    else:
        array = np.transpose(trial[2]).flatten().tolist()
    true_labels = []
    next = array.pop(0)

    # swing = 1, stance = 0

    for i in range(trial[5], trial[6] + 1):
            if not end:
                if swing:
                    if i > next:
                        swing = False
                        if len(array) == 0:
                            end = True
                        else:
                            next = array.pop(0)
                else:
                    if i >= next:
                        swing = True
                        if len(array) == 0:
                            end = True
                        else:
                            next = array.pop(0)

            if swing:
                # true_labels.append(1)
                true_labels.append(0)
            else:
                # true_labels.append(0)
                true_labels.append(1)

    return np.array(true_labels)


def kmeans_6D(trial):
    left_X, right_X = np.hsplit(trial[4], 2)

    left_kmeans = KMeans(n_clusters = 2, random_state=0).fit(left_X)
    left_kmeans_labels = left_kmeans.labels_
    left_centroids = left_kmeans.cluster_centers_

    right_kmeans = KMeans(n_clusters = 2, random_state=0).fit(right_X)
    right_kmeans_labels = right_kmeans.labels_
    right_centroids = right_kmeans.cluster_centers_

    # print(left_X)
    # print(left_kmeans_labels)
    # print(true_labels_6D(True))
    # print()

    # print(right_X)
    # print(right_kmeans_labels)
    # print(true_labels_6D(False))
    # print()

    return left_X, left_kmeans_labels, true_labels_6D(True, trial), right_X, right_kmeans_labels, true_labels_6D(False, trial)





def kmeans_6D_prediction(trial):
    left_X, right_X = np.hsplit(trial[4], 2)

    left_true_label = true_labels_6D(True, trial)
    right_true_label = true_labels_6D(False, trial)

    shuffle(left_X, left_true_label)
    l_x_train, l_x_test, l_train_true_labels, l_test_true_labels = train_test_split(left_X, left_true_label, test_size=0.25, random_state=1, shuffle=False)

    l_kmeans = KMeans(n_clusters = 2, random_state=0).fit(l_x_train)
    l_kmeans_labels = l_kmeans.labels_
    l_centroids = l_kmeans.cluster_centers_

    l_predict_labels = l_kmeans.predict(l_x_test)

    shuffle(right_X, right_true_label)
    r_x_train, r_x_test, r_train_true_labels, r_test_true_labels = train_test_split(right_X, right_true_label, test_size=0.25, random_state=1, shuffle=False)

    r_kmeans = KMeans(n_clusters = 2, random_state=0).fit(r_x_train)
    r_kmeans_labels = r_kmeans.labels_
    r_centroids = r_kmeans.cluster_centers_

    r_predict_labels = r_kmeans.predict(r_x_test)

    # print(x_train)
    # print(kmeans_labels)
    # print(train_true_labels)
    # print()
    
    # print(x_test)
    # print(predict_labels)
    # print(test_true_labels)
    # print()

    return l_x_train, l_kmeans_labels, l_train_true_labels, l_x_test, l_predict_labels, l_test_true_labels, r_x_train, r_kmeans_labels, r_train_true_labels, r_x_test, r_predict_labels, r_test_true_labels


for trial in dataSetList:
    data, k_label, true_label = kmeans_12D(2, False, trial)
    # print("K-means 12D Unsplit")
    # percentage(true_label, k_label)
    show_tsne(data, k_label, true_label, "K-means 12D Unsplit", trial)

    train_data, train_k_label, train_true_label, test_data, test_k_label, test_true_label = kmeans_12D_prediction(2, trial)
    # print("K-means 12D Split 75% Train")
    # percentage(train_true_label, train_k_label)
    show_tsne(train_data, train_k_label, train_true_label, "K-means 12D Split 75% Train", trial)
    # print("K-means 12D Split 25% Test")
    # percentage(test_true_label, test_k_label)
    show_tsne(test_data, test_k_label, test_true_label, "K-means 12D Split 25% Test", trial)

    l_data, l_k_label, l_true_label, r_data, r_k_label, r_true_label = kmeans_6D(trial)
    # print("K-means Left 6D Unsplit")
    # percentage(l_true_label, l_k_label)
    show_tsne(l_data, l_k_label, l_true_label, "K-means Left 6D Unsplit", trial)
    # print("K-means Right 6D Unsplit")
    # percentage(r_true_label, r_k_label)
    show_tsne(r_data, r_k_label, r_true_label, "K-means Right 6D Unsplit", trial)

    l_x_train, l_kmeans_labels, l_train_true_labels, l_x_test, l_predict_labels, l_test_true_labels, r_x_train, r_kmeans_labels, r_train_true_labels, r_x_test, r_predict_labels, r_test_true_labels = kmeans_6D_prediction(trial)
    # print("K-means Left 6D Split 75% Train")
    # percentage(l_train_true_labels, l_kmeans_labels)
    show_tsne(l_x_train, l_kmeans_labels, l_train_true_labels, "K-means Left 6D Split 75% Train", trial)
    # print("K-means Left 6D Split 25% Test")
    # percentage(l_test_true_labels, l_predict_labels)
    show_tsne(l_x_test, l_predict_labels, l_test_true_labels, "K-means Left 6D Split 25% Test", trial)
    # print("K-means Right 6D Split 75% Train")
    # percentage(r_train_true_labels, r_kmeans_labels)
    show_tsne(r_x_train, r_kmeans_labels, r_train_true_labels, "K-means Right 6D Split 75% Train", trial)
    # print("K-means Right 6D Split 25% Test")
    # percentage(r_test_true_labels, r_predict_labels)
    show_tsne(r_x_test, r_predict_labels, r_test_true_labels, "K-means Right 6D Split 25% Test", trial)