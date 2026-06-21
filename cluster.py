from debugg import debug

from typing import List
from scipy.spatial.distance import euclidean

from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt



# ===========================================



def get_average_distance(list):
    """
        Returns the average distance contained in this list.
    """

    sum_dist = 0
    size = len(list)

    for element in range(size-1):
        sum_dist += euclidean(list[element], list[element+1])

    return sum_dist/size


def get_cluster_amount(clusters):
    """
        Returns how much clusters we have
    """
    return (clusters[-1]+1)



# ===========================================



def gen_cluster(list, threshold: float = 3):
    """
        From a given list and threshold

        returns the clusterization of all elements according to threshold.
    """

    limit_dist = get_average_distance(list) * threshold
    size = len(list)
    clusters = [0] * size

    # index 0 is already 0
    for element in range(size-1):
        if (euclidean(list[element], list[element+1]) > limit_dist):
            clusters[element+1] = clusters[element]+1
        else : 
            clusters[element+1] = clusters[element]

    return clusters


def gen_cluster_centroid(list, clusters):
        """
            Returns a list of the centroids of each cluster
        """
        amount_clusters = get_cluster_amount(clusters)      # how many clusters?
        centroids = [[0]*len(list[0])] * amount_clusters    # what is the centroid of centroids[cluster_id]
        cluster_elements = [0] * amount_clusters            # how many elements in cluster_elements[cluster_id]

        # sum all elements in each cluster
        for id in range(len(list)):
            centroids[clusters[id]] = [x + y for x, y in zip(centroids[clusters[id]], list[id])]
            cluster_elements[clusters[id]] += 1

        # now divide to get the average
        for cluster in range(amount_clusters):
             centroids[cluster] = [x / cluster_elements[cluster] for x in centroids[cluster]]

        return centroids


def cluster_fitness(list, clusters, centroids):
        """
            Returns the sum of the distance between each centroid and his vertices
        """
        sum_dist = 0

        debug("\t\t Calculando distancias ao centroid")

        # sum all distance to centroid
        for id in range(len(list)):
            sum_dist += euclidean(centroids[clusters[id]], list[id])

        return sum_dist



# ===========================================



def print_clusters(dataset: List, name, cluster, fitness, Threshold):
    dataFrame = pd.DataFrame(dataset)
    n = len(dataFrame.columns)

    for i,j in combinations(range(n), 2):
        plt.scatter(
            dataFrame.iloc[:,i],
            dataFrame.iloc[:,j],
            c=cluster
        )

        plt.xlabel(dataFrame.columns[i])
        plt.ylabel(dataFrame.columns[j])

        plt.title(F"{name} - Clusters: {get_cluster_amount(cluster)} - Fitness: {fitness:.3f} - Threshold: {Threshold:.2f}")

        plt.savefig(f"./results/{name}/cluster-{dataFrame.columns[i]}_{dataFrame.columns[j]}.png")
        plt.close()