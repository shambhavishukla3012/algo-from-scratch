# Python version - 3.8.9
# Assumption- similar items are clustered together
# and cluster are in nested list format that is stored in pickle file

import pickle
import numpy as np

# Function to calculate distance
def distance(a ,b):
    x = np.array(a)
    y = np.array(b)
    return np.linalg.norm(a - b)

# Functiont to open a pickle file
def open_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f) 
    return data

# Function to get the nearest neighbors
def knn(input,embed_file,cluster_file,k):  
    # open the files
    cluster = open_file(cluster_file) # ouput-> nested list where each list represents the clusters
    embed = open_file(embed_file) # output -> dictionary wherre key is the node and its values is the embedding

    # # Get new point key and values
    # new_pointKey = [*input.keys()][0]
    # new_point = [*input.values()][0]
    
    # Input point and its embedding
    new_pointKey = input
    new_point = embed[input]

    # check which cluster item belong -> only retain all the values of that cluster
    # check if point in which cluster
    if (new_pointKey in x for x in cluster): # if point in the cluster nested lists
        for i in range(len(cluster)): # fetch the index of the cluster item belongs
            if new_pointKey in cluster[i]:
                cluster_index = i
        
        # only retain embeddings of the cluster input item belongs
        embed = {key: embed[key] for key in cluster[cluster_index]}
        # remove the input from the embedding dictionary
        del embed[input]

        keys = [*embed.keys()] # all keys
        data = [*embed.values()] # all key values
        # calculate distance of the point to every other item(cluster item)
        distance = np.linalg.norm(np.array(data) - np.array(new_point), axis=1)
        # sort indexes of the cluster according to the distance
        nearest_neighbor_ids = distance.argsort()[:k]
        # fetch ingredient names of nearest neighbor
        result = ([keys[i] for i in nearest_neighbor_ids])
        print(result)
    else:
        print("Item not in the dataset")


## Usage
input = 'Cornstarch'# Input is the name of the ingredient
embed_file = 'embed.pickle' # embedding file -> dictionary wherre key is the node and its values is the embedding
cluster_file = 'cluster.pickle' # cluster file -> nested list where each list represents the clusters
k = 15 # number of nearest neighbors to fetch

knn(input,embed_file,cluster_file,k)
