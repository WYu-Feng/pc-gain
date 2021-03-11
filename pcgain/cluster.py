from sklearn.cluster import KMeans , SpectralClustering , SpectralBiclustering , MiniBatchKMeans , AgglomerativeClustering
from sklearn import metrics
import numpy as np


def KM(data_x, k):
#KMeans cluster 
    data_class = KMeans(n_clusters=k, random_state=9).fit_predict(data_x)
    
    #translate to one-hot
    data_class_np = np.zeros(shape=(len(data_class),k))
    for i in range(len(data_class)):
        data_class_np[i,data_class[i]] = 1    
    return data_class_np , data_class

def AC(data_x , k):
#Agglomerative cluster
    data_class = AgglomerativeClustering(n_clusters = k).fit(data_x).labels_
    
    #translate to one-hot
    data_class_np = np.zeros(shape=(len(data_class),k))
    for i in range(len(data_class)):
        data_class_np[i,data_class[i]] = 1    
    return data_class_np , data_class     

def SC(data_x , k):
#Spectral cluster
    data_class = SpectralClustering(n_clusters=k).fit_predict(data_x)
    
    #translate to one-hot
    data_class_np = np.zeros(shape=(len(data_class),k))
    for i in range(len(data_class)):
        data_class_np[i,data_class[i]] = 1    
    return data_class_np , data_class     

def SB(data_x , k):
#SpectralBi cluster
    data_class = SpectralBiclustering(n_clusters=k).fit(data_x).row_labels_
    
    #translate to one-hot
    data_class_np = np.zeros(shape=(len(data_class),k))
    for i in range(len(data_class)):
        data_class_np[i,data_class[i]] = 1    
    return data_class_np , data_class   
    
def KMPP(data_x , k):
#Kmeans++ cluster
    data_class = MiniBatchKMeans(n_clusters=k).partial_fit(data_x).labels_
    
    #translate to one-hot
    data_class_np = np.zeros(shape=(len(data_class),k))
    for i in range(len(data_class)):
        data_class_np[i,data_class[i]] = 1    
    return data_class_np , data_class   
    
## Two kinds of clustering evaluation indexes
def Calinski_Harabasz(data_x , labels):
#Calinski Harabasz index
    CH = metrics.calinski_harabasz_score(data_x, labels)
    return CH
    
def Silhouette_Coefficient(data_x , labels):
#Silhouette Coefficient index
    SC = metrics.silhouette_score(data_x, labels, metric='euclidean')
    return SC