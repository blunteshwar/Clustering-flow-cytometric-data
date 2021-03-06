import fcsparser
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import statistics 
import timeit
from sklearn.cluster import KMeans
from scipy import stats
import scipy

# data is a np matrix with each row as a cell and the "Time" column being removed 
# channels is a list of attributes with "Time" removed

def ParseData():
    path='./ayan.fcs'
    meta, data = fcsparser.parse(path, meta_data_only=False, reformat_meta=True)
    data=data.to_numpy()
    channels=list(meta['_channel_names_'])
    channels.remove("Time")
    data=np.delete(data,-1,1)
    return(data,channels)

def get_csv(data):
    with open('ayan.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(channels)
        writer.writerows(data)


def get_csv_with_clusters(path,channels):
    with open(path+"cluster_labels.csv", newline='') as f:
        reader = csv.reader(f)
        labels = list(reader)
        labelf=[0,0,0,0,0,0,0]
        print(type(labels[0]))
        print(labels[0])
        for i in range(len(labels)):
            labels[i]=int(float(labels[i][0]))
            labelf[labels[i]]=labelf[labels[i]]+1
        print(labelf)

        x=np.column_stack((data,labels))
    channels.append("cluster_label")
    with open(path+"with_clusters.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(channels)
        writer.writerows(x)

def get_cluster_indexed_list(number_of_clusters,path): #cil = clustered indexed list
    cil=[]
    
    for i in range(number_of_clusters):
        empty=[]
        cil.append(empty)
    with open(path+"with_clusters.csv",'rt')as f:
        data = csv.reader(f)
        flag=0
        for row in data:
            if flag==0:
                flag=1
                continue
            ind=int(float(row[-1]))
            cil[ind].append(row)
    return(cil)

def generate_cluster_data(number_of_clusters,path,channels):  #generating the cluster data and writing in a csv
    path=path+"cluster_data/"
    spread = {
    "mean": -1,
    "mode": -1,
    "median": -1,
    "std_dev": -1,
    "min": -1,
    "max": -1
    }
    
    head=["attribute","mean","mode","median","std_dev","min","max"]
    for i in range(number_of_clusters):
        with open(path+"cluster"+str(i)+".csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(head)
            npa=np.array(cil[i])
            for j in range(len(channels)):
                
                colum=[float(qq) for qq in npa[:,j]]
                spread["mean"]=np.mean(colum)
                spread['mode']=str(stats.mode(colum).mode)
                spread["median"]=np.median(colum)
                spread["std_dev"]=statistics.stdev(colum)
                spread["min"]=min(colum)
                spread["max"]=max(colum)
                lk=[channels[j],spread["mean"],spread['mode'],spread["median"],spread["std_dev"],spread["min"],spread["max"]]
                writer.writerow(lk)
        



def plot_data(cil,channels,path):
    path=path+"log10_plots/"
    color_list=['b','g','tab:orange','violet','r','lime','darkgoldenrod','sienna','black','lightseagreen','tab:grey','crimson','darkslategrey']
    n=len(cil[0][0])-1
    number_of_clusters=len(cil)
    for k in range(n): # the x-axis
        j=k+1           # the y axis
        while j<n:
            for i in range(number_of_clusters):
                az=np.array(cil[i])
                x_axis=[float(t)+1 for t in az[:,k]]
                y_axis=[float(t)+1 for t in az[:,j]]
                plt.scatter(np.log10(x_axis),np.log10(y_axis),s=0.00005,c=color_list[i])
            plt.xlabel(channels[k])
            plt.ylabel(channels[j])
            plot_name=str(channels[k])+" VS "+str(channels[j])
            plt.title(plot_name)
            local_path=path+plot_name+'.png'
            plt.savefig(local_path)
            plt.clf()
            #plt.show()
            j=j+1

#computing parameters for canopy clustering
def compute_parameters(data,path):
    print(type(data))  
    k=7
    sum1=0
    sum2=0
    dis1=0
    dis2=0
    for lk in range(10):
        for i in range(6):
            k=7+i
            kmeans = KMeans(n_clusters=k,init='k-means++', random_state=0).fit(data)
            cluster_centres=kmeans.cluster_centers_

            mini=float('inf') 
            for j in range(len(cluster_centres)):
                g=j+1
                while(g<len(cluster_centres)):
                    d=scipy.spatial.distance.euclidean(cluster_centres[j], cluster_centres[g])
                    if(d<mini):
                        mini=d
                    g=g+1
            sum1=sum1+mini
            dis1=dis1+1
            print(k)

            y=kmeans.labels_
            X_dist = kmeans.transform(data)**2
            df = pd.DataFrame(X_dist.sum(axis=1).round(2), columns=['sqdist'])
            df['label'] = y
            max_indices = []
            for label in np.unique(kmeans.labels_):
                X_label_indices = np.where(y==label)[0]
                max_label_idx = X_label_indices[np.argmax(X_dist[y==label].sum(axis=1))]
                max_indices.append(max_label_idx)
            sum2=sum2+min(max_indices)
            dis2=dis2+1
            

    t1=sum1/dis1
    sum2=sum2/2
    t2=sum2/dis2
    return(t1,t2)
   
#performing canopy clustering
def create_canopies(t1,t2,data):
    representatives=[]

    labels=[(-1,0) for i in range(len(data))]
    for u in range(10):
        print("iteration= ",u)
        for i in range(len(data)):
            x=data[i]
            if(labels[i][1]==1):
                continue
            if not representatives:
                labels[i]=(0,1)
                #1-> fixed 0->processing
                representatives.append((x,i))
                
            else:
                mini=float('inf')
                rep=-1
                for y in representatives:
                    d=scipy.spatial.distance.euclidean(x,y[0])
                    if(d<mini):
                        mini=d
                        rep=y
                if(mini<t2):
                    label_number=labels[rep[1]][0]
                    labels[i]=(label_number,1)
                elif(mini<t1):
                    label_number=labels[rep[1]][0]
                    labels[i]=(label_number,0)
                else:
                    label_number=len(representatives)
                    print(label_number,i)
                    labels[i]=(label_number,1)
                    representatives.append((x,i))
        
    return(labels,representatives)



#the clustering algo
def make_clusters(data,path):
    #make clusters and create a file cluster_labels.csv and returns number of clusters
    # clusters are from 0 to n-1 where n=number of clusters



    kmeans = KMeans(n_clusters=10000,init='k-means++',precompute_distances=True,n_jobs=-1, random_state=0).fit(data)
    labels=np.array(kmeans.labels_)
    superpixeled_data=kmeans.cluster_centers_
    
    '''with open(path+"cluster_labels.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(labels)'''
    np.savetxt(path+"cluster_labels.csv", labels, delimiter=",")
    return(7)

def precomp(path):
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    if not os.path.exists("./log10_plots/"):
        os.mkdir("./log10_plots/")
        os.mkdir("./cluster_data/")
    os.chdir(os.path.dirname(os.getcwd()))




algo_name="Final_Implementation"
path="./"+algo_name+"/"
precomp(path)

data,channels=ParseData()
print(len(data))


t1=185130.14921875
t2=58035.166666666664
if t1==0:
    t1,t2=compute_parameters(data,path)
start = timeit.default_timer()
labels,representatives = create_canopies(t1,t2,data)
stop = timeit.default_timer()

print('Time: ', stop - start)  
'''number_of_clusters=make_clusters(data,path)
get_csv_with_clusters(path,channels)
number_of_clusters=7
cil=get_cluster_indexed_list(number_of_clusters,path)
plot_data(cil,channels,path)
generate_cluster_data(number_of_clusters,path,channels)'''

