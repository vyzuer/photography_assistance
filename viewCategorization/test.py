import kmeans
import ward
import sys
import dbscan
import ap

# dbPath = "/home/scps/myDrive/Copy/Flickr-code/DBR/clustering_1004/"
# file_name = "/home/scps/myDrive/Copy/Flickr-code/DBR/clustering_1004/segments.list"

if len(sys.argv) != 4:
    print "Usage : test.py dump_path feature.list num_clusters"
    sys.exit(0)

db_path = sys.argv[1]
file_name = sys.argv[2]
n_clusters = sys.argv[3]

def cluster_k(db_path, file_name, n_clusters):
    score = kmeans.k_means(dump_path=db_path, file_name=file_name, n_clusters=int(n_clusters))    
    print '{0}:{1}'.format(n_clusters, score)

def cluster_w(db_path, file_name, n_clusters):
    score = ward.cluster(dump_path=db_path, file_name=file_name, n_clusters=int(n_clusters))    
    print '{0}:{1}'.format(n_clusters, score)

def cluster_dbscan(db_path, file_name):
    dbscan.cluster(db_path, file_name)

def cluster_ap(db_path, file_name):
    ap.cluster(db_path, file_name)

cluster_ap(db_path, file_name)
# cluster_dbscan(db_path, file_name)
# cluster_k(db_path, file_name, n_clusters)    
# cluster_w(db_path, file_name, n_clusters)    

