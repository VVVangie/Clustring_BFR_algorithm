#/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --executor-memory 4G --driver-memory 4G task.py

import pyspark
import sys
import time
import math
import random
import copy
import numpy as np
from sklearn.cluster import KMeans



input_file = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]
#input_file = 'hw6_clustering.txt'
#n_cluster = 10
#output_file = 'new_ans.py'


time_start = time.time()

def intermediate_output(DS_lst, CS_lst, RS_lst):
    num_ds_points = 0
    for i in range(len(DS_lst)):
        num_ds_points += len(DS_lst[i])
        
    num_cs_clusters = len(CS_lst)
    
    num_cs_points = 0
    for i in range(len(CS_lst)):
        num_cs_points += len(CS_lst[i])
                         
    num_rs = len(RS_lst)
    
    return [num_ds_points,num_cs_clusters,num_cs_points,num_rs]

def get_stats(cluster_lst):
    stats_lst = []
    for one_cluster in cluster_lst:
        clu_stats = []
        N = len(one_cluster)
        SUM = []
        SUMSQ = []
        std_dev = []
        for i in range(dimension):
            i_sum = 0
            i_sumsq =0
            for one_point in one_cluster:
                i_sum += one_point[i]
                i_sumsq += (one_point[i])**2
            SUM.append(i_sum)
            SUMSQ.append(i_sumsq)
            i_var = (i_sumsq / N) - (i_sum / N)**2
            i_std = math.sqrt(i_var)
            std_dev.append(i_std)
        stats_lst.append([N,SUM,SUMSQ,std_dev])
    return stats_lst

def mahalanobis_distance(point,cluster_stats):
    #get the cluster centroid:
    centroid = []
    mah_dis2_sum = 0
    for i in range(dimension):
        i_cen = cluster_stats[1][i] / cluster_stats[0]
        centroid.append(i_cen)
        #get the mah distance:
        i_std = cluster_stats[3][i]
        if i_std != 0:
            i_dis2 = ((point[i]- i_cen) / i_std)**2
        else:
            i_dis2 = ((point[i]- i_cen)/ 0.00001)**2
        mah_dis2_sum += i_dis2
    mah_dis = math.sqrt(mah_dis2_sum)
    return mah_dis

def update_stats(point,chosen_index,cluster_stats):
    the_stats = cluster_stats[chosen_index]
    the_stats[0] += 1
    std_dev = []
    for i in range(dimension):
        the_stats[1][i] += point[i]
        the_stats[2][i] += (point[i])**2
        i_var = (the_stats[2][i] / the_stats[0]) - (the_stats[1][i] / the_stats[0])**2
        i_std = math.sqrt(i_var)
        std_dev.append(i_std)
    the_stats[3] = std_dev
    return the_stats

def updatd_stats_when_ds_cs(cs_clu,ds_index,ds_cluster_stats):
    stats = ds_cluster_stats[ds_index]
    stats[0] += len(cs_clu)
    std_deviation = []
    for i in range(dimension):
        for c_point in cs_clu:
            stats[1][i] += c_point[i]
            stats[2][i] += (c_point[i])**2
        i_variance = (stats[2][i] / stats[0]) - (stats[1][i] / stats[0])**2
        i_std_dev = math.sqrt(i_variance)
        std_deviation.append(i_std_dev)
    stats[3] = std_deviation
    return stats

def get_centroid(cluster_stats,index):
    centroid = []
    mah_dis2_sum = 0
    for i in range(dimension):
        i_cen = cluster_stats[index][1][i] / cluster_stats[index][0]
        centroid.append(i_cen)
    return centroid



#Main:
points_file = open(input_file, "r")
all_points_map = dict() #key:point, value:label
#all_points_map = []
left_points = []
for line in points_file.readlines():
    point = []
    current_line = line.strip().split(',')
    for each in current_line[2:]:
        point.append(float(each))
    all_points_map[tuple(point)] = current_line[0]
    left_points.append(point)

points_sum_length = len(left_points)
choose_num = math.ceil(float(points_sum_length)*0.2)
dimension = len(left_points[0])

#random.seed(553)
chosen_points = left_points[:choose_num]
left_points = left_points[choose_num:]

#Step 2. 
index_del = []
first_RS = []

import_points = np.array(chosen_points)
kmeans = KMeans(n_clusters = 5 * n_cluster).fit(import_points)
label_result = kmeans.labels_.tolist()

#Step 3.
for label in set(label_result):
    point_num = label_result.count(label)

    if point_num == 1:
        point_index = label_result.index(label)
        the_point = chosen_points[point_index]
        first_RS.append(the_point)
        index_del.append(point_index)
clustered_points = [clu for i,clu in enumerate(chosen_points) if i not in index_del]

#Step 4.
updated_import_points = np.array(clustered_points)
kmeans_2 = KMeans(n_clusters = n_cluster).fit(updated_import_points)
clusters_1 = kmeans_2.labels_.tolist()
centroids_1 = kmeans_2.cluster_centers_.tolist()

#Step 5.
DS_cluster_1 = []
for cluster_label in set(clusters_1):
    cluster_points = []
    for i in range(len(clusters_1)):
        if clusters_1[i] == cluster_label:
            cluster_points.append(clustered_points[i])
    #cluster_centroid = tuple(centroids_1[cluster_label])
    DS_cluster_1.append(cluster_points)

#Step 6.
CS_map_1 = []
if len(first_RS) > 5 * n_cluster:
    RS_import_points = np.array(first_RS)
    kmeans_3 = KMeans(n_clusters = 5 * n_cluster).fit(RS_import_points)
    RS_kmeans_result = kmeans_3.labels_.tolist()
    #RS_centrodis = kmeans_3.cluster_centers_.tolist()
    rs_del_i = []
    for labe in set(RS_kmeans_result):
        labe_num = RS_kmeans_result.count(labe)
        if labe_num >1:
            cs_cluster = []
            for a in range(len(RS_kmeans_result)):
                if RS_kmeans_result[a] == labe:
                    cs_cluster.append(first_RS[a])
                    rs_del_i.append(a)
            CS_map_1.append(cs_cluster)
    first_RS = [clu for i,clu in enumerate(first_RS) if i not in rs_del_i]

#intermediate output for first round:
intermediate_output_lst = []
inte_output_1 = intermediate_output(DS_cluster_1, CS_map_1, first_RS)
intermediate_output_lst.append(inte_output_1)

#Repeat Steps 7 – 12.
now_DS = DS_cluster_1
now_CS = CS_map_1
now_RS = first_RS
current_DS_stats = get_stats(now_DS)
current_CS_stats = get_stats(now_CS)
round_num = 2

while round_num <= 5:
    #Step 7.
    if round_num < 5:
        updated_chosen_point = left_points[:choose_num]
        left_points = left_points[choose_num:]
    else:
        updated_chosen_point = left_points
    

    for the_point in updated_chosen_point:
        #Step 8.
        #get nearest distance to DS:
        ds_nearest_dis = float('+inf')
        for i in range(len(now_DS)):
            the_clu_stats = current_DS_stats[i]
            the_distance = mahalanobis_distance(the_point,the_clu_stats)
            if the_distance < ds_nearest_dis:
                ds_nearest_dis = the_distance
                nearest_clu_index = i
        #Step 9.
        #get nearest distance to CS:
        cs_nearest_dis = float('+inf')
        for j in range(len(now_CS)):
            chosen_clu_stats = current_CS_stats[j]
            chosen_distance = mahalanobis_distance(the_point,chosen_clu_stats)
            if chosen_distance < cs_nearest_dis:
                cs_nearest_dis = chosen_distance
                closest_clu_index = j
            
        if ds_nearest_dis < 2*(math.sqrt(dimension)):
            now_DS[nearest_clu_index].append(the_point)
            current_DS_stats[nearest_clu_index] = update_stats(the_point,nearest_clu_index,current_DS_stats)
        elif cs_nearest_dis < 2*(math.sqrt(dimension)):
            now_CS[closest_clu_index].append(the_point)
            current_CS_stats[closest_clu_index] = update_stats(the_point,closest_clu_index,current_CS_stats)
        else:
            now_RS.append(the_point) #Step 10.
            
    #Step 11.
    if len(now_RS) > 5 * n_cluster:
        RS_import_ones = np.array(now_RS)
        kmeans_run = KMeans(n_clusters = 5 * n_cluster).fit(RS_import_ones)
        RS_kmeans_outcome = kmeans_run.labels_.tolist()
        #RS_kmeans_centroids = kmeans_run.cluster_centers_.tolist()
        del_index_collect = []
        for labels in set(RS_kmeans_outcome):
            new_cs_cluster = []
            labe_sum = RS_kmeans_outcome.count(labels)
            if labe_sum >1:
                for ind in range(len(RS_kmeans_outcome)):
                    if RS_kmeans_outcome[ind] == labels:
                        new_cs_cluster.append(now_RS[ind])
                        del_index_collect.append(now_RS[ind])
                now_CS.append(new_cs_cluster)
        for g in del_index_collect:
            now_RS.remove(g)
        current_CS_stats = get_stats(now_CS)
        
    #Step 12.
    combine_pairs = []
    already_combined = []
    for i in range(len(now_CS)):
        cs_1 = now_CS[i]
        cs_1_cen = get_centroid(current_CS_stats,i)
        if i not in already_combined:
            for j in range(i+1, len(now_CS)):
                if j not in already_combined:
                    the_cs_stats = current_CS_stats[j]
                    cs_distance = mahalanobis_distance(cs_1_cen,the_cs_stats)
                    if cs_distance < 2*(math.sqrt(dimension)):
                        combine_pairs.append((i,j))
                        already_combined.append(i)
                        already_combined.append(j)
    updated_CS =[clu for i,clu in enumerate(now_CS) if i not in already_combined]
    for pair in combine_pairs:
        merge_points_lst = now_CS[pair[0]] + now_CS[pair[1]]
        updated_CS.append(merge_points_lst)
    now_CS = updated_CS
    current_CS_stats = get_stats(now_CS)
    
    if round_num == 5:
        del_cs_index = []
        for i in range(len(now_CS)):
            cs_cen = get_centroid(current_CS_stats,i)
            smallest_dis = float('+inf')
            for j in range(len(now_DS)):
                the_chosen_ds = current_DS_stats[j]
                cs_ds_distance = mahalanobis_distance(cs_cen,the_chosen_ds)
                if cs_ds_distance < smallest_dis:
                    smallest_dis = cs_ds_distance
                    smallest_clu_i = j
                    if cs_ds_distance < 2*(math.sqrt(dimension)):
                        now_DS[smallest_clu_i] += now_CS[i]
                        del_cs_index.append(i)
                        #current_DS_stats[smallest_clu_i] = updatd_stats_when_ds_cs(now_CS[i],smallest_clu_i,current_DS_stats)
                        #now_CS.pop(i)
                        #current_CS_stats.pop(i)
        now_CS =[clu for i,clu in enumerate(now_CS) if i not in del_cs_index]
        
    inte_output_new = intermediate_output(now_DS, now_CS, now_RS)
    intermediate_output_lst.append(inte_output_new)
                        
    round_num += 1
    
cluster_results = []
for i in range(len(now_DS)):
    for point in now_DS[i]:
        point_label = all_points_map[tuple(point)]
        #point_label = all_points_map.index(point)
        cluster_results.append([point_label,i])
for j in range(len(now_CS)):
    for po in now_CS[j]:
        point_name = all_points_map[tuple(po)]
        #point_name = all_points_map.index(po)
        cluster_results.append([point_name,-1])
for poin in now_RS:
    point_n = all_points_map[tuple(poin)]
    cluster_results.append([point_n,-1])
cluster_results = sorted(cluster_results,key=lambda x: x[0],reverse=False)

    
with open(output_file, 'w')as f:
    f.write("The intermediate results:\n")
    for i in range(len(intermediate_output_lst)):
        round_name = i+1
        f.write("Round ")
        f.write(str(round_name).replace("'",""))
        f.write(": ")
        f.write((str(intermediate_output_lst[i])[1:-1]).replace("'","").replace(" ",""))
        f.write('\n')
    f.write('\n')
    f.write("The clustering results:\n")
    for j in cluster_results:
        f.write((str(j)[1:-1]).replace("'","").replace(" ",""))
        f.write('\n')
                


time_end = time.time()
print("Duration:{0:.2f}".format(time_end - time_start))