# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 21:35:49 2020

@author: HP
"""

import numpy as np
from scipy.spatial import distance

def cos_sim(array1, array2, array3, array4):

#A = np.array([[1,23,2,5,6,2,2,6,2],[12,4,5,5],[1,2,4],[1],[2],[2]], dtype=object )
#B = np.array([[1,23,2,5,6,2,2,6,2],[12,4,5,5],[1,2,4],[1],[2],[2]], dtype=object )

    #Aflat = np.hstack(array1)
    #Bflat = np.hstack(array2)
    #ml = max(len(array1),len(array2))
    #array1 = np.concatenate((array1 , np.zeros(ml-len(array1))))
    #array2 = np.concatenate((array2 , np.zeros(ml-len(array2))))

    ml1=max(len(array1),len(array2))
    ml2=max(len(array3),len(array4))

    array1 = np.concatenate((array1, np.zeros(ml1 - len(array1))))
    array2 = np.concatenate((array2, np.zeros(ml1 - len(array2))))
    array3 = np.concatenate((array3, np.zeros(ml2 - len(array3))))
    array4 = np.concatenate((array4, np.zeros(ml2 - len(array4))))


    # print("array1 ",array1)
    # print("array2 ",array2)

    dist_pose = distance.cosine(array1, array2)
    dist_his = distance.cosine(array3, array4)
    #dist = distance.cosine(array1, array2)
    similiarity_pose=1-dist_pose
    similiarity_his=1-dist_his

    
    print("Disatnce_pose",dist_pose)
    print ("Similiarity_pose",similiarity_pose)
    print("Disatnce_his",dist_his)
    print ("Similiarity_his",similiarity_his)

    #suma1=0
    #sumb1=0
    #total=0
    #for i,j in zip(array1,array2):
        #suma1+=i*i
        #sumb1+=j*j
        #total+=i*j
        
    #cosine_sim=total/((math.sqrt(suma1))*(math.sqrt(sumb1)))
    #print(cosine_sim)


    if similiarity_his >= .7 and similiarity_pose> .7:
        return 0
    else:
        return 1
