import numpy as np
import pandas as pd
from time import clock
from datetime import datetime
import json
from sys import getsizeof

print "Start:", datetime.now().time()

##### Time to load
start = clock()
#####

## Load in json file of our processed training ratings
with open('review_dict.txt', 'r') as f1:
    review_dict = json.load(f1)

##
with open('X_dict.txt', 'r') as f2:
	X_dict = json.load(f2)

## Features (# of keywords) in matrix
n_features = 82

#####
end = clock()
print 'Time to load:', (end-start)
#####


##### Time to build user_dict
start = clock()
#####

#init_theta = np.random.randn(n_features, 1).tolist()

user_dict = {}
feat_dict = {}

count = 0

# for each user in review_dict
for user in review_dict.keys():          

    #print 'user:', user
    #feat_dict[user] = {}
    #feat_dict[user]['feat_matrix'] = []
    review_dict[user]['rate_matrix'] = []
    review_dict[user]['bus_matrix'] = []

    # for each business in review_dict[user]['reviews]
    for business in review_dict[user]['reviews'].keys():
        #feat_dict[user]['feat_matrix'].append(X_dict[business])
        review_dict[user]['rate_matrix'].append([review_dict[user]['reviews'][business]])
        review_dict[user]['bus_matrix'].append(business)        

        #print 'business:', business

    #review_dict[user]['theta_matrix'] = init_theta
    #del review_dict[user]['reviews']


    #if (count < 2000):
    #    user_dict[user] = review_dict[user]
    
    count += 1
    #if (count == 2000):
    #    break
    
    if (count % 50000 == 0):
        print 'count:', count
        
    #if (count >= 247500 and count % 250 == 0):
    #    print 'count:', count


#####
end = clock()
print 'Time to build user_dict:', (end-start)
#####


"""
##### Time to create user_str
start = clock()
print start
#####

user_str = json.dumps(review_dict)

#####
end = clock()
print 'Time to create user_str:', (end-start)
#####
"""


##### Time to write user_dict to file
print datetime.now().time()
start = clock()
#####

with open('user_dict.txt', 'w') as f:
    json.dump(review_dict, f)

#with open('feat_dict.txt', 'w') as f1:
#    json.dump(feat_dict, f1)

#####
end = clock()
print 'Time to write user_dict:', (end-start)
#####


#-----------------------------------------------

print "End:", datetime.now().time()
    
