"""
Terrance Taubes
2017951160
Homework 4: Recommender System
"""

import numpy as np
import pandas as pd
from time import clock
from datetime import datetime
import json

#   = commented out
##  = info / message
### = important notice / value


"""
- Preprocessing of review and functions to obtain dictionaries of reviews

Obtain:
    review:      DataFrame containing entire review.csv sorted by user_id
    review_dict: Dictionary containing ' user_id : {business_id : stars} '
"""


#-----------------------------------------------
#-----------------------------------------------

print "Start:", datetime.now().time()

##### Time to read review.csv
start = clock()
#####


""" review.csv """

## review: DataFrame containing entire review.csv sorted by user_id

review = pd.read_csv("review.csv", sep=",")
#print review.head()


#####
end = clock()
print 'Time to read csv:', (end-start)
#####


review = review.sort_values('user_id')
review['user_id'] = review['user_id'].astype('str')
review['business_id'] = review['business_id'].astype('str')
#print review
#print review.head()

review_users = review.loc[:, 'user_id']
review_users = review_users.as_matrix()
#print type(review_users[0])
#print review_users

review_index = review.index.values
#print review_index

#for a,b in zip(review_index, review_users):
#    print a,b


#for i in range(5):
#    current = review.loc[review_index[i]]
#    print current['business_id']



### 1,048,575 Reviews Total
###   476,608 Unique Users


#-----------------------------------------------
#-----------------------------------------------

## review_dict


review_dict = {}
count = 0
unique = 0
n_features = 82


##### Time to iterate reviews
start = clock()
#####


while (count < review.shape[0]):
#while (count < 50):

    unique = unique + 1
    review_dict[review_users[count]] = {}
    review_dict[review_users[count]]['reviews'] = {}
    current_ratings = []

    while (True):

        current_entry = review.loc[review_index[count]]
        review_dict[review_users[count]]['reviews'][current_entry['business_id']] = current_entry['stars']
        current_ratings.append(current_entry['stars'])

        count = count + 1

        if (count == review.shape[0] or review.loc[review_index[count]]['user_id'] != current_entry['user_id']):
            #print count-1, ':', review_dict[review_users[count-1]]['reviews']
            #print current_ratings
            avg = round(sum(current_ratings) / float(len(current_ratings)), 2)
            review_dict[review_users[count-1]]['avg'] = avg
            #print review_dict[review_users[count-1]]['avg']
            for bus in review_dict[review_users[count-1]]['reviews'].keys():
                review_dict[review_users[count-1]]['reviews'][bus] = round(review_dict[review_users[count-1]]['reviews'][bus] - avg, 2)
            #print review_dict[review_users[count-1]]['reviews']
            break

    #print review_dict[review_users[count-1]]


#####
end = clock()
#####

"""
start1 = clock()
for k, key in enumerate(review_dict.keys()):
    print str(k)+':','user_id:', key, '--', 'reviews:', review_dict[key]['reviews']
    #for key2 in review_dict[key].keys():
    #    print key2 + ":", review_dict[key][key2]

end1 = clock()
"""

print 'Time to build review_dict', (end-start)

print 'Unique user_id\'s:', unique

## review_dict as string

review_str = json.dumps(review_dict)

with open('review_dict.txt', 'w') as f:
    f.write(review_str)

#-----------------------------------------------

print "End:", datetime.now().time()
    
