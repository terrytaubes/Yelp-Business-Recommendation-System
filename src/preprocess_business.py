"""
Terrance Taubes
2017951160
Homework 4: Recommender System
"""

#   = commented out
##  = info / message
### = important notice / value

import numpy as np
import pandas as pd
from time import clock
from datetime import datetime
import math
import json


"""
- Preprocessing of business.csv to extract data

Obtain:
    business:   DataFrame containing entire business.csv
    attributes: List containing all attribute names
    X_dict:     Dictionary containing ' business_id : [theta] '
"""


#-----------------------------------------------
#-----------------------------------------------

print "Start:", datetime.now().time()

##### Time to read business_csv
start = clock()
#####


""" business.csv """

## business: DataFrame containing entire business.csv

business = pd.read_csv("business.csv", sep=",")
#print business.head()


#####
end = clock()
print 'Time to read csv:', (end-start)
#####

""" Business IDs """

## business_id: np.array containing all business ids

business_id = business.loc[:, 'business_id']
business_id = business_id.as_matrix()


### 144,072 Business Total


""" Collect Attributes (Features) """

remove_strings = ['Ambience: {', 'BusinessParking: {', 'GoodForMeal: {', \
                  'BestNights: {', 'Music: {', '}', '[', ']', '"', "'", " "]

replace_strings_1 = ['full_bar', 'beer_and_wine', 'free', 'True', 'quiet', \
                     'average', 'casual', 'dressy', 'outdoor']
replace_strings_0 = ['none', 'no', 'False', 'loud', 'very_loud']


## attributes: List containing all attribute names


attributes = []

###
start = clock()
###


for i in range(0, 400):

    temp_attr = business.loc[i, 'attributes']
    if type(temp_attr) == type(""):
        pass
    elif math.isnan(temp_attr):
        continue

    for string in remove_strings:
        if string in temp_attr:
            #print "found", string
            temp_attr = temp_attr.replace(string, "")
    
    for string in replace_strings_1:
        if string in temp_attr:
            temp_attr = temp_attr.replace(string, "1")

    for string in replace_strings_0:
        if string in temp_attr:
            temp_attr = temp_attr.replace(string, "0")
            
    temp_attr = temp_attr.split(",")


    for j in range(len(temp_attr)):
        #print (temp_attr[j].split(":"))[0]
        curr_attr = (temp_attr[j].split(":"))[0]
            
        if (curr_attr not in attributes):
            attributes.append(curr_attr)

###
end = clock()
print 'Time to collect features:', (end-start)
###


#for i in range(len(attributes)):
#    print i, ":", attributes[i]


### 82 Attributes Total



""" Building Feature Vector """


## X_dict: Dictionary containing 'business_id : [theta]' pairs
X_dict = {}

X = np.zeros((business.shape[0], len(attributes)))

###
start = clock()

for i in range(business.shape[0]):

    temp_busi = business.loc[i, 'attributes']
    if type(temp_busi) == type(""):
        pass
    elif math.isnan(temp_busi):
        X_dict[business_id[i]] = X[i].tolist()
        continue

    for string in remove_strings:
        if string in temp_busi:
            #print "found", string
            temp_busi = temp_busi.replace(string, "")
    
    for string in replace_strings_1:
        if string in temp_busi:
            temp_busi = temp_busi.replace(string, "1")

    for string in replace_strings_0:
        if string in temp_busi:
            temp_busi = temp_busi.replace(string, "0")
            
    temp_busi = temp_busi.split(",")

    #print temp_busi

    list_busi = []
    vals_0 = []
    vals_1 = []

    for j in range(len(temp_busi)):
        #print (temp_attr[j].split(":"))[0]
        vals = (temp_busi[j].split(":"))
                    
        vals_0.append(vals[0])
        vals_1.append(vals[1])

    list_busi.append(vals_0)
    list_busi.append(vals_1)
    
    #for k in range(len(list_busi[0])):
    #    print k, list_busi[0][k], list_busi[1][k]

    for f, feature in enumerate(attributes):
        if (feature in list_busi[0]):

            for a, att in enumerate(list_busi[0]):
                if (feature == att):
                    if (list_busi[1][a] == '1'):
                        X[i][f] = 1
                        break
                    else:
                        break

    X_dict[business_id[i]] = X[i].tolist()

###
end = clock()
print 'Time to build X_dict:', (end-start)
###


## X_dict as string

X_str = json.dumps(X_dict)
#print X_str

with open('X_dict.txt', 'w') as f:
    f.write(X_str)

#-----------------------------------------------

print "End:", datetime.now().time()
