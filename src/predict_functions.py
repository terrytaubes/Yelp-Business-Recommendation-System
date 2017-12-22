"""
Terrance Taubes
2017951160
Homework 4: Recommender System
"""

import numpy as np
import pandas as pd
from time import clock
from datetime import datetime
from scipy import optimize
from math import ceil, floor
import json


## Predictions and Functions module

### 1,048,575 Reviews Total
###   476,608 Users Total
###   144,072 Business Total
###        82 Attributes Total

###   182,545 Samples Total


#-----------------------------------------------

## reading of csv's and json's

print "Start:", datetime.now().time()

########## sample.csv
start = clock()
sample = pd.read_csv('sample.csv', sep=',')
end = clock()
print 'Time to read sample:', (end-start)
##########

########## user_dict.txt
start = clock()
with open('user_dict.txt', 'r') as f2:
    user_dict = json.load(f2)
end = clock()
print 'Time to load user_dict:', (end-start)
##########

########## X_dict.txt
start = clock()
with open('X_dict.txt', 'r') as f1:
    X_dict = json.load(f1)
end = clock()
print 'Time to load X_dict:', (end-start)
##########


#-----------------------------------------------
#-----------------------------------------------


##### Functions #####

# Mimic Octave's Reshape
# Separate input X_theta vector into X and theta vectors
def reshape(X_theta, n_users, n_business, n_features):

    feats = X_theta[:n_business * n_features]
    X = feats.reshape((n_features, n_business)).transpose()
    params = X_theta[n_business * n_features:]
    theta = params.reshape(n_features, n_users).transpose()

    return X, theta



# Regularized Cost Function - returns J (cost)
def reg_cost_function(X_theta, y, n_users, n_business, n_features, lmda):
    X, theta = reshape(X_theta, n_users, n_business, n_features)


    J = (0.5) * sum( (X.dot(theta.T) - y) ** 2) \
        + ((lmda / 2.0) * (sum(sum(theta**2)) + sum(sum(X**2))))

    return J

# Regularized Gradient Computation - returns grad_vector (X_grad, theta_grad)
def reg_gradient(X_theta, y, n_users, n_business, n_features, lmda):
    X, theta = reshape(X_theta, n_users, n_business, n_features)

    error = X.dot(theta.T) - y
    X_grad = error.dot(theta) + (lmda * X)
    theta_grad = error.T.dot(X) + (lmda * theta)


    grad_vector = np.r_[X_grad.T.flatten(), theta_grad.T.flatten()]

    return grad_vector



#-----------------------------------------------
#-----------------------------------------------


n_features = 82
lmda = 1

## predict_csv : send predictions to this csv

predict_csv = pd.DataFrame(columns=['user_id-business_id', 'stars'])

init_param = np.random.randn(n_features)

#####
print "Begin:", datetime.now().time()
start = clock()
#####

for i in range(sample.shape[0]):

    curr_sample = sample.loc[i, 'user_id-business_id']
    curr_rating = sample.loc[i, 'stars']
    
    split = curr_sample.split('-')
    curr_user, curr_business = split[0], split[1]

    curr_feats = []

    if (curr_user in user_dict.keys()):
        for business in user_dict[curr_user]['reviews'].keys():
            curr_feats.append(X_dict[business])

        features = np.array(curr_feats)
        y = np.array(user_dict[curr_user]['rate_matrix'])
        n_business = features.shape[0]

        X_theta = np.r_[features.T.flatten(), init_param]

    
        min_cost_optimal_params = optimize.fmin_cg( \
        reg_cost_function, fprime=reg_gradient, x0=X_theta, \
        args=(y, 1, n_business, n_features, lmda), \
        maxiter=100, disp=False, full_output=True)
    


        final_X_theta = min_cost_optimal_params[0]

        final_X, final_theta = reshape(final_X_theta, 1, n_business, n_features)


        for b in range(len(user_dict[curr_user]['bus_matrix'] )):
            X_dict[user_dict[curr_user]['bus_matrix'][b]] = final_X[b]

        curr_X = np.array(X_dict[curr_business])
        
        o_predict = curr_X.dot(final_theta.T) + user_dict[curr_user]['avg']

    else:

        curr_feats.append(X_dict[curr_business])

        features = np.array(curr_feats)
        y = np.array([[curr_rating]])
        n_business = features.shape[0]

        X_theta = np.r_[features.T.flatten(), init_param]
        
        min_cost_optimal_params = optimize.fmin_cg( \
        reg_cost_function, fprime=reg_gradient, x0=X_theta, \
        args=(y, 1, n_business, n_features, lmda), \
        maxiter=100, disp=False, full_output=True)

        final_X_theta = min_cost_optimal_params[0]

        final_X, final_theta = reshape(final_X_theta, 1, n_business, n_features)


        curr_X = np.array(X_dict[curr_business])
        
        o_predict = curr_X.dot(final_theta.T)


        
    ## prediction adjustments and roundings
    if (o_predict <= 1.0):
        predict = 1

    elif (o_predict >= 5.0):
        predict = 5

    #else:
    #    predict = int(floor(o_predict))
    
    elif ((o_predict - floor(o_predict)) == 0):
        predict = int(o_predict)

    else:
        predict = int(ceil(o_predict))
        
    #elif ((o_predict - floor(o_predict)) < .5):
    #    predict = int(floor(o_predict))

    #elif ((o_predict - floor(o_predict)) >= .5):
    #     predict = int(ceil(o_predict))
    

    #else:
    #    print "user:", curr_user
    #    print o_predict, predict, curr_rating


    predict_csv = predict_csv.append({'user_id-business_id': curr_sample, 'stars':predict}, ignore_index=True)
    
    #if i % 25000 == 0:
    #    end = clock()
    #    print 'complete:', total, (end-start)
    
    if i == 10:
        break

#####
end = clock()
#####

#print predict_csv

print 'Time to predict:', (end-start)

"""
predict_csv['stars'] = predict_csv['stars'].astype(int)

predict_csv.to_csv('predict.csv', index=False)
"""

#-----------------------------------------------
#-----------------------------------------------


print "End:", datetime.now().time()
