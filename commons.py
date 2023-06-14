import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': True, 'mathtext.fontset': 'stix'})

# Ploting parameters
labelsize   = 26
ticksize    = 26
tick_major  = 10.
tick_minor  = 5.
tickwidth   = 1.5
tickpad     = 6.
title_font  = 30
text_font   = 20
legend_font = 20
alpha_c     = 0.3
ls_def  = 'solid'
ls_def2 = 'dashed'
ls_def3 = 'dotted'
ln_def  = 2
c_def   = 'darkorange'
c_def2  = 'g'

# function that gets probability to survive of a given feature in a dataframe df
def get_prob_inbins(feature, nbins, df):
    bin_edges = np.linspace(df[feature].min(), df[feature].max(), nbins+1)
    bin_vals  = (bin_edges[1::] + bin_edges[:-1])/2.
    prob      = np.zeros(nbins)
    for i in range(nbins):
        df_tmp = df[ (df[feature] >= bin_edges[i]) & (df[feature] < bin_edges[i+1]) ]
        if(df_tmp.shape[0] != 0):
            prob[i] = ( df_tmp['Survived'].sum() / df_tmp.shape[0] )
    return bin_vals, prob.tolist()

# function that gets the number distribution of a given feature, i.e., number of passengers with that feature
def get_dist_inbins(feature, nbins, df):
    bin_edges = np.linspace(df[feature].min(), df[feature].max(), nbins+1)
    bin_vals  = (bin_edges[1::] + bin_edges[:-1])/2.
    dist      = np.zeros(nbins)
    for i in range(nbins):
        df_tmp = df[ (df[feature] >= bin_edges[i]) & (df[feature] < bin_edges[i+1]) ]
        if(df_tmp.shape[0] != 0):
            dist[i] = df_tmp.shape[0]
    dist /= max(dist)
    return bin_vals, dist.tolist()


# Function that loads the training data
def load_training_data():
    df_train = pd.read_csv('data_store/data_my_train.csv')
    df_valid = pd.read_csv('data_store/data_my_valid.csv')

    # Training/validation features (passenger characteristics)
    data_train_features = df_train.drop(['PassengerId', 'Survived'], axis = 1)
    data_valid_features = df_valid.drop(['PassengerId', 'Survived'], axis = 1)
    print ('')
    print ('Training with the folowing columns ... ')
    print (data_train_features.columns.tolist())
    data_train_features = data_train_features.values
    data_valid_features = data_valid_features.values

    # Training/validation labels (survived or not)
    data_train_labels = df_train[['Survived']].values
    data_valid_labels = df_valid[['Survived']].values

    N_features = np.shape(data_train_features)[1]
    N_train    = len(data_train_labels)
    N_valid    = len(data_valid_labels)

    # Normalize features by their max
    for i in range(N_features):
        data_train_features[:,i] /= max(data_train_features[:,i])
    for i in range(N_features):
        data_valid_features[:,i] /= max(data_valid_features[:,i])

    print ('')
    print ('Shape of training data:')
    print ('Features:', np.shape(data_train_features))
    print ('Labels:', np.shape(data_train_labels))
    print ('Shape of validation data:')
    print ('Features:', np.shape(data_valid_features))
    print ('Labels:', np.shape(data_valid_labels))

    return df_train, df_valid, data_train_features, data_valid_features, data_train_labels, data_valid_labels, N_features, N_train, N_valid

# Function that loads the test data
def load_test_data():
    df_test            = pd.read_csv('data_store/data_modified_test.csv')
    N_test             = df_test.shape[0]
    pid                = df_test['PassengerId'].values
    data_test_features = df_test.drop(['PassengerId'], axis = 1).values
    for i in range(N_features):
        data_test_features[:,i] /= max(data_test_features[:,i])
    return df_test, data_test_features, N_test, pid

