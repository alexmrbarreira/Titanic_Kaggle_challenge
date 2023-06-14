import numpy as np
import pandas as pd

# =======================================================
# Prepare training data; add new features representations 
# =======================================================

def prepare_data(filename):
    df      = pd.read_csv(filename, index_col=0)
    df_size = df.shape[0]

    print ('The original data set has the following columns:')
    print (df.columns.tolist())

    nan_cols = df.columns[df.isnull().any()].tolist()
    print ('The following columns have NaN:')
    print (nan_cols)

    # Replace NaN in age with median passenger age
    median_passenger_age = df['Age'].dropna().median()
    df['Age'] = df['Age'].fillna(median_passenger_age)

    # Replace NaN in Embarked with most common
    most_comment_embarked_vals = df['Embarked'].mode()
    df['Embarked'] = df['Embarked'].fillna(most_comment_embarked_vals)

    # Replace NaN in fare with median value
    median_fare = df['Fare'].dropna().median()
    df['Fare'] = df['Fare'].fillna(median_fare)

    # Add a new feature which is the number of characters in passengers names
    name     = df['Name'].values        # name
    len_name = [ len(name[i]) for i in range(len(name)) ] # number of characters in name
    df['Namechars'] = len_name

    # Add a new feature which is 0 for male and 1 for female (to have a number representation of this)
    sex      = df['Sex'].values
    sex_vals = np.zeros(len(sex))
    sex_vals[np.where(sex ==   'male')] = 0
    sex_vals[np.where(sex == 'female')] = 1
    df['Sex_vals'] = sex_vals

    # Add a new feature which is 0 for embark-C, 1 for embark-Q and 2 for embark-S
    embarked      = df['Embarked'].values
    embarked_vals = np.zeros(len(embarked))
    embarked_vals[np.where(embarked=='C')] = 0
    embarked_vals[np.where(embarked=='Q')] = 1
    embarked_vals[np.where(embarked=='S')] = 2
    df['Embarked_vals'] = embarked_vals

    # Drop undesired columns and check for remaining nan
    features_todrop = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'] 
    df              = df.drop(features_todrop, axis = 1)
   
    print ('The final data set has the following columns:')
    print (df.columns.tolist())

    nan_cols = df.columns[df.isnull().any()].tolist()
    print ('The following columns still have NaN:')
    print (nan_cols)

    return df

print ('')
print ('Preparing training data ...')
df_train = prepare_data('data_store/data_kaggle_train.csv')
print ('')
print ('Preparing test data ...')
df_test  = prepare_data('data_store/data_kaggle_test.csv')

df_train.to_csv('data_store/data_modified_train.csv')
df_test.to_csv('data_store/data_modified_test.csv')

# =======================================================
# Split into training/validation
# =======================================================

fraction = 0.7
select   = np.random.rand(len(df_train)) < fraction

df_my_train = df_train[select]
df_my_valid = df_train[~select]

print ('')
print ('Original training set from Kaggle has size:')
print (df_train.shape[0])
print ('Original testing set from Kaggle has size:')
print (df_test.shape[0])
print ('Size of modified training set:')
print (df_my_train.shape[0])
print ('Size of modified validation set:')
print (df_my_valid.shape[0])
print ('Size of modified test set:')
print (df_test.shape[0])

df_my_train.to_csv('data_store/data_my_train.csv')
df_my_valid.to_csv('data_store/data_my_valid.csv')


