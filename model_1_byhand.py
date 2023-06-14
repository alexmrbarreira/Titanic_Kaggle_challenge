from commons import *

# ======================================================== 
# Load data
# ======================================================== 

df_train = pd.read_csv('data_store/data_my_train.csv')
df_valid = pd.read_csv('data_store/data_my_valid.csv')

validation_survived = df_valid['Survived'].values
N_valid = len(validation_survived)

# ======================================================== 
# Build simple model
# ========================================================

def get_prediction(df):
    N_data = df.shape[0]
    predicted_survived  = np.zeros(N_data)
    for i in range(N_data):
        sex_now    = df['Sex_vals'][i]
        class_now  = df['Pclass'][i]
        age_now    = df['Age'][i]
        name_now   = df['Namechars'][i]
        embark_now = df['Embarked_vals'][i]
        sibsp_now  = df['SibSp'][i]
        fare_now   = df['Fare'][i]
        # 1) Save all female
        if (sex_now == 1):
            predicted_survived[i] = 1
        # 2) Kill all male, save some exceptions
        else:
#            # 2.1) Save all male with more than some name characters
#            if (name_now > 35.):
#                predicted_survived[i] = 1
#            # 2.2) Save all male with less some years
            if (age_now < 9):
                predicted_survived[i] = 1
#            # 2.3) Save all male from Cherbourg
#            if (embark_now == 1):
#                predicted_survived[i] = 1
#            # 2.4) Save all male with fare between some values
#            if ( (fare_now > 100.) and (fare_now < 150.) ):
#                predicted_survived[i] = 1
#            # 2.5) Save all male in first class
#            if ( class_now == 1): 
#                predicted_survived[i] = 1
    return predicted_survived

# ======================================================== 
# Performance in validation set 
# ========================================================

prediction_valid = get_prediction(df_valid)
accuracy = len(np.where(prediction_valid == validation_survived)[0]) / len(validation_survived)

print ('')
print ('Model accuracy in the validation set:', accuracy)
print ('')

# ========================================================
# Make prediction for Kaggle test set
# ========================================================

df_test = pd.read_csv('data_store/data_modified_test.csv')
pid     = df_test['PassengerId'].values

prediction_test = get_prediction(df_test)

print ('')
print ('Predicted fraction of survivors:', sum(prediction_test)/len(prediction_test))
print ('')

# Save to file
df_out = pd.DataFrame(np.transpose(np.array([pid.astype(int), prediction_test.astype(int)])), columns = ['PassengerId', 'Survived'])
df_out.to_csv('prediction_store/data_kaggle_my_prediction_model_1_byhand.csv', index = False)

