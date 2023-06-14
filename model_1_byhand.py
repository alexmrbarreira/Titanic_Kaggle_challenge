from commons import *

# ======================================================== 
# Load data
# ======================================================== 

df_train, df_valid, data_train_features, data_valid_features, data_train_labels, data_valid_labels, N_features, N_train, N_valid = load_training_data() # function defined in commons.py

# ======================================================== 
# Build simple models
# ========================================================

def get_prediction_model_1_save_female(df):
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
        # 1) Save all female, kill all male
        if (sex_now == 1):
            predicted_survived[i] = 1
    return predicted_survived

def get_prediction_model_1_save_female_young_male(df):
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
        # 1) Save all female and all male below age 9
        if (sex_now == 1):
            predicted_survived[i] = 1
        else:
            if (age_now < 9):
                predicted_survived[i] = 1
    return predicted_survived

# ======================================================== 
# Performance in validation set 
# ========================================================

prediction_model_1_save_female_valid            = get_prediction_model_1_save_female(df_valid)
prediction_model_1_save_female_young_male_valid = get_prediction_model_1_save_female_young_male(df_valid)

accuracy_model_1_save_female_valid            = len(np.where(prediction_model_1_save_female_valid            == data_valid_labels[:,0])[0]) / len(data_valid_labels)
accuracy_model_1_save_female_young_male_valid = len(np.where(prediction_model_1_save_female_young_male_valid == data_valid_labels[:,0])[0]) / len(data_valid_labels)


print ('')
print ('Model "Save all female" accuracy in the validation set:', accuracy_model_1_save_female_valid)
print ('Model "Save all female and young male" accuracy in the validation set:', accuracy_model_1_save_female_young_male_valid)
print ('')

# ========================================================
# Make prediction for Kaggle test set
# ========================================================

df_test, data_test_features, N_test, pid = load_test_data() # function defined in commons.py

prediction_model_1_save_female_test            = get_prediction_model_1_save_female(df_test)
prediction_model_1_save_female_young_male_test = get_prediction_model_1_save_female_young_male(df_test)

print ('')
print ('Predicted fraction of survivors in model "Save all female":'               , sum(prediction_model_1_save_female_test)/len(prediction_model_1_save_female_test))
print ('Predicted fraction of survivors in model "Save all female and young male":', sum(prediction_model_1_save_female_young_male_test)/len(prediction_model_1_save_female_young_male_test))
print ('')

# Save to file "Save all female"
df_out = pd.DataFrame(np.transpose(np.array([pid.astype(int), prediction_model_1_save_female_test.astype(int)])), columns = ['PassengerId', 'Survived'])
df_out.to_csv('prediction_store/data_kaggle_my_prediction_model_1_byhand_save_female.csv', index = False)

# Save to file "Save all female and young male"
df_out = pd.DataFrame(np.transpose(np.array([pid.astype(int), prediction_model_1_save_female_young_male_test.astype(int)])), columns = ['PassengerId', 'Survived'])
df_out.to_csv('prediction_store/data_kaggle_my_prediction_model_1_byhand_save_female_young_male.csv', index = False)


