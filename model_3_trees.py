from commons import *
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# ======================================================== 
# Load data
# ======================================================== 

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

# ======================================================== 
# Build and train decision tree model
# ========================================================

# Create and train model
model_decision_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 14)
model_decision_tree.fit(data_train_features, data_train_labels[:,0])

# Predictions for train and validation set
prediction_decision_tree_train = model_decision_tree.predict(data_train_features)
prediction_decision_tree_valid = model_decision_tree.predict(data_valid_features)

accuracy_decision_tree_train = len(np.where(prediction_decision_tree_train == data_train_labels[:,0])[0]) / N_train
accuracy_decision_tree_valid = len(np.where(prediction_decision_tree_valid == data_valid_labels[:,0])[0]) / N_valid

print ('')
print ('Performance of the decision tree classifier')
print ('Accuracy on the training set:', accuracy_decision_tree_train, '; % of survivors:', sum(prediction_decision_tree_train)/len(prediction_decision_tree_train))
print ('Accuracy on the validation set:', accuracy_decision_tree_valid, '; % of survivors:', sum(prediction_decision_tree_valid)/len(prediction_decision_tree_valid))

pickle.dump(model_decision_tree, open('model_store/model_3_decision_tree.pickle', "wb"))

# ========================================================
# Build and train random forest model
# ========================================================

# Create and train model
model_random_forest = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', random_state = 14)
model_random_forest.fit(data_train_features, data_train_labels[:,0])

# Predictions for train and validation set
prediction_random_forest_train = model_random_forest.predict(data_train_features)
prediction_random_forest_valid = model_random_forest.predict(data_valid_features)

accuracy_random_forest_train = len(np.where(prediction_random_forest_train == data_train_labels[:,0])[0]) / N_train
accuracy_random_forest_valid = len(np.where(prediction_random_forest_valid == data_valid_labels[:,0])[0]) / N_valid

print ('')
print ('Performance of the random forest classifier')
print ('Accuracy on the training set:', accuracy_random_forest_train, '; % of survivors:', sum(prediction_random_forest_train)/len(prediction_random_forest_train))
print ('Accuracy on the validation set:', accuracy_random_forest_valid, '; % of survivors:', sum(prediction_random_forest_valid)/len(prediction_random_forest_valid))

pickle.dump(model_random_forest, open('model_store/model_3_random_forest.pickle', "wb"))

# ======================================================== 
# Plot feature importances 
# ========================================================

feat_importances_decision_tree = model_decision_tree.feature_importances_
feat_importances_random_forest = model_random_forest.feature_importances_
feat_names                     = df_train.drop(['PassengerId', 'Survived'], axis = 1).columns.tolist() 
xx                             = range(len(feat_names))

argsort                        = np.argsort(feat_importances_random_forest)[::-1]
feat_importances_random_forest = feat_importances_random_forest[argsort]
feat_importances_decision_tree = feat_importances_decision_tree[argsort]
feat_names                     = [feat_names[i] for i in argsort] 

def plotter(x, y, color, label_use):
    plt.scatter(x, y, c = color, s = 80, label = label_use)
    plt.plot(x, y, c = color, linewidth = 2., linestyle = 'dashed')

fig1 = plt.figure(1, figsize=(11., 7.))
fig1.subplots_adjust(left=0.10, right=0.94, top=0.94, bottom=0.18, wspace = 0.35, hspace = 0.45)

panel = fig1.add_subplot(1,1,1)
plt.title(r'Feature importance (impurity decrease)', fontsize = title_font)
plotter(xx, feat_importances_decision_tree, 'b', 'Decision tree')
plotter(xx, feat_importances_random_forest, 'g', 'Random forest')
plt.xticks(xx, feat_names, fontsize = labelsize-6, rotation = 25.)
plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize)
params = {'legend.fontsize': 20}; plt.rcParams.update(params); plt.legend(loc = 'upper right', ncol = 1)
plt.annotate(r'Decision tree val. accuracy: ' + str(round(accuracy_decision_tree_valid, 2)), xy = (0.05, 0.25), xycoords = 'axes fraction', c = 'b', fontsize = text_font)
plt.annotate(r'Random forest val. accuracy: ' + str(round(accuracy_random_forest_valid, 2)), xy = (0.05, 0.18), xycoords = 'axes fraction', c = 'g', fontsize = text_font)

fig1.savefig('fig_store/fig_model_3_feature_importances.png')

plt.show()

# ========================================================
# Make prediction for Kaggle test
# ========================================================

df_test            = pd.read_csv('data_store/data_modified_test.csv')
N_test             = df_test.shape[0]
pid                = df_test['PassengerId'].values
data_test_features = df_test.drop(['PassengerId'], axis = 1).values
for i in range(N_features):
    data_test_features[:,i] /= max(data_test_features[:,i])

prediction_decision_tree_test = model_decision_tree.predict(data_test_features)
prediction_random_forest_test = model_random_forest.predict(data_test_features)

print ('')
print ('Predicted fraction of survivors from decision tree:', sum(prediction_decision_tree_test)/len(prediction_decision_tree_test))
print ('Predicted fraction of survivors from random forest:', sum(prediction_random_forest_test)/len(prediction_random_forest_test))
print ('')

# Save to file - decision tree
df_out = pd.DataFrame(np.transpose(np.array([pid.astype(int), np.array(prediction_decision_tree_test).astype(int)])), columns = ['PassengerId', 'Survived'])
df_out.to_csv('prediction_store/data_kaggle_my_prediction_model_3_decision_tree.csv', index = False)

# Save to file - random forest
df_out = pd.DataFrame(np.transpose(np.array([pid.astype(int), np.array(prediction_random_forest_test).astype(int)])), columns = ['PassengerId', 'Survived'])
df_out.to_csv('prediction_store/data_kaggle_my_prediction_model_3_random_forest.csv', index = False)

