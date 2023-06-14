from commons import *
import tensorflow as tf
import pickle

# ======================================================== 
# Load data and models
# ======================================================== 

df_train, df_valid, data_train_features, data_valid_features, data_train_labels, data_valid_labels, N_features, N_train, N_valid = load_training_data() # function defined in commons.py

model_2_dnn           = tf.keras.models.load_model('model_store/model_2_dnn.h5')
model_3_decision_tree = pickle.load(open('model_store/model_3_decision_tree.pickle', "rb"))
model_3_random_forest = pickle.load(open('model_store/model_3_random_forest.pickle', "rb"))

# ========================================================
# Measure feature importance by randomizing entries
# ========================================================

def get_feature_importance(model, data_features, data_labels, is_model_cnn):
    N_data     = len(data_features[:,0])
    N_features = len(data_features[0,:])
    # Default model prediction and accuracy
    prediction_default = model.predict(data_features)
    if(is_model_cnn):
        prediction_default = [np.argmax(prediction_default[i]) for i in range(N_data)]
    accuracy_default   = len(np.where(prediction_default == data_labels[:,0])[0]) / N_data
    # Measure feature importance by size of accuracy loss after randomization
    feature_importance = np.zeros(N_features)
    for j in range(N_features):
        data_features_now = np.copy(data_features)
        np.random.shuffle(data_features_now[:,j])
        prediction_now = model.predict(data_features_now)
        if(is_model_cnn):
            prediction_now = [np.argmax(prediction_now[i]) for i in range(N_data)]
        accuracy_now   = len(np.where(prediction_now == data_labels[:,0])[0]) / N_data
        feature_importance[j] = accuracy_default/accuracy_now
    return feature_importance, accuracy_default

def get_average_feature_importance(model, data_features, data_labels, is_model_cnn, N_random):
    N_features                 = len(data_features[0,:])
    average_feature_importance = np.zeros(N_features)
    for i in range(N_random):
        if(np.mod(i, N_random/10)==0):
            print ('Random realization', i, 'out of', N_random)
        average_feature_importance += get_feature_importance(model, data_features, data_labels, is_model_cnn)[0]
    return average_feature_importance/N_random, get_feature_importance(model, data_features, data_labels, is_model_cnn)[1]

# Number of random realizations
N_random = 10

# Compute importances using the training data
print ('')
print ('Computing importances for the training data ... ')
print ('')
print ('Getting feature importances for the neural network ... ')
feature_importance_model_2_dnn_train           = get_average_feature_importance(model_2_dnn          , data_train_features, data_train_labels, True, N_random)
print ('Getting feature importances for the decision tree ... ')
feature_importance_model_3_decision_tree_train = get_average_feature_importance(model_3_decision_tree, data_train_features, data_train_labels, False, N_random)
print ('Getting feature importances for the random forest ... ')
feature_importance_model_3_random_forest_train = get_average_feature_importance(model_3_random_forest, data_train_features, data_train_labels, False, N_random)

# Compute importances using the validation data
print ('')
print ('Computing importances for the validation data ... ')
print ('')
print ('Getting feature importances for the neural network ... ')
feature_importance_model_2_dnn_valid           = get_average_feature_importance(model_2_dnn          , data_valid_features, data_valid_labels, True, N_random)
print ('Getting feature importances for the decision tree ... ')
feature_importance_model_3_decision_tree_valid = get_average_feature_importance(model_3_decision_tree, data_valid_features, data_valid_labels, False, N_random)
print ('Getting feature importances for the random forest ... ')
feature_importance_model_3_random_forest_valid = get_average_feature_importance(model_3_random_forest, data_valid_features, data_valid_labels, False, N_random)

# ========================================================
# Make plot
# ========================================================

def plotter(feature_names, feature_importance, accuracy, color, label):
    # Sort by decreasing importance
    argsort = np.argsort(feature_importance)[::-1]
    feature_importance = feature_importance[argsort]
    feature_names = [feature_names[i] for i in argsort]
    xx = range(len(feature_names))
    plt.scatter(xx, feature_importance, c = color, s = 80, label = label)
    plt.plot(  xx, feature_importance , c = color, linewidth = 2., linestyle = 'dashed')
    plt.ylim(0.95, 1.30)
    plt.xticks(xx, feature_names, fontsize = labelsize, rotation = 25.)
    plt.tick_params(length=tick_major, width=tickwidth , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = tickpad, labelsize = ticksize-11)
    params = {'legend.fontsize': 20}; plt.rcParams.update(params); plt.legend(loc = 'upper right', ncol = 1)
    plt.annotate(r'Accuracy: '  + str(round(accuracy, 2)), xy = (0.52, 0.75), xycoords = 'axes fraction', c = color, fontsize = text_font)
    return 0


fig1 = plt.figure(1, figsize=(17., 10.))
fig1.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.10, wspace = 0.25, hspace = 0.35)

feature_names = ['Class', 'Age', 'SibSp', 'Parch', 'Fare', 'Namechars', 'Sex', 'Embarked'] 

# Add upper panels for training feature importances
panel = fig1.add_subplot(2,3,1)
plotter(feature_names, feature_importance_model_3_decision_tree_train[0], feature_importance_model_3_decision_tree_train[1], 'b', 'Decision tree')
plt.ylabel('1 / accuracy loss by randomization', fontsize = labelsize-4)
panel = fig1.add_subplot(2,3,2)
plt.title('Feature importance (training accuracy loss by randomization)', fontsize = title_font)
plotter(feature_names, feature_importance_model_3_random_forest_train[0], feature_importance_model_3_random_forest_train[1], 'g', 'Random forest')
panel = fig1.add_subplot(2,3,3)
plotter(feature_names, feature_importance_model_2_dnn_train[0], feature_importance_model_2_dnn_train[1], 'r', 'Neural network')

# Add lower panels for validation feature importances
panel = fig1.add_subplot(2,3,4)
plotter(feature_names, feature_importance_model_3_decision_tree_valid[0], feature_importance_model_3_decision_tree_valid[1], 'b', 'Decision tree')
plt.ylabel('1 / accuracy loss by randomization', fontsize = labelsize-4)
panel = fig1.add_subplot(2,3,5)
plt.title('Feature importance (validation accuracy loss by randomization)', fontsize = title_font)
plotter(feature_names, feature_importance_model_3_random_forest_valid[0], feature_importance_model_3_random_forest_valid[1], 'g', 'Random forest')
panel = fig1.add_subplot(2,3,6)
plotter(feature_names, feature_importance_model_2_dnn_valid[0], feature_importance_model_2_dnn_valid[1], 'r', 'Neural network')

fig1.savefig('fig_store/fig_feature_importances.png')

#plt.show()
