from commons import *
import tensorflow as tf

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
# Build, compile and train model
# ========================================================

nepochs    = 50
batch_size = 32

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(N_features,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2 , activation = 'softmax')
])

# Callback to reduce learning rate when loss stops decreasing, and early stopping, and checkpoints
initial_lr     = 0.001
patience_val   = nepochs/20.
model_name     = 'model_store/model_2_dnn.h5'
reduce_lr      = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.95, patience=patience_val, min_lr=1.0e-7)
early_stopping = tf.keras.callbacks.EarlyStopping(patience = patience_val*10., monitor = 'val_accuracy')
checkpoint     = tf.keras.callbacks.ModelCheckpoint(model_name, save_best_only=True, monitor = 'val_accuracy')

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = initial_lr), loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])

print ('')
print ('Model summary:')
print (model.summary())
print ('Model input shape:' , model.input_shape)
print ('Model output shape:', model.output_shape)

history = model.fit(data_train_features, data_train_labels, 
                    steps_per_epoch  = int(np.ceil(N_train / batch_size)), 
                    epochs           = nepochs, 
                    shuffle          = True, 
                    batch_size       = batch_size, 
                    validation_data  = (data_valid_features, data_valid_labels), 
                    validation_steps = int(np.ceil(N_valid/batch_size)), 
                    callbacks        = [reduce_lr, early_stopping, checkpoint])

model = tf.keras.models.load_model(model_name)

# ==========================================================================
# Plot history
# ==========================================================================

nepochs_trained = len(history.history['loss'])
ee = range(1, nepochs_trained+1)

fig0 = plt.figure(0, figsize=(17.5, 7.))
fig0.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.12, wspace = 0.45, hspace = 0.30)

panel = fig0.add_subplot(1,3,1)
plt.title('Training loss', fontsize = 30)
plt.plot(ee, history.history['loss']    , c = 'b', linewidth = 2. , label = 'Loss')
plt.plot(ee, history.history['val_loss'], c = 'g', linewidth = 2. ,label = 'Validation loss')
plt.xlim(0.5*min(ee), 1.05*max(ee))
plt.xlabel(r'Epochs' , fontsize = 28)
plt.ylabel(r'Loss'          , fontsize = 28)
plt.tick_params(length=10., width=1.5 , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = 6, labelsize = 30)
params = {'legend.fontsize': 20}; plt.rcParams.update(params); plt.legend(loc = 'upper right', ncol = 1)

panel = fig0.add_subplot(1,3,2)
plt.title('Training accuracy', fontsize = 30)
plt.plot(ee, history.history['accuracy']    , c = 'b', linewidth = 2. , label = 'Accuracy')
plt.plot(ee, history.history['val_accuracy'], c = 'g', linewidth = 2. ,label = 'Validation Accuracy')
plt.xlim(0.5*min(ee), 1.05*max(ee))
plt.ylim(-0.01, 1.01)
plt.xlabel(r'Epochs' , fontsize = 28)
plt.ylabel(r'Accuracy' , fontsize = 28)
plt.tick_params(length=10., width=1.5 , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = 6, labelsize = 30)
params = {'legend.fontsize': 20}; plt.rcParams.update(params); plt.legend(loc = 'lower right', ncol = 1)

panel = fig0.add_subplot(1,3,3)
plt.title('Learning rate', fontsize = 30)
plt.plot(ee, history.history['lr']      , c = 'r', linewidth = 2. ,label = 'Learning rate')
plt.xlim(0.5*min(ee), 1.05*max(ee))
plt.xlabel(r'Epochs' , fontsize = 28)
plt.ylabel(r'Learning rate'          , fontsize = 28)
plt.tick_params(length=10., width=1.5 , bottom=True, top=True, left=True, right=True, direction = 'in', which = 'major', pad = 6, labelsize = 30)
params = {'legend.fontsize': 20}; plt.rcParams.update(params); plt.legend(loc = 'lower left', ncol = 1)

fig0.savefig('fig_store/fig_model_2_dnn_training_history.png')

plt.show()

# ========================================================
# Performance in validation set
# ========================================================

prediction = model.predict(data_valid_features)
prediction = [np.argmax(prediction[i]) for i in range(N_valid)]

accuracy   = len(np.where(prediction == data_valid_labels[:,0])[0]) / N_valid

print ('')
print ('Model accuracy in the validation set:', accuracy)
print ('')

# ========================================================
# Make prediction for Kaggle test
# ========================================================

df_test            = pd.read_csv('data_store/data_modified_test.csv')
N_test             = df_test.shape[0]
pid                = df_test['PassengerId'].values
data_test_features = df_test.drop(['PassengerId'], axis = 1).values
for i in range(N_features):
    data_test_features[:,i] /= max(data_test_features[:,i])

prediction_test = model.predict(data_test_features)
prediction_test = [np.argmax(prediction_test[i]) for i in range(N_test)]

print ('')
print ('Predicted fraction of survivors:', sum(prediction_test)/len(prediction_test))
print ('')

# Save to file
df_out = pd.DataFrame(np.transpose(np.array([pid.astype(int), np.array(prediction_test).astype(int)])), columns = ['PassengerId', 'Survived'])
df_out.to_csv('prediction_store/data_kaggle_my_prediction_model_2_dnn.csv', index = False)
