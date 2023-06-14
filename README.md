# Titanic_Kaggle_challenge

Simple set of scripts with a simple approach to the famous Kaggle Titanic Challenge. The challenge consists of predicting whether different passengers survive based on characteristics such as age, sex, ticket fare, etc. See the Kaggle website for more details: https://www.kaggle.com/competitions/titanic.

These scripts serve as a basic starting point to explore more elaborate data cleaning and modeling approaches. They compare three different modeling approaches: (i) models built "by hand" after visually inspecting basic data trends, (ii) a dense neural network and (iii) decision tree and random forest classifiers.

### Table of contents
- [Dependencies](#dependencies)
- [Script overview](#script-overview)
- [Results overview](#results-overview)

### Dependencies

- numpy, scipy and matplotlib
- pandas
- tensorflow
- scikit-learn

### Script overview

To run the whole pipeline, execute as follows:

*python prepare_training_data.py ; python plot_data_trends.py ; python model_1_byhand.py ; python model_2_dnn.py ; python model_3_trees.py ; python plot_features_importances.py*

#### prepare_training_data.py
This prepares the Kaggle dataset for learning: (i) it numerically encodes features such as gender and embarcation point, (ii) deals with NaN in the data, (iii) constructs new composite features, (iv) splits into 70% training and 30% validation sets.

#### plot_data_trends.py
This plots some basic data trends such as the number of passengers and survive probability as a function of the various features. It plots also the correlation matrix of the features.

#### model_1_byhand.py
This defines a couple of prediction models built by hand from visualy inspecting the basic data trends. Eg., a model where all female survive, or a model where also all male under the age of 9 also survive. It checks the performance of the models on the validation set.

#### model_2_dnn.py

#### model_3_trees.py

#### plot_features_importances.py

#### prepare_training_data.py
