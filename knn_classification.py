import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def split_data(x, y):
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x, y, test_size=0.2, random_state=42)
    print(x_train_data.shape, x_test_data.shape, y_train_data.shape, y_test_data.shape)

    return x_train_data, x_test_data, y_train_data, y_test_data


def normalize_data(input_file):

    input_file.replace({'Gender': {'F': 0, 'M': 1}}, inplace=True)
    column_name = input_file.columns

    for name in column_name:
        if name != 'Gender':
            dummy_var = pd.get_dummies(input_file[name], prefix=name+'_' , dtype=int)
            input_file = pd.concat([input_file, dummy_var], axis=1)

    for c in column_name:
        if c != 'Gender':
            input_file.drop(c, axis=1, inplace=True)

    print(input_file.info())

    return input_file


def visualize_data(input_file):

    column_name = list(input_file.columns)
    column_name.remove('Gender')
    print(column_name)
    for name in range(len(column_name)):
        plt.subplot(4, 5, name+1)
        print(input_file[column_name[name]].shape, input_file['Gender'].shape)
        plt.scatter(input_file[column_name[name]], input_file['Gender'])

    plt.show()


def performance_metrics(x_train, x_test, y_train, y_test, predicted_val, knn_obj):
    # scores
    print("Training score:", knn_obj.score(x_train, y_train))
    print("Test score:", knn_obj.score(x_test, y_test))

    # reports
    print("confusion matrix:\n", confusion_matrix(y_test, predicted_val))
    print("classification report:\n", classification_report(y_test, predicted_val))

def data_processing():
    source_file = pd.read_csv("gender_classification.csv")
    print(source_file.info())

    # normalize data
    normalized_input_file = normalize_data(source_file)

    # visualize data
    #visualize_data(normalized_input_file)

    # define predictors and target variables
    x = normalized_input_file.drop(['Gender'], axis=1).values
    y = normalized_input_file['Gender'].values

    # split data
    x_train, x_test, y_train, y_test = split_data(x, y)

    # KNN classification
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(x_train, y_train)
    predicted_val = knn.predict(x_test)
    print("predicted values:", predicted_val)
    print("Actual values:", y_test)

    # Check performance
    performance_metrics(x_train, x_test, y_train, y_test, predicted_val, knn)

    # check performance on more k-neighbor values
    neighbor_val = np.arange(1, 26)
    training_score = {}
    test_score = {}
    for n in neighbor_val:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(x_train, y_train)
        predicted_val = knn.predict(x_test)
        training_score.update({n: knn.score(x_train, y_train)})
        test_score.update({n: knn.score(x_test, y_test)})

    #print("Training scores:", training_score)
    #print("Test scores:", test_score)

    # Sorted test scores by values
    sorted_test_scores = sorted(test_score.items(), key=lambda x: x[1], reverse=True)
    print("Top score with k values:", sorted_test_scores[0])

data_processing()