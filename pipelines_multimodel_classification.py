import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score, precision_score


# remove unwanted columns
def remove_columns(input_file):
    input_file.drop(['Cable ID', 'Timestamp', 'Energy Consumption (W)', 'Edge Device Used', 'Processing Speed (ms)'],
                    axis=1, inplace=True)
    print("After removing columns:", input_file.info())


# check null values for data pre processing
def check_null(input_file):
    null_count = input_file.isnull().sum()
    print("Null count:", null_count)


# visualize data in form of boxplot to understand the distribution
def visualize_data(input_file):
    column_name = list(input_file.columns)
    print("Columns name:", column_name)
    column_name.remove('Cable State')
    total_columns = len(column_name)

    for name in range(total_columns):
        plt.subplot(1, total_columns, name+1)
        plt.boxplot(input_file[column_name[name]])
        plt.title(column_name[name])

    plt.show()


# normalize values for independent variables (x) or feature variables
def normalize_data(x_train, y_train, x_test):
    scaler = StandardScaler()
    x_train_scalar = scaler.fit_transform(x_train)
    x_test_scalar = scaler.transform(x_test)

    return x_train_scalar, x_test_scalar


# Train the model on training set and apply on test data. I have taken range of K values.
def knn_training_model(x_train, y_train, x_test, y_test):
    neighbor_range = np.arange(1, 3)
    test_score_range = {}
    training_score_range = {}
    for n in neighbor_range:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(x_train, y_train)
        training_score_range.update({n: knn.score(x_train, y_train)})
        pred_val = knn.predict(x_test)
        cm = confusion_matrix(pred_val, y_test, labels=knn.classes_)
        confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
        confusion_matrix_display.plot()
        plt.show()
        print("Accuracy {} with k value {}:".format(accuracy_score(pred_val, y_test), n))
        print("Precision {} with k value {}:".format(precision_score(pred_val, y_test, average='weighted'), n))

    top_training_score = sorted(training_score_range.items(), key=lambda x: x[1], reverse=True)
    print("Top training score: ", top_training_score[0])


# main function
def data_processing():
    input_src_file = pd.read_csv("/kaggle/input/cable-multi-state-monitoring-system-dataset/cable_monitoring_dataset.csv")
    print(input_src_file.info())

    # remove unwanted columns
    remove_columns(input_src_file)

    # check for null values and remove it
    check_null(input_src_file)

    # visualize data
    visualize_data(input_src_file)

    # divide the data
    X = input_src_file.drop(['Cable State'], axis=1).values
    y = input_src_file['Cable State'].values

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # normalize data
    X_train, X_test = normalize_data(X_train, y_train, X_test)

    # knn classifier
    training_score = knn_training_model(X_train, y_train, X_test, y_test)


data_processing()