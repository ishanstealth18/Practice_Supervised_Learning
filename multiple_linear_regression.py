import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def process_outlier(src_file, column_name):
    # outlier for experience
    first_quartile = src_file[column_name].quantile(0.25)
    third_quartile = src_file[column_name].quantile(0.75)
    print("25%: {} ", "75%: {}", first_quartile, third_quartile)
    iqr = third_quartile - first_quartile
    min_threshold = 0.0
    max_threshold = 0.0
    if first_quartile > 0:
        min_threshold = first_quartile - (1.5 * iqr)
    if third_quartile > 0:
        max_threshold = third_quartile + (1.5 * iqr)

    print("Min threshold: {} ", "Max threshold: {}", min_threshold, max_threshold)
    outlier_count = src_file.loc[(src_file[column_name] > max_threshold)].count()
    print("Outlier count:", outlier_count)

    return min_threshold, max_threshold


def make_boxplot(input_data):

    column_count = len(input_data.count())
    column_names = input_data.columns
    print(column_count)

    for x in range(len(column_names)):
        pyplot.subplot(1, column_count, x+1)
        pyplot.boxplot(input_data[column_names[x]])
        pyplot.title(column_names[x])
        pyplot.plot()

    pyplot.show()


def split_data(x_val, y_val):
    X_train, X_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=42)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    return X_train, X_test, y_train, y_test


def normalize_data(input_file):
    scaler = StandardScaler()
    # fit = calculate mean and std of each feature column, transform = apply that mean and std to each value of feature
    # column

    column_name = input_file.columns
    for c in column_name:
        input_file[[c]] = scaler.fit_transform(input_file[[c]])

    #print(input_file.head(10))
    return input_file


def linear_regression_model(x_trained_data, x_test_data, y_trained_data, y_test_data):
    reg_model = LinearRegression()
    # .fit() will calculate co efficients
    reg_model.fit(x_trained_data, y_trained_data)
    # score for trained data model
    r_sq = reg_model.score(x_trained_data, y_trained_data)
    print("coeff:", r_sq)
    print("intercept: ", reg_model.intercept_)
    print("slope: ", reg_model.coef_)
    x_pred = reg_model.predict(x_test_data)
    print("Predicted test data:", x_pred)
    print("Actual test data:", y_test_data)
    pyplot.scatter(x_pred, y_test_data)
    pyplot.xlabel("Predicted values")
    pyplot.ylabel("Actual values")
    pyplot.plot([min(y_test_data), max(y_test_data)], [min(y_test_data), max(y_test_data)], color='red', linewidth=2)
    pyplot.plot([min(x_pred), max(x_pred)], [min(x_pred), max(x_pred)], color='green', linewidth=2)
    pyplot.show()


def data_processing():
    input_src_file = pd.read_csv("multiple_linear_regression_dataset.csv")
    print(input_src_file.shape)
    make_boxplot(input_src_file)
    # detect and process outlier
    # for 'experience' column
    min_experience_val, max_experience_val = process_outlier(input_src_file, "experience")
    experience_index = input_src_file[(input_src_file["experience"] < min_experience_val) |
                                         (input_src_file["experience"] > max_experience_val)].index
    print(experience_index)
    input_src_file.drop(index=experience_index, inplace=True)
    print(input_src_file.shape)

    # for 'income' column
    min_income_val, max_income_val = process_outlier(input_src_file, "income")
    income_index = input_src_file[(input_src_file["income"] < min_income_val) |
                                      (input_src_file["income"] > max_income_val)].index
    print(income_index)
    input_src_file.drop(index=income_index, inplace=True)
    print(input_src_file.shape)

    # normalize data
    normalized_val = normalize_data(input_src_file)

    # split the drizzle data
    x = normalized_val.drop(['income'], axis=1).values
    y = normalized_val['income'].values

    X_train_val, X_test_val, y_train_val, y_test_val = split_data(x, y)

    linear_regression_model(X_train_val, X_test_val, y_train_val, y_test_val)


data_processing()




