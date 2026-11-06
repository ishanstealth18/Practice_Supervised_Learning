import numpy as np
import pandas as pd
import sys
import data_analysis
import seaborn as sns
from matplotlib import pyplot as plt


def one_hot_encode(df, column_name):
    # Perform one-hot encoding on the specified column
    one_hot_encoded_df = pd.get_dummies(df[column_name], prefix=column_name, dtype=int, drop_first=True)

    # Concatenate the one-hot encoded columns with the original DataFrame
    df = pd.concat([df, one_hot_encoded_df], axis=1)

    # Drop the original column from the DataFrame
    df = df.drop(column_name, axis=1)
    return df

def scaling(df, column_name):
    # Perform min-max scaling on the specified column
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    df[column_name] = (df[column_name] - min_value) / (max_value - min_value)
    return df

def split_data(df, target_column, test_size=0.2):
    from sklearn.model_selection import train_test_split

    # Split the DataFrame into features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def model(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

def main():
    # Load csv file
    df = data_analysis.main()

    #drop unnecessary columns
    df = df.drop(['Date dd/mm/yyyy', 'Day', 'Month'], axis=1)

    # do one hot encoding
    cols_to_encode = ['Industry Vertical', 'SubVertical', 'City', 'InvestmentnType', 'Investors Name']
    encoded_df = one_hot_encode(df, cols_to_encode)

    # check correlation
    correlation_matrix = encoded_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.1f', linewidths=0.5)
    plt.show()


    encoded_df['Year'] = encoded_df['Year'].fillna(encoded_df['Amount_USD'].median())
    encoded_df = one_hot_encode(encoded_df, 'Year')


    # log transform target variable
    encoded_df['Amount_USD'] = np.log1p(encoded_df['Amount_USD'])

    # split data
    X_tr, X_te, y_tr, y_te = split_data(encoded_df, target_column='Amount_USD', test_size=0.2)

    # model and accuracy
    model(X_tr, X_te, y_tr, y_te)

    return df


if __name__ == "__main__":
    sys.exit(main())