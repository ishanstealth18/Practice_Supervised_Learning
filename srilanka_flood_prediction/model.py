import sys
import data_cleaning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def split_data(df):
    """
    Split the dataframe into features and target
    :param df: pandas dataframe
    :return: X, y
    """
    X = df.drop(['flood_category'], axis=1)
    y = df['flood_category']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_tr, X_te, y_tr, y_te


def standardize_features(X_train, X_test, cols):
    """
    Standardize features to make them on equal scale
    :param X_train: training features
    :param X_test: testing features
    :param cols: list of columns to standardize
    :return: standardized training and testing features
    """
    scaler = StandardScaler()
    X_train[cols] = scaler.fit_transform(X_train[cols])
    X_test[cols] = scaler.transform(X_test[cols])
    return X_train, X_test, scaler


def create_model(X_train, y_train):
    base_model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
    base_model.fit(X_train, y_train)

    return base_model


def predict(base_model, X_test):
    """
    Predict the target variable using the trained model
    :param base_model: trained model
    :param X_test: testing features
    :return: predicted values
    """
    y_pred = base_model.predict(X_test)
    return y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate the model using accuracy score, classification report and confusion matrix
    :param y_test: true values
    :param y_pred: predicted values
    :return: None
    """
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


def main():
    df = data_cleaning.main()
    print(df.info())

    # split the dataframe into features and target
    X_train, X_test, y_train, y_test = split_data(df)

    # standardize features to make them on equal scale
    standardize_cols = ['precipitation_sum', 'soil_moisture_0_to_7cm_mean', 'soil_moisture_7_to_28cm_mean',
                        'temperature_2m_max', 'wind_speed_10m_max', 'rain_48h',
                        'soil_saturation_index'
                        ]

    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test, standardize_cols)

    # develop model
    base_model = create_model(X_train_scaled, y_train)

    # predict
    base_pred = predict(base_model, X_test_scaled)

    # Accuracy score
    evaluate_model(y_test, base_pred)

    # save model
    joblib.dump(base_model, 'flood_risk_model.pkl')
    joblib.dump(scaler, 'scaled_data.pkl')




if __name__ == '__main__':
    sys.exit(main())