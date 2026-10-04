import pandas as pd
import sys
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def show_historgram(df, column_name_list):
    fig, ax = plt.subplots(4,4, figsize=(13, 8))
    ax = ax.flatten()
    for i, column in enumerate(column_name_list):
        if column != 'latitude' and column != 'longitude' and column != 'year':
            ax[i].hist(df[column], bins=100, edgecolor='black')
            ax[i].set_title(f'Histogram of {column}')
            ax[i].set_xlabel(column)
            ax[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.show()


def check_outliers(df, column_name_list):
    fig, ax = plt.subplots(4, 4, figsize=(13, 8))
    ax = ax.flatten()
    for i, column in enumerate(column_name_list):
        if column != 'latitude' and column != 'longitude' and column != 'year':
            ax[i].boxplot(df[column])
            ax[i].set_title(f'Boxplot of {column}')
            ax[i].set_xlabel(column)
            ax[i].set_ylabel('Values')
    plt.tight_layout()
    plt.show()

    for c in column_name_list:
        if c != 'latitude' and c != 'longitude' and c != 'year':
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[c] < lower_bound) | (df[c] > upper_bound)]
            print(f'Column: {c}, Outliers: {len(outliers)}')


def check_geographical_relations(df, geographical_features):
    # From the chart below we see that Ratnapura district has highest flood risk score, Sabarangamuwa province has
    # highest flood risk score and Wet Zone has highest flood risk score.

    fig,ax = plt.subplots(3, 1, figsize=(15, 8))
    ax = ax.flatten()
    for i, column in enumerate(geographical_features):
        sns.barplot(x=column, y='flood_risk_score', data=df, ax=ax[i])
        ax[i].set_title(f'Flood Risk Distribution by {column}')
        ax[i].set_xlabel(column)
        ax[i].set_ylabel('Flood Risk Score')
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def check_atmospheric_relations(df, atmospheric_features):
    # From the chart below it is heavily sensitive to rainfall volume (which has a clear threshold-based ceiling
    # effects), moderately influenced by temperature (acting within a specific warm window), and mostly independent of
    # wind speed (which acts more like background noise or an indirect marker of storms).

    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    ax = ax.flatten()
    for i, column in enumerate(atmospheric_features):
        sns.scatterplot(x=column, y='flood_risk_score', data=df, ax=ax[i])
        ax[i].set_title(f'Flood Risk Distribution by {column}')
        ax[i].set_xlabel(column)
        ax[i].set_ylabel('Flood Risk Score')

    plt.tight_layout()
    plt.show()


def check_hydrological_relations(df, hydrological_features):
    # While rainfall creates a predictable logarithmic risk curve, soil saturation behaves like a switch: below 0.3 it
    # suppresses risk, but above 0.3 it opens the door for catastrophic, high-scoring flood events.

    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    ax = ax.flatten()
    for i, column in enumerate(hydrological_features):
        sns.scatterplot(x=column, y='flood_risk_score', data=df, ax=ax[i])
        ax[i].set_title(f'Flood Risk Distribution by {column}')
        ax[i].set_xlabel(column)
        ax[i].set_ylabel('Flood Risk Score')

    plt.tight_layout()
    plt.show()


def correlation_analysis(df, correlation_col):
    cor = df[correlation_col].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cor, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

    upper_tri = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
    top_correlations = (upper_tri.unstack()
                        .dropna()
                        .reset_index())

    top_correlations.columns = ['Column 1', 'Column 2', 'Correlation']
    top_correlations['Abs_Correlation'] = top_correlations['Correlation'].abs()
    top_10 = top_correlations.sort_values(by='Abs_Correlation', ascending=False).head(10)
    print(top_10[['Column 1', 'Column 2', 'Correlation']])

    # drop highly correlated columns
    df.drop(['rain_sum', 'rain_24h', 'rain_72h'], axis=1, inplace=True)

    return df






def data_cleaning_analysis(df):

    # rename columns for better readability
    df.rename({'soil_moisture_0_to_7cm_mean': 'soil_moisture_0_7', 'soil_moisture_7_to_28cm_mean': 'soil_moisture_7_28',
               'soil_saturation_index': 'soil_saturation'}, axis=1, inplace=True)

    # check early distributions
    numeric_features = list(df.select_dtypes(include=['int64', 'float64']).columns)
    print(numeric_features)

    # check initial distributions
    show_historgram(df, numeric_features)

    # check outliers => outliers are almost 12% of total records. We cannot remove them as they are important for flood
    # risk analysis. We will keep them for now.
    check_outliers(df, numeric_features)

    # check relation between Geographical Identifiers vs flood_risk
    geographical_features = ['district', 'province', 'climatic_zone']
    check_geographical_relations(df, geographical_features)

    # check relation between Atmospheric Observations vs flood_risk
    atmospheric_features = ['precipitation_sum', 'rain_sum', 'temperature_2m_max', 'wind_speed_10m_max']
    check_atmospheric_relations(df, atmospheric_features)

    # check hydrological features vs flood_risk
    hydrological_features = ['rain_24h', 'rain_48h', 'rain_72h', 'soil_saturation']
    check_hydrological_relations(df, hydrological_features)

    # check correlation between features and flood_risk_score
    correlation_col = atmospheric_features + hydrological_features
    correlation_analysis(df, correlation_col)


def main():
    # load csv file to df
    df = pd.read_csv('sri_lanka_flood_risk_modeled.csv')
    print(df.info())

    # clean data
    data_cleaning_analysis(df)


if __name__ == "__main__":
    sys.exit(main())