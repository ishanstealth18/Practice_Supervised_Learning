import sys
import pandas as pd


def remove_columns(df, cols_list):
    """
    Remove columns from the dataframe
    :param df: pandas dataframe
    :param cols_list: list of columns to remove
    :return: None
    """
    df.drop(cols_list, axis=1, inplace=True)
    #print("After removing correlated columns:")
    #print(df.info())

    return df


def convert_date_column(df, date_col):
    """
    Convert date column to datetime and set days, months and years as separate columns
    :param df: pandas dataframe
    :param date_col: name of the date column
    :return: None
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['day'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['year'] = df[date_col].dt.year
    df.drop([date_col], axis=1, inplace=True)
    #print("After converting date column:")
    #print(df.info())

    return df


def convert_categorical_columns(df, categorical_cols):
    """
    Convert categorical columns into numeric type
    :param df: pandas dataframe
    :param categorical_cols: list of categorical columns to convert
    :return: None
    """
    for col in categorical_cols:
        dummy_col = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
        df = pd.concat([df, dummy_col], axis=1)
        df.drop([col], axis=1, inplace=True)

    df['flood_category'] = df['flood_category'].astype('category')

    #print("After converting categorical columns:")
    #print(df.info())

    return df


def main():
    # load csv file to df
    df = pd.read_csv('sri_lanka_flood_risk_modeled.csv')
    print(df.info())

    # remove highly correlated columns
    correlated_cols = ['rain_sum', 'rain_24h', 'rain_72h']
    df = remove_columns(df, correlated_cols)

    # convert date column to datetime and set days.months and years as separate columns
    df = convert_date_column(df, 'date')

    # remove unnecessary columns
    cols_to_remove = ['day', 'month', 'year', 'flood_risk_score', 'latitude', 'longitude']
    df = remove_columns(df, cols_to_remove)

    # convert categorical columns into numeric type
    categorical_cols = ['district', 'province', 'climatic_zone']
    df = convert_categorical_columns(df, categorical_cols)

    return df



if __name__ == "__main__":
    sys.exit(main())