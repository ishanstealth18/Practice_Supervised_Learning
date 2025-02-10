import pandas as pd
import numpy as np
from matplotlib import pyplot


def check_null_data(src_file):
    #print(src_file.isna().sum().sort_values())
    pass


def process_outlier(src_file, column_name):
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
    boxplot_df = input_data.drop(["date", "weather"], axis=1)
    print(boxplot_df.info())
    column_names = boxplot_df.columns.values
    for x in column_names:
        pyplot.boxplot(boxplot_df[x])
        pyplot.title(x)
        pyplot.show()


def data_processing():
    input_src_file = pd.read_csv("seattle-weather.csv")
    print(input_src_file.shape)
    check_null_data(input_src_file)
    #make_boxplot(input_src_file)
    # detect and process outlier
    min_precipitation_val, max_precipitation_val = process_outlier(input_src_file, "precipitation")
    precipitation_index = input_src_file[(input_src_file["precipitation"] < min_precipitation_val) |
                                         (input_src_file["precipitation"] > max_precipitation_val)].index
    input_src_file.drop(precipitation_index, inplace=True)
    print(input_src_file.shape)
    min_wind_val, max_wind_val = process_outlier(input_src_file, "wind")
    wind_index = input_src_file[(input_src_file["wind"] < min_wind_val) |
                                         (input_src_file["wind"] > max_wind_val)].index
    input_src_file.drop(wind_index, inplace=True)
    print(input_src_file.shape)
    #make_boxplot(input_src_file)

data_processing()




