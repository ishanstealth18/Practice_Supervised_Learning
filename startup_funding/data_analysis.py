import pandas as pd
import sys
import sqlite3

from matplotlib import pyplot as plt


def create_db(df):
    # Create a SQLite database
    conn = sqlite3.connect('startup_fund.db')

    df.to_sql('startup_funding', conn, if_exists='replace', index=False)
    conn.close()


def check_null_values(df):
    # check null values
    #print(df.isna().sum())

    # we see that column 'Remarks' has2625 null values out of total3044 records which is huge, lets check what unique
    # values are present in this column
    #print(df['Remarks'].unique())
    # we do not get much values from 'Remarks' column, so we can drop this column from our dataset

    # check each columns and decide which columns to drop based on null values and unique values or we can impute
    # values based on existing values

    # industry vertical => we can use 'Mode' to impute values as this is a categorical column
    #print(df['Industry Vertical'].value_counts())
    df.fillna({'Industry Vertical': df['Industry Vertical'].mode()[0]}, inplace=True)

    # subvertical => we can use 'Mode' to impute values as this is a categorical column
    #print(df['SubVertical'].value_counts())
    df.fillna({'SubVertical': df['SubVertical'].mode()[0]}, inplace=True)

    # city location
    # rename column 'City Location' to 'City'
    df.rename(columns={'City  Location': 'City'}, inplace=True)
    #print(df['City'].value_counts())
    df.fillna({'City': df['City'].mode()[0]}, inplace=True)

    # InvestmentnType
    df.fillna({'InvestmentnType': df['InvestmentnType'].mode()[0]}, inplace=True)

    # Amount in USD
    df.rename(columns={'Amount in USD': 'Amount_USD'}, inplace=True)
    df['Amount_USD'] = df['Amount_USD'].str.replace(',', '')
    df['Amount_USD'] = pd.to_numeric(df['Amount_USD'], errors='coerce')
    df.fillna({'Amount_USD': df['Amount_USD'].mean()}, inplace=True)
    df['Amount_USD'] = df['Amount_USD'].astype(int)

    # Investors Name
    df.fillna({'Investors Name': df['Investors Name'].mode()[0]}, inplace=True)

    return df


def drop_columns(df, col_list):
    # drop unnecessary columns
    df.drop(columns=col_list, inplace=True)

    return df


def check_verticals(df):
    # reduce vertical categories to top 5 verticals and rest to 'Other'
    top_verticals = df['Industry Vertical'].value_counts().nlargest(5).index
    df['Industry Vertical'] = df['Industry Vertical'].apply(lambda x: x if x in top_verticals else 'Other')

    # apply same to sub verticals
    top_sub_verticals = df['SubVertical'].value_counts().nlargest(5).index
    df['SubVertical'] = df['SubVertical'].apply(lambda x: x if x in top_sub_verticals else 'Other')
    #print(df['Industry Vertical'].value_counts())
    #print(df['SubVertical'].value_counts())

    # apply same to cities
    df['City'] = df['City'].replace({'Bengaluru': 'Bangalore', 'New Delhi': 'Delhi', 'Gurgaon': 'Gurugram', 'Noida':
        'Delhi'})
    top_cities = df['City'].value_counts().nlargest(10).index
    df['City'] = df['City'].apply(lambda x: x if x in top_cities else 'Other')

    # Investment Type
    df['InvestmentnType'] = df['InvestmentnType'].str.replace('\\n', '')
    df['InvestmentnType'] = df['InvestmentnType'].replace({'Seed\Funding': 'Seed Funding', 'Seed/ Angel Funding':
        'Seed Funding', 'Seed / Angel Funding': 'Seed Funding', 'Seed/Angel Funding': 'Seed Funding'})
    top_investment_types = df['InvestmentnType'].value_counts().nlargest(5).index
    df['InvestmentnType'] = df['InvestmentnType'].apply(lambda x: x if x in top_investment_types else 'Other')

    # drop rows where Amount_USD is 0 or negative
    negative_amount_index = df[df['Amount_USD'] <= 0].index
    df.drop(negative_amount_index, inplace=True)

    # Investment Name
    df['Investors Name'] = df['Investors Name'].replace({'undisclosed investors': 'Undisclosed Investors',
                                                         'Undisclosed investors': 'Undisclosed Investors',
                                                         'Undisclosed Investor': 'Undisclosed Investors',
                                                         'Undisclosed': 'Undisclosed Investors'})

    top_investment_name = df['Investors Name'].value_counts().nlargest(5).index
    df['Investors Name'] = df['Investors Name'].apply(lambda x: x if x in top_investment_name else 'Other')

    return df


def convert_dates(df):
    # convert dates
    df['Date dd/mm/yyyy'] = pd.to_datetime(df['Date dd/mm/yyyy'], format='%d/%m/%Y', errors='coerce')
    df['Year'] = df['Date dd/mm/yyyy'].dt.year.astype('Int64')
    df['Month'] = df['Date dd/mm/yyyy'].dt.month
    df['Day'] = df['Date dd/mm/yyyy'].dt.day

    return df


def do_analysis(df):
    # check funding per Industrial verticals
    fund_per_verticals = df.groupby(['Industry Vertical', 'SubVertical'])['Amount_USD'].sum()
    #print(fund_per_verticals)

    fund_per_verticals.unstack().plot(kind='barh')
    plt.xlabel('Amount in USD')
    plt.ylabel('Industry Vertical')
    plt.title('Funding per Industrial Verticals')
    plt.show()

    # city vs industry verticals v/s funding
    funding_per_city = df.groupby(['City', 'Industry Vertical'])['Amount_USD'].sum()
    funding_per_city.unstack().plot(kind='barh')
    plt.xlabel('Amount in USD')
    plt.ylabel('City')
    plt.title('Funding per City')
    plt.show()

    # investment type vs industrial verticals v/s funding
    funding_per_investment_type = df.groupby(['InvestmentnType', 'Industry Vertical'])['Amount_USD'].sum()
    funding_per_investment_type.unstack().plot(kind='barh')
    plt.xlabel('Amount in USD')
    plt.ylabel('Investment Type')
    plt.title('Funding per Invest Type')
    plt.show()

    # funding per year
    funding_per_year = df.groupby(['Year', 'Industry Vertical'])['Amount_USD'].sum()
    funding_per_year.unstack().plot(kind='barh')
    plt.xlabel('Amount in USD')
    plt.ylabel('Year')
    plt.title('Funding per Year')
    plt.show()

    #investment name vs funding v/s industrial verticals
    funding_per_investment_name = df.groupby(['Investors Name', 'Industry Vertical'])['Amount_USD'].sum()
    funding_per_investment_name.unstack().plot(kind='barh')
    plt.xlabel('Amount in USD')
    plt.ylabel('Investors Name')
    plt.title('Funding per Investors Name')
    plt.show()


def main():
    # Load csv file
    df = pd.read_csv('startup_funding.csv')

    # create db file
    #create_db(df)

    # check null values
    df_after_removing_null = check_null_values(df)

    # drop unnecessary columns
    remove_cols = ['Remarks', 'Sr No', 'Startup Name']
    df_after_drop = drop_columns(df_after_removing_null, remove_cols)

    # check verticals and sub verticals, cities, investment type
    df_after_verticals = check_verticals(df_after_drop)

    # convert dates
    df_after_date_conversion= convert_dates(df_after_verticals)
    # do some analysis
    #do_analysis(df_after_date_conversion)

    return df_after_date_conversion


if __name__ == "__main__":
    sys.exit(main())
