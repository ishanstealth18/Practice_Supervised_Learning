import pandas as pd
import sys
import sqlite3
from matplotlib import pyplot as plt
import seaborn as sns



def create_db(df):
    conn = sqlite3.connect('gujarat_mandi_db.db')
    df.to_sql(name='gujarat_mandi', con=conn, if_exists='replace', index=False)
    conn.close()


def drop_columns(df, column_name):
    df.drop(column_name, axis=1, inplace=True)
    # print('After dropping the columns')
    # print(df.info())


def check_null_values(df):
    #print(df.isna().sum())
    pass


def drop_null_values(df, col):
    df = df.dropna(subset=col)
    return df


def rename_commodities(df):
    veg_rename_mapping = {'Amla(Nelli Kai)': 'Amla', 'Bhindi(Ladies Finger)': 'Ladies Finger', 'Cucumbar(Kheera)':
        'Cucumbar', 'Elephant Yam(Suran)/Amorphophallus': 'Suran', 'Ginger(Green)': 'Ginger', 'Pointed gourd(Parval)':
                              'Parval', 'Rat Tail Radish(Mogari)': 'Mogari', 'Surat Beans(Papadi)': 'Papadi',
                          'Yam(Ratalu)': 'Ratalu', 'Long '
                                                   'Melon(Kakri)': 'Kakri', 'Pea Pod/Pea Cod/हरी मटर': 'Pea Pod',
                          'Ridgeguard(Tori)': 'Tori', 'Cowpea(Veg)': 'Cowpea',
                          'Little gourd(Kundru)': 'Kundru', 'Jack Fruit(Ripe)': 'Jack Fruit',
                          'Squash(Chappal Kadoo)': 'Squash',
                          'Spiny Gourd / Kartali(Kantola)\n': 'Kantola'}

    df['commodity'] = df['commodity'].replace(veg_rename_mapping)

    leafy_veg_mapping = {'Coriander(Leaves)': 'Coriander', 'Methi(Leaves)': 'Methi', 'Mint(Pudina)': 'Mint'}
    df['commodity'] = df['commodity'].replace(leafy_veg_mapping)

    pulses_rename_mapping = {'Green Gram(Moong)(Whole)': 'Green Gram', 'Black Gram(Urd Beans)(Whole)': 'Black Gram',
                             'Red gram/Arhar/Tur(whole)': 'Red Gram', 'Pegeon Pea(Arhar Fali)': 'Pegeon Pea',
                             'Bengal Gram(Gram)(Whole)': 'Bengal Gram',
                             'Cowpea(Lobia/Karamani)': 'Cowpea', 'Indian Beans(Seam)': 'Indian Beans',
                             'French Beans(Frasbean)': 'French Beans',
                             'Peas(Dry)': 'Peas', 'Gram Raw(Chholia)': 'Gram Raw', 'Kulthi(Horse Gram)': 'Kulthi',
                             'Kabuli Chana(Chickpeas-White)': 'Kabuli Chana',
                             'Red gram split/Arhar dal/Tur dal': 'Red gram split', 'Lentil(Masur)(Whole)': 'Masur',
                             'Green Gram Dal(Moong Dal)': 'Green Gram Dal',
                             'Black Gram Dal(Urd Dal)': 'Black Gram Dal', 'Chennangi(Whole)': 'Chennangi',
                             'Swan Phali(Flat Bean)': 'Swan Phali'}
    df['commodity'] = df['commodity'].replace(pulses_rename_mapping)

    seed_rename_mapping = {'Sesamum(Sesame,Gingelly,Til)': 'Sesamum', 'Soapnut(Antawala/Retha)': 'Soapnut',
                           'Cummin Seed(Jeera)': 'Cummin Seed',
                           'Suva(Dill Seed)': 'Suva', 'Guar Seed(Cluster Beans Seed)': 'Guar Seed',
                           'Sunflower/Sunflower Seed': 'Sunflower Seed'}
    df['commodity'] = df['commodity'].replace(seed_rename_mapping)

    crops_rename_mapping = {'Paddy(Common)': 'Paddy', 'Barley(Jau)': 'Barley', 'Groundnut pods(raw)': 'Groundnut Pods',
                            'Ragi(Finger Millet)': 'Raagi', 'Paddy(Basmati)': 'Paddy',
                            'Kodo Millet(Varagu)': 'Kodo Millet', 'Foxtail Millet(Navane)': 'Foxtail Millet',
                            'Chicory(Chikori/Kasni)': 'Chicory', 'Bajra(Pearl Millet/Cumbu)': 'Bajra',
                            'Jowar(Sorghum)': 'Jowar'}
    df['commodity'] = df['commodity'].replace(crops_rename_mapping)

    fruits_rename_mapping = {'Papaya(Raw)': 'Papaya', 'Ber(Zizyphus/Borehannu)': 'Ber', 'Chikoos(Sapota)': 'Chikoos',
                             'Banana - Green': 'Banana', 'Mango(Raw-Ripe)': 'Mango', 'Mousambi(Sweet Lime)': 'Mousambi',
                             'Karbuja(Musk Melon)': 'Karbuja',
                             'Pear(Marasebu)': 'Pear', 'Jamun(Narale Hannu)': 'Jamun', 'Galgal(Lemon)': 'Lemon',
                             'Custard Apple(Sharifa)': 'Custard Apple',
                             'Sweet Pumpkin': 'Pumpkin', 'White Pumpkin': 'Pumpkin'}
    df['commodity'] = df['commodity'].replace(fruits_rename_mapping)

    spices_rename_mapping = {'Ginger(Dry)': 'Ginger', 'Rose(Loose))': 'Rose', 'Turmeric(raw)': 'Turmeric'}
    df['commodity'] = df['commodity'].replace(spices_rename_mapping)

    return df


def add_commodity_category(df):
    # commodity values list

    leafy_veg_list = ['Coriander', 'Methi', 'Mint', 'Spinach', 'Leafy Vegetable', 'Season Leaves', 'Chow Chow']
    pulses_list = ['Green Gram', 'Black Gram', 'Red Gram', 'Pegeon Pea', 'Bengal Gram', 'Cowpea', 'Indian Beans',
                   'French Beans',
                   'Peas', 'Gram Raw', 'Kulthi', 'Kabuli Chana', 'Red gram split', 'Masur', 'Green Gram Dal',
                   'Black Gram Dal',
                   'Chennangi', 'Swan Phali', 'Moath Dal', 'Big Gram', 'Peas Wet', 'Soyabean', 'Beans', 'Cluster beans',
                   'Other Pulses',
                   'Green Peas', 'Mataki', 'Field Pea', 'Avare Dal', 'White Peas', 'Groundnut(Split)', 'Chennangi Dal',
                   'Masur Dal',
                   'Alasande Gram', 'Lak(Teora)']
    seeds_list = ['Sesamum', 'Castor Seed', 'Mustard', 'Corriander seed', 'Ground Nut Seed', 'Methi Seeds',
                  'Black pepper',
                  'Soapnut', 'Cotton Seed', 'Cummin Seed', 'Rajgir', 'Suva', 'Coconut Seed', 'Ambada Seed', 'Kinnow',
                  'Guar Seed', 'Sunflower Seed', 'Rayee', 'Sabu Dan', 'Tamarind Seed', 'Taramira', 'Asalia',
                  'Cashewnuts', 'Copra']
    crops_list = ['Cotton', 'Wheat', 'Groundnut', 'Paddy', 'Rice', 'Ajwan', 'Jute', 'Barley', 'Javi', 'Millets',
                  'Groundnut pods',
                  'Rubber', 'Ragi', 'Safflower', 'Beaten Rice', 'Kodo Millet', 'Foxtail Millet', 'Chicory',
                  'Sanai/Sunhemp',
                  'Tobacco', 'Bajra', 'Jowar']
    fruits_list = ['Papaya', 'Apple', 'Banana', 'Ber', 'Chikoos', 'Grapes', 'Guava', 'Pomegranate', 'Amaranthus',
                   'Pumpkin',
                   'Mango', 'Tamarind Fruit', 'Orange', 'Mousambi', 'Lime', 'Water Melon', 'Karbuja', 'Cherry',
                   'Sugarcane',
                   'Pineapple', 'Pear', 'Seetapal', 'Jamun', 'Lemon', 'Plum', 'Pumpkin', 'Peach', 'Litchi', 'Ramphal',
                   'Strawberry',
                   'Dates', 'Custard Apple']
    spices_list = ['Cardamom', 'Soanf', 'Dry Chillies', 'Turmeric', 'Raya', 'Chili Red', 'Tender Coconut',
                   'Lotus Sticks', 'Ginger', 'Rose', 'Turmeric']
    oil_list = ['Castor Oil', 'Ground Nut Oil', 'Sarasum', 'Coconut Oil', 'Mustard Oil']
    other_list = ['Gur(Jaggery)', 'Goat']

    def assign_commodity(item):
        if item in pulses_list:
            return 'pulses'
        elif item in seeds_list:
            return 'seeds'
        elif item in crops_list:
            return 'crops'
        elif item in seeds_list:
            return 'seeds'
        elif item in fruits_list:
            return 'fruits'
        elif item in spices_list:
            return 'spices'
        elif item in oil_list:
            return 'oil'
        elif item in leafy_veg_list:
            return 'leafy_vegetables'
        elif item in other_list:
            return 'other'
        else:
            return 'vegetables'

    df['commodity_category'] = df['commodity'].apply(assign_commodity)

    return df


def findings(df):
    # check how much metric tonne of commodity arrived per year => year 2025 has maximum commodity arrival
    arrival_per_year = df.groupby(['year'], as_index=False)['arrivals_mt'].sum()

    plt.bar(x=arrival_per_year['year'], height=arrival_per_year['arrivals_mt'])
    plt.xlabel('Year')
    plt.ylabel('Arrival (Metric Tonne)')
    plt.title('Arrival per year')
    plt.show()

    # check arrival of commodity(category) per year => Majority of time crops and vegetables have highest arrival
    commodity_arrival_per_year = df.groupby(['year', 'commodity_category'], as_index=False)['arrivals_mt'].sum()
    unique_year = sorted(commodity_arrival_per_year['year'].unique())
    fig, ax = plt.subplots(4, 4, figsize=(12, 9))
    ax = ax.flatten()

    for i, col in enumerate(unique_year):
        filter_year = commodity_arrival_per_year[commodity_arrival_per_year['year'] == col]
        ax[i].barh(width=filter_year['arrivals_mt'], y=filter_year['commodity_category'])
        ax[i].set_xlabel('Arrival (Metric Tonne)')
        ax[i].set_title(col)

    plt.suptitle('Arrival of commodity per year per commodity category')
    plt.tight_layout()
    plt.show()

    # check commodity category arrival per year
    commodity_category_per_year = pd.pivot_table(df, values='arrivals_mt', index='year', columns='commodity_category',
                                                 aggfunc='sum', fill_value=0)
    # print(commodity_category_per_district)

    commodity_category_per_year.plot(kind='bar', figsize=(12, 6), width=0.8)

    plt.title('Total Commodity Category Arrivals Per Year', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Arrivals (MT)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # check commodity category arrival per district => Most of the commodity arrives either in Ahmedabad, Bhavnagar or
    # Rajkot
    commodity_category_per_district = pd.pivot_table(df, values='arrivals_mt', index='district',
                                                     columns='commodity_category',
                                                     aggfunc='sum', fill_value=0)

    # print(commodity_category_per_district)

    commodity_category_per_district.plot(kind='bar', figsize=(12, 6), width=0.8)

    plt.title('Total Commodity Category Arrivals Per District', fontsize=14, fontweight='bold')
    plt.xlabel('District', fontsize=12)
    plt.ylabel('Arrivals (MT)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # commodity category average total minimum price per year => Minimum price for most of the commodities in 2025 was
    # very high
    # and 2023 year was second highest
    commodity_min_price = pd.pivot_table(data=df, index='commodity_category', columns='year',
                                         values='min_price_rs_quintal', aggfunc='mean', fill_value=0)
    # print(commodity_min_price)
    commodity_min_price.plot(kind='bar', figsize=(12, 6), width=0.8)

    plt.title('Total Commodity Category Minimum price Per Year', fontsize=14, fontweight='bold')
    plt.xlabel('Commodity Category', fontsize=12)
    plt.ylabel('Total Minimum Price Per Quintal', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # commodity group average max price for all years
    all_price_commodity_per_year = pd.pivot_table(data=df, index='commodity_category', columns='year',
                                                  values=['max_price_rs_quintal'], aggfunc='mean', fill_value=0)
    # print(all_price_commodity_per_year)
    all_price_commodity_per_year.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title('Commodity Category Max price Per Year', fontsize=14, fontweight='bold')
    plt.xlabel('Commodity Category', fontsize=12)
    plt.ylabel('Max Price Per Quintal', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # commodity group average modal price for all years
    all_price_commodity_per_year = pd.pivot_table(data=df, index='commodity_category', columns='year',
                                                  values=['modal_price_rs_quintal'], aggfunc='mean', fill_value=0)
    # print(all_price_commodity_per_year)
    all_price_commodity_per_year.plot(kind='bar', figsize=(12, 6), width=0.8)
    plt.title('Commodity Category Modal price Per Year', fontsize=14, fontweight='bold')
    plt.xlabel('Commodity Category', fontsize=12)
    plt.ylabel('Modal Price Per Quintal', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

    # check relation between commodity group arrival and its modal price => Seeds and spices are the top commodity
    # category sold at highest Modal price. Vegetables are the commodity sold at least modal price. Crops are the
    # commodity which arrives in large quantity but sold at less modal price
    commodity_arrival_modal_price_relation = df.groupby(['commodity_category', 'year'], as_index=False)[['arrivals_mt',
                                                        'modal_price_rs_quintal']].mean()
    #print(commodity_arrival_modal_price_relation)

    plt.figure(figsize=(12, 7))

    # Plot all years together in one chart
    sns.scatterplot(
        data=commodity_arrival_modal_price_relation,
        x='year',  # Continuous timeline tracking on X
        y='modal_price_rs_quintal',  # Price on Y
        hue='commodity_category',  # Color-coded by commodity grouping
        size='arrivals_mt',  # Point thickness updates by arrival metric volume
        sizes=(40, 400),  # Keeps sizes distinct but readable
        alpha=0.8,
        palette='tab10'
    )

    # Custom formatting for readability
    plt.xlabel('Year', fontsize=12, labelpad=10)
    plt.ylabel('Avg Modal Price (Rs/Quintal)', fontsize=12, labelpad=10)
    plt.title('Commodity Price vs. Volume Arrival Trends Across Years', fontsize=14, fontweight='bold', pad=15)

    # Force the X-axis to display clean, whole-number year tick marks
    plt.xticks(commodity_arrival_modal_price_relation['year'].unique(), rotation=45)

    # Place the unified legend cleanly on the right side of the graph
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # check market per district with maximum arrival from 2011-2026 => Seeds arrives maximum in Lakhani APMC. Gondal
    # APMC has Fruits and Vegetables coming in large quantity

    idx = df.groupby(['commodity_category'])['arrivals_mt'].idxmax()

    # 2. Extract those exact rows from the original DataFrame, including 'market'
    market_per_district_max_arrival = df.loc[idx, ['commodity_category', 'market', 'arrivals_mt']]

    # 3. Optional: Reset index to keep the output clean
    market_per_district_max_arrival = market_per_district_max_arrival.reset_index(drop=True)

    #print(market_per_district_max_arrival)
    sns.barplot(data=market_per_district_max_arrival, x='commodity_category', y='arrivals_mt', hue='market')
    plt.xlabel('Commodity Category')
    plt.ylabel('Arrivals in mt')
    plt.title('Max Arrival per commodity in Market')
    plt.show()


def initial_analysis(df):
    # drop unnecessary columns
    drop_columns(df, ['variety', 'state_id', 'district_id', 'market_id', 'commodity_id', 'variety_id', 'state'])

    # check null values
    check_null_values(df)

    # drop NaN records
    df = drop_null_values(df, ['date', 'year', 'max_price_rs_quintal'])

    # convert dtype
    df['date'] = pd.to_datetime((df['date']))
    df['year'] = df['year'].astype(int)

    # rename commodities for simplicity
    df = rename_commodities(df)

    # add commodity category column
    df = add_commodity_category(df)

    # findings
    #findings(df)

    return df


def main():
    # load dataset
    df = pd.read_csv('bharat_mandi_gujarat_2011_2026.csv')

    # create db file
    # create_db(df)

    # initial analysis
    df = initial_analysis(df)

    return df


if __name__ == "__main__":
    sys.exit(main())
