import argparse
import os
import numpy as np
import pandas as pd

def preprocess(data_dir, verbose=False):
    if verbose:
        print("Reading csvs...")
    train_users = pd.read_csv(os.path.join(data_dir, "train_users.csv"))
    test_users = pd.read_csv(os.path.join(data_dir, "test_users.csv"))
    sessions = pd.read_csv(os.path.join(data_dir, "sessions.csv"))
    age_bkts = pd.read_csv(os.path.join(data_dir, "age_gender_bkts.csv")) 
    countries = pd.read_csv(os.path.join(data_dir, "countries.csv")) 
    labels, country_destinations = pd.factorize(train_users['country_destination'])
    np.save(os.path.join(data_dir, "train_labels"), labels)
    np.save(os.path.join(data_dir, "country_labels"), np.array(country_destinations.tolist()))   
    train_users.drop('country_destination', axis=1, inplace=True)
    n_train = len(train_users)

    full_users = pd.concat([train_users, test_users], axis=0)

    if verbose:
        print(full_users.shape)
        print("Adding time features...")

    # Add time features
    year = full_users['timestamp_first_active'].astype(str).str[:4]
    month = full_users['timestamp_first_active'].astype(str).str[4:6]
    day = full_users['timestamp_first_active'].astype(str).str[6:8]
    full_users['timestamp_first_active'] = year + "-" +  month + "-" + day
    full_users['timestamp_first_active'] = pd.to_datetime(full_users['timestamp_first_active'])

    full_users['timestamp_first_active_year'] = full_users['timestamp_first_active'].dt.year
    full_users['timestamp_first_active_month'] = full_users['timestamp_first_active'].dt.month
    full_users['timestamp_first_active_day'] = full_users['timestamp_first_active'].dt.day
    
    full_users['date_account_created'] = pd.to_datetime(full_users['date_account_created'])

    full_users['date_account_created_year'] = full_users['date_account_created'].dt.year
    full_users['date_account_created_month'] = full_users['date_account_created'].dt.month
    full_users['date_account_created_day'] = full_users['date_account_created'].dt.day

    full_users['date_first_booking'] = pd.to_datetime(full_users['date_first_booking'])

    full_users['date_first_booking_year'] = full_users['date_first_booking'].dt.year
    full_users['date_first_booking_month'] = full_users['date_first_booking'].dt.month
    full_users['date_first_booking_day'] = full_users['date_first_booking'].dt.day
    full_users['date_first_booking_year'].fillna(-1, inplace=True)
    full_users['date_first_booking_month'].fillna(-1, inplace=True)
    full_users['date_first_booking_day'].fillna(-1, inplace=True)
    
    full_users['account_created_first_active_elapsed'] = (full_users['date_account_created'] - full_users['timestamp_first_active']).dt.days
    full_users['first_booking_first_active_elapsed'] = (full_users['date_first_booking'] - full_users['timestamp_first_active']).dt.days
    full_users['first_booking_first_active_elapsed'].fillna(-1, inplace=True)
    full_users['first_booking_account_created_elapsed'] = (full_users['date_first_booking'] - full_users['date_account_created']).dt.days
    full_users['first_booking_account_created_elapsed'].fillna(-1, inplace=True)
    full_users.drop(columns=['date_account_created', 'date_first_booking', 'timestamp_first_active'], inplace=True)

    # Join countries and age buckets
    if verbose:
        print("Joining countries and age buckets...")

    min_year, max_year = 1915, 2015
    full_users['age'].where(full_users['age'] < min_year, max_year - full_users['age'], inplace=True)
    full_users.loc[full_users['age'] < 0, 'age'] = 0
    mean = full_users['age'].mean()
    std = full_users['age'].std()
    n_missing = full_users['age'].isnull().sum()
    full_users.loc[full_users['age'].isnull(), 'age'] = np.random.normal(loc=mean, scale=std, size=n_missing).astype(int)
    full_users.loc[full_users['age'] < 0, 'age'] = 0
    bkt_labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90-94", "95-99", "100+"]
    bins = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99, full_users['age'].max()]
    full_users['age_bucket'] = pd.cut(full_users['age'], bins=bins, labels=bkt_labels, right=True)    
 
    country_age_buckets = pd.merge(countries, age_bkts, on='country_destination', how='left')
    country_age_buckets['gender'] = country_age_buckets['gender'].str.upper()
    country_age_buckets.drop(columns=['lat_destination', 'lng_destination', 'year'], inplace=True)
    country_age_buckets['language'] = country_age_buckets['destination_language '].str[:2]
    country_age_buckets = country_age_buckets.groupby(by=["age_bucket", "gender", "language"]).sum()
    country_age_buckets['distance_km'].fillna(-1, inplace=True)
    country_age_buckets['destination_km2'].fillna(-1, inplace=True) 
    country_age_buckets['language_levenshtein_distance'].fillna(-1, inplace=True)
    country_age_buckets['population_in_thousands'].fillna(-1, inplace=True)
    country_age_buckets.reset_index(inplace=True)
    full_users = pd.merge(full_users, country_age_buckets, on=['age_bucket', 'language', 'gender'], how='left')
 
    if verbose:
        print("Joining sessions...")

    # Join sessions table
    total_secs_elapsed = pd.DataFrame(data=sessions.groupby('user_id')['secs_elapsed'].sum())
    total_secs_elapsed.reset_index(inplace=True)
    full_users = pd.merge(full_users, total_secs_elapsed, left_on='id', right_on='user_id', how='left')
    full_users.drop(columns='user_id', inplace=True)
    session_data = []
    session_cols = sessions.columns.tolist()
    session_cols.remove('user_id')
    session_cols.remove('secs_elapsed')
    for col in session_cols:
        sessions.loc[sessions[col].isnull(), col] = 'none'
        crosstab = pd.crosstab(sessions['user_id'], sessions[col])
        crosstab.columns = ["{}_{}".format(col, c) for c in crosstab.columns.tolist()]
        session_data.append(crosstab)
    session_data = pd.concat(session_data, axis=1)
    session_data.reset_index(inplace=True)
    full_users = pd.merge(full_users, session_data, left_on='id', right_on='user_id', how='left')
    full_users.drop(columns='user_id', inplace=True)

    if verbose:
        print("Convert categorical...")

    categories = ['age_bucket', 'gender', 'signup_method', 'signup_flow', 'language', 
                  'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
                  'signup_app', 'first_device_type', 'first_browser']
    for col in categories:
        full_users[col] = full_users[col].astype('category')
    full_users_lgb = full_users.copy()
    categorical_columns = full_users_lgb.select_dtypes(['category']).columns
    full_users_lgb[categorical_columns] = full_users_lgb[categorical_columns].apply(lambda x: x.cat.codes)
    full_users_lgb[categorical_columns] = full_users_lgb[categorical_columns].where(full_users_lgb[categorical_columns] >=0, 999)
    full_users = pd.get_dummies(full_users, columns=categories, prefix=categories)
    full_users.columns = full_users.columns.str.replace('\s+', '-')
    full_users_lgb.columns = full_users_lgb.columns.str.replace('\s+', '-')
    if verbose:
        print("One-hot encoding")
        print(full_users.shape) 
        print("Numeric category")
        print(full_users_lgb.shape)
        print("Impute missing...")

    full_users.fillna(-1, inplace=True)
    full_users_lgb.fillna(-1, inplace=True)

    ids = full_users['id']
    train_ids = ids[:n_train]
    test_ids = ids[n_train:]
    full_users.drop(columns='id', inplace=True)
    full_users_lgb.drop(columns='id', inplace=True)

    train_users = full_users[:n_train]
    test_users = full_users[n_train:]
    train_users.reset_index(drop=True, inplace=True)
    test_users.reset_index(drop=True, inplace=True)

    train_users_lgb = full_users_lgb[:n_train]
    test_users_lgb = full_users_lgb[n_train:]
    train_users_lgb.reset_index(drop=True, inplace=True)
    test_users_lgb.reset_index(drop=True, inplace=True)   
 
    train_ids.to_csv(os.path.join(data_dir, "train_ids.csv"), header=['id'], index=False) 
    test_ids.to_csv(os.path.join(data_dir, "test_ids.csv"), header=['id'], index=False)

    train_users.to_feather(os.path.join(data_dir, "train.feather"))
    test_users.to_feather(os.path.join(data_dir, "test.feather"))

    train_users_lgb.to_feather(os.path.join(data_dir, "train_lgb.feather"))
    test_users_lgb.to_feather(os.path.join(data_dir, "test_lgb.feather"))

    if verbose:
        print("One-hot encoding")
        print("train: {}".format(train_users.shape)) 
        print("test: {}".format(test_users.shape)) 
        print("Numeric category")
        print("train: {}".format(train_users_lgb.shape)) 
        print("test: {}".format(test_users_lgb.shape)) 
        print("Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SpringLeaf Data Preprocessing")
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default="data")
    args = parser.parse_args()
    preprocess(args.data_dir, verbose=True)