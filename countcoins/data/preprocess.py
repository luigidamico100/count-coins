import pandas as pd
from sklearn.model_selection import train_test_split

def add_split_column(df, test_size=.3):
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['currencies'])
    df_train['set'], df_test['set'] = 'train', 'test'
    df_processed = pd.concat((df_train, df_test), axis=0)

    return df_processed