import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath:str) -> pd.DataFrame:
    """
    load `messages` and `categories`
    Args:
       messages_filepath: messages filepath
       categories_filepath: categories filepath
    Return:
       a pandas DataFrame combined 2 datasets.
    """
    messages = pd.read_csv(messages_filepath,index_col='id')
    categories = pd.read_csv(categories_filepath,index_col='id')
    df = pd.concat([messages,categories],axis=1)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    clean `load_data` DataFrame. extract category to binary numbers and delete duplicated results.
    Args:
       df:  `load_data` DataFrame.
    Return:
       df_clean: clearned  DataFrame.
    """
    # create a dataframe of the 36 individual category columns
    df_categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = df_categories.iloc[0,:]
    category_colnames = [r.split('-')[0] for r in row]
    # rename the columns of `categories`
    df_categories.columns = category_colnames
    # convert category values to just numbers 0 or 1
    for column in category_colnames:
        df_categories[column] = df_categories[column].apply(lambda x:x.split('-')[1])
        df_categories[column] = df_categories[column].astype('int8')
    df_rebulid = pd.concat([df.drop(['categories'],axis=1),df_categories],axis=1)
    # column `related` has value =2, delete them
    df_rebulid=df_rebulid[df_rebulid['related']<=1]
    # drop duplicates
    df_clean = df_rebulid.drop_duplicates()
    return df_clean

def save_data(df: pd.DataFrame, database_filename:str)-> None:
    """
    save pandas DataFrame to database
    Args:
      df:pandas DataFrame ready to save
      database_filename: database filepath used to save data.
   Return:
      None.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index=False,if_exists='replace')  


def main():
    """
    process data to database
    Args:
      None
   Return:
      None
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()