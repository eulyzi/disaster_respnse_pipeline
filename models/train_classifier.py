import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import lightgbm as lgbm 

import pickle
import nltk
from typing import Tuple

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(database_filepath:str)->Tuple[pd.DataFrame,pd.DataFrame,list]:
    """
    load data from database
    Args:
      database_filepath: database path
    Return:
      X: pandas.DataFrame, feature matrix data for machine learning
      Y: pandas.DataFrame, target  matrix data for machine learning
      category_names: list,the names of target category
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_response',con=engine)
    X = df['message']
    Y = df.drop(['message','original','genre'],axis=1)
    category_names = list(Y.columns)
    return X,Y,category_names


def tokenize(text:str)->list:
    """
    tokenize text
    Args:
       text: text needed to tokenize
    Return:
       clean_tokens: tokenized text list
    """
    # get rid of non-words like `,`,` / `and split text into word
    text = re.sub('[^a-zA-Z0-9]',' ',text)
    tokens = word_tokenize(text)
    # lemmatize word
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    clean_tokens = [w for w in clean_tokens if w  not in stopwords.words("english")]
    return clean_tokens


def build_model():
    """
    build model use `Pipeline` and `GridSearchCV`
    Args:
        None
    Return:
        None
    """
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf',MultiOutputRegressor(lgbm.LGBMClassifier()))
                    ])
    # gridsearch
    parameters = {
    'clf__estimator__num_leaves':[32,64]
    }
    cv = GridSearchCV(pipeline,param_grid=parameters)
    return cv
    
    
    
def evaluate_model(model:Pipeline, X_test:pd.DataFrame, Y_test:pd.DataFrame, category_names:list)->None:
    """
    evaluate model with test data
    Args:
      model: Pipeline,fitted model with train data
      X_test: pd.DataFrame,test feature data
      Y_test: pd.DataFrame,test target data
   Return:
      None
    """
    y_pred = model.predict(X_test)
    report=classification_report(Y_test, y_pred,target_names=category_names)
    print(report)


def save_model(model:Pipeline, model_filepath:str)->None:
    """
    save model to local
    Args:
      model:  Pipeline,fitted model with train data
      model_filepath: model save path
    Return:
      None.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
        train machine learning model and save
        Args:
          None
       Return:
          None
     """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()