import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from google.cloud import storage


def get_data():
    ''' returns a DataFrame '''
    df =pd.read_csv('../raw_data/data_final_indé_medium3.csv')
    return df

def clean_data(df):
    pass


if __name__ == '__main__':
    df=get_data()
