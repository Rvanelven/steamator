import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


def get_data():
    ''' returns a DataFrame '''
    df =pd.read_csv('../raw_data/data_final_indé_medium2.csv')
    return df

def clean_data(df):
    list_of_tags = df['top_5_tags'].tolist()
    vectorizer = TfidfVectorizer().fit(list_of_tags)

    data_vectorized = vectorizer.transform(list_of_tags)
    pass

if __name__ == '__main__':
    df=get_data()
