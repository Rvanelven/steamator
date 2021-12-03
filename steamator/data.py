import pandas as pd

def get_data():
    ''' returns a DataFrame '''
    df =pd.read_csv('../raw_data/data_final.csv')
    return df

def clean_data(df):
    #TO DO
    pass

if __name__ == '__main__':
    df=get_data()
