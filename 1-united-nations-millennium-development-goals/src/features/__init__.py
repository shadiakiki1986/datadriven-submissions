def load_raw():
    import pandas as pd
    import os
    df = pd.read_csv(os.path.join(os.path.pardir,'data','raw','TrainingSet.csv'))
    return df

def imply_columns(df):
    cols = {}
    cols['meta']   = ['Unnamed: 0', 'Series Name', 'Country Name', 'Series Code']
    cols['features'] = list(set(df.columns) - set(cols['meta']))
    cols['features'].sort()
    # assert not df['Series Code'].duplicated().any()
    assert not df['Unnamed: 0'].duplicated().any()
    return cols

def make_submission(df):
    if ', '.join(df.columns) != ', 2008 [YR2008], 2012 [YR2012]':
      raise ValueError("Wrong columns passed")

    import datetime as dt
    import os

    fn = 'submission_%s.csv'%(dt.datetime.today().strftime('%Y%m%d_%H%M%S'))
    fn = os.path.join(os.path.pardir, 'data', 'interim', fn)
    df.to_csv(fn, index=False)

    from zipfile import ZipFile, ZIP_DEFLATED
    fn2 = '%s.zip'%fn
    with ZipFile(fn2, 'w', ZIP_DEFLATED) as myzip:
        myzip.write(fn)
        
    return fn, fn2
