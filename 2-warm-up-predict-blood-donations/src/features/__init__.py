def load_raw():
    import pandas as pd
    import os
    df = {}
    df['train'] = pd.read_csv(os.path.join(os.path.pardir,'data','raw','train.csv'))
    df['submit'] = pd.read_csv(os.path.join(os.path.pardir,'data','raw','test.csv'))
    return df

def imply_columns(df):
    cols = {}
    cols['meta']   = ['Unnamed: 0']
    cols['target'] = list(set(df['train'].columns) - set(df['submit'].columns) - set(cols['meta']))
    cols['target'].sort()
    cols['features'] = list(set(df['submit'].columns) - set(cols['meta']))
    cols['features'].sort()
    assert not df['train']['Unnamed: 0'].duplicated().any()
    #
    # despite df['submit'] having duplicates in "Unnamed: 0", which would serve as an index,
    # it cannot be de-duplicated because the drivendata.org submission is expected to have 
    # the duplicates included
    # 
    # df['submit'] = df['submit'][~df['submit']['Unnamed: 0'].duplicated()]
    # assert not df['submit']['Unnamed: 0'].duplicated().any()
    #
    # for df['train'], 'Unnamed: 0' is already de-duplicated, so it can be used as an index
    df['train'] = df['train'].set_index('Unnamed: 0')
    #test  = test.set_index('Unnamed: 0')
    return cols

# feature engineering from
# https://github.com/saeedhadikhanloo/MyProjectsCodes/blob/master/BloodDonation/BloodDonationVar1.ipynb
def append_features(df):
    df['synth: Ave Dist Two Don'] = (df['Months since First Donation'] - df['Months since Last Donation'])/df['Number of Donations']
    df['synth: Last More Than Ave'] = df['Months since Last Donation'] > df['synth: Ave Dist Two Don']
    df['synth: Last More Than Ave'] = df['synth: Last More Than Ave'].astype(int)
    df['synth: Prod'] = df['synth: Last More Than Ave'] * df['synth: Ave Dist Two Don']
    df['synth: Pord2'] = df['Months since Last Donation'] * df['Number of Donations']
    return df


def make_submission(df_upload):
    if ', '.join(df_upload.columns) != ', Made Donation in March 2007':
      raise ValueError("Wrong columns passed")

    import datetime as dt
    import os

    fn = 'submission_%s.csv'%(dt.datetime.today().strftime('%Y%m%d_%H%M%S'))
    fn = os.path.join(os.path.pardir, 'data', 'interim', fn)
    df_upload.to_csv(fn, index=False)

    from zipfile import ZipFile, ZIP_DEFLATED
    fn2 = '%s.zip'%fn
    with ZipFile(fn2, 'w', ZIP_DEFLATED) as myzip:
        myzip.write(fn)
        
    return fn, fn2
