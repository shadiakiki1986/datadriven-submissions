import pandas as pd
import os

def load_raw():
    myjoin = lambda x: os.path.join(os.path.pardir,'data','raw',x)
    df_all = {
        'features_train': pd.read_csv(myjoin('dengue_features_train.csv'), index_col=['city','week_start_date'], parse_dates=['week_start_date']).sort_index(),
        'labels_train': pd.read_csv(myjoin('dengue_labels_train.csv')),
        'features_test': pd.read_csv(myjoin('dengue_features_test.csv'), index_col=['city','week_start_date'], parse_dates=['week_start_date']).sort_index(),
        'submission': pd.read_csv(myjoin('submission_format.csv')),
    }
    
    # add week_start_date to labels_train
    for k1,k2 in (('labels_train', 'features_train'), ('submission', 'features_test')):
        df_all[k1] = df_all[k1].merge(
            df_all[k2].reset_index()[['city', 'week_start_date', 'year', 'weekofyear']]
        )
        assert pd.notnull(df_all[k1]['week_start_date']).all()
        df_all[k1].set_index(['city', 'week_start_date'], inplace=True)
    
    return df_all



def make_submission(df):
    df = df[['city','year','weekofyear','total_cases']]
    
    #if ', '.join(df.columns) != ', 2008 [YR2008], 2012 [YR2012]':
#      raise ValueError("Wrong columns passed")

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