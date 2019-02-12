import sys
import pandas as pd

def stratified_sample_df(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_

def read_csv_in_chuncks(file_name, header = False, chunk_size = 300):
    iterator = 0
    print(f"Reading '{file_name}' in {chunk_size} chunks...")
    print("")
    for chunk in pd.read_csv(file_name, delimiter = ',', chunksize=chunk_size, parse_dates=[1] ): 
        yield (chunk)
        iterator = iterator + 1
        sys.stdout.print(f'\r{iterator} cunks read.')