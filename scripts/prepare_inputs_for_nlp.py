import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay

def prepare():
    os.makedirs('./my_project/download/time_series', exist_ok=True)
    os.makedirs('./my_project/layer_1/links', exist_ok=True)

    # tickers sample
    tickers = ['SPY','QQQ','IWM']

    # generate business days
    start = pd.to_datetime('2023-01-02')
    dates = pd.bdate_range(start, periods=30)

    # synthetic close prices
    rng = np.random.default_rng(42)
    data = {t: (100 + np.cumsum(rng.normal(scale=1.0, size=len(dates)))) for t in tickers}
    close_df = pd.DataFrame(data, index=dates)
    close_df.to_csv('./my_project/download/time_series/close.csv')

    # homologous terms
    rows = []
    for t in tickers:
        rows.append({'Ticker': t, 'Term1': f"{t} company", 'Term2': f"{t} earnings"})
    homo_df = pd.DataFrame(rows)
    os.makedirs('./my_project/download', exist_ok=True)
    homo_df.to_csv('./my_project/download/homologous.csv', index=False)

    # create 10-day intervals and links
    intervals = []
    dates_list = list(close_df.index)
    for i in range(0, len(dates_list), 10):
        start = dates_list[i]
        end = dates_list[min(i+9, len(dates_list)-1)]
        intervals.append((start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))

    for _, row in homo_df.iterrows():
        ticker = row['Ticker']
        terms = [row['Term1'], row['Term2']]
        df_links = pd.DataFrame(index=[start for start, _ in intervals], columns=terms)
        for (start, end) in intervals:
            for term in terms:
                q = term.replace(' ', '+')
                link = f"https://www.google.com/search?q={q}&tbs=cdr:1,cd_min:{start},cd_max:{end}"
                df_links.loc[start, term] = link
        out_path = f'./my_project/layer_1/links/{ticker}_links.csv'
        df_links.to_csv(out_path)
        print('wrote', out_path)

if __name__=='__main__':
    prepare()
