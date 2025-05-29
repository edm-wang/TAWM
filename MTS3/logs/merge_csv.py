# import numpy as np
import pandas as pd

if __name__ == '__main__':
    for task in ['mw-basketball']:
        mts3_H = 'H50' # 'H3' # 'H50' # 'H33' # 'H11' # 
        df_all = None
        for i, dt in enumerate([0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02, 0.03, 0.05]):
            csv_name = f'MTS3-{mts3_H}-{task}-multidt-{dt}.csv'
            # csv_name = f'MTS3-{mts3_H}-{task}-multidt-{dt}.csv'
            df_ = pd.read_csv(csv_name)
            if i==0: df_all = df_
            else: df_all = pd.concat([df_all, df_], axis=0, ignore_index=True)

        df_all.to_csv(f'MTS3-{mts3_H}-{task}-multidt.csv', index=False)
