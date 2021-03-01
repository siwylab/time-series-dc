import numpy as np
import pandas as pd

# Load dataset
df = pd.read_pickle('/home/dan/Documents/siwylab/AWS/Full_filt_101_cx_el.pkl')


df['x_start'] = df.apply(lambda a: np.argmin(np.abs(a['xcm_um']-20)), axis=1)
df['x_end'] = df.apply(lambda a: np.argmin(np.abs(180-a['xcm_um'])), axis=1)


def pad_columns(columns, df):
    max_length = 35
    # Default value for data_outer
    data_outer = df

    # Enumerate over columns, align and pad selected sequences while ignoring nans
    for i, column in enumerate(columns):
        for ii, data in enumerate(df[column]):
            # Skip erroneously long rows
            if df.iloc[i]['seq_len'] > max_length:
                continue
            start = df.iloc[i]['x_start']
            end = df.iloc[i]['x_end']
            cleaned = np.nan_to_num(data, nan=1.0)[start:end]
            # Prepend data with start token
            padded = np.pad(np.array(cleaned), (0, max_length-len(cleaned)))
            if not ii:
                data_array = padded
            else:
                data_array = np.vstack((data_array, padded))
        # Subtract mean and divide by variance
        data_mean = np.mean(data_array)
        data_std = np.std(data_array)
        data_array = (data_array-data_mean)/data_std
        if not i:
            data_outer = np.expand_dims(data_array, axis=2)
        else:
            data_outer = np.concatenate((data_outer, np.expand_dims(data_array, axis=2)), axis=2)
    return data_outer


lstm_x = pad_columns(['padded_aspect', 'padded_perimeter', 'padded_area', 'padded_deform'], df)
lstm_y = df['y'].to_numpy()

# TODO: Incorporate into other df script
