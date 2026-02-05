import numpy as np
import pandas as pd
import pickle
import os

_EPSILON = 1e-08

#########################
# 0) Utility Functions
#########################
def f_get_Normalization(X, norm_mode):    
    num_Patient, num_Feature = np.shape(X)
    
    if norm_mode == 'standard': # zero mean, unit variance
        for j in range(num_Feature):
            std_ = np.nanstd(X[:,j])
            mean_ = np.nanmean(X[:, j])
            if std_ != 0:
                X[:,j] = (X[:,j] - mean_)/std_
            else:
                X[:,j] = (X[:,j] - mean_)
    elif norm_mode == 'normal': # min-max normalization
        for j in range(num_Feature):
            min_ = np.nanmin(X[:,j])
            max_ = np.nanmax(X[:,j])
            X[:,j] = (X[:,j] - min_)/(max_-min_+_EPSILON)
    else:
        print("norm_mode ERROR!")
    return X


def f_get_fc_mask1(meas_time, num_Event, num_Category):
    """
    mask1: shape [N, num_Event, num_Category]
           1's until the last measurement time
    """
    N = meas_time.shape[0]
    mask = np.zeros((N, num_Event, num_Category))
    for i in range(N):
        lm = int(meas_time[i,0])
        if lm >= num_Category:
            lm = num_Category-1
        mask[i, :, :lm+1] = 1.0
    return mask


def f_get_fc_mask2(time, label, num_Event, num_Category):
    """
    mask2: for log-likelihood
           if label!=0 => exact time index=1
           if label=0 => fill 1 after censoring time
    """
    N = time.shape[0]
    mask = np.zeros((N, num_Event, num_Category))
    for i in range(N):
        y_ = int(label[i,0])
        t_ = int(time[i,0])
        if t_ >= num_Category:
            t_= num_Category-1
        if y_ != 0:
            # not censored => mark single
            mask[i, y_-1, t_] = 1.
        else:
            # censored => fill 1 from t_+1 to end
            if (t_+1)<num_Category:
                mask[i,:, (t_+1):] = 1.
    return mask


def f_get_fc_mask3(time, meas_time, num_Category):
    '''
        mask3 is required calculate the ranking loss (for pair-wise comparision)
        mask3 size is [N, num_Category]. 
        - For longitudinal measurements:
             1's from the last measurement to the event time (exclusive and inclusive, respectively)
             denom is not needed since comparing is done over the same denom
        - For single measurement:
             1's from start to the event time(inclusive)
    '''
    mask = np.zeros([np.shape(time)[0], num_Category]) # for the first loss function
    if np.shape(meas_time):  # lonogitudinal measurements 
        for i in range(np.shape(time)[0]):
            t1 = int(meas_time[i, 0]) # last measurement time
            t2 = int(time[i, 0]) # censoring/event time
            mask[i,(t1+1):(t2+1)] = 1  # this excludes the last measurement time and includes the event time
    else:                    # single measurement
        for i in range(np.shape(time)[0]):
            t = int(time[i, 0]) # censoring/event time
            mask[i,:(t+1)] = 1  # this excludes the last measurement time and includes the event time
    return mask


##### TRANSFORMING DATA
def f_construct_dataset(df, feat_list):
    '''
        id   : patient indicator
        tte  : time-to-event or time-to-censoring
            - must be synchronized based on the reference time
        times: time at which observations are measured
            - must be synchronized based on the reference time (i.e., times start from 0)
        label: event/censoring information
            - 0: censoring
            - 1: event type 1
            - 2: event type 2
            ...
    '''
    grouped  = df.groupby(['id'])
    id_list  = pd.unique(df['id'])
    max_meas = np.max(grouped.count())[0]

    data     = np.zeros([len(id_list), max_meas, len(feat_list)+1])
    pat_info = np.zeros([len(id_list), 5])

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)

        pat_info[i,4] = tmp.shape[0]                         # number of measurement
        pat_info[i,3] = np.max(tmp['times'])     # last measurement time
        pat_info[i,2] = tmp['label'][0]      # cause
        pat_info[i,1] = tmp['tte'][0]         # time_to_event
        pat_info[i,0] = tmp['id'][0]      

        data[i, :int(pat_info[i, 4]), 1:]  = tmp[feat_list]
        data[i, :int(pat_info[i, 4]-1), 0] = np.diff(tmp['times'])
    
    return pat_info, data


def make_df_with_baseline(df, baseline_list, dynamic_list):
    """
    df: (id, tte, times, label, ...) columns
    baseline_list: static features
    dynamic_list : time-series features

    return:
     df_new: for each id, 'baseline row + time-series rows (times shifted by +1)'
    """
    out_list = []
    grouped = df.groupby('id')
    for pid, tmp in grouped:
        tmp = tmp.copy().reset_index(drop=True)
        n_meas = tmp.shape[0]
        if n_meas<1:
            continue

        # (1) baseline row => copy tmp.iloc[0], but set times=0
        base_row = tmp.iloc[[0]].copy()  
        base_row['times'] = 0.0
        # baseline feats remain same for 'baseline_list'
        # dynamic feats => set 0 or np.nan
        for col in dynamic_list:
            base_row[col] = np.nan

        # (2) shift tmp times => times= times+1
        tmp['times'] = tmp['times'] + 1.0
        # (3) concat => baseline row + shifted rows
        combined = pd.concat([base_row, tmp], ignore_index=True)
        out_list.append(combined)

    df_new = pd.concat(out_list, ignore_index=True)
    df_new['tte'] = df_new['tte'] + 1.0
    return df_new


def import_dataset(norm_mode = 'standard', multi = False):
    df_                = pd.read_csv('./data/pbc2_cleaned.csv')
    baseline_list = ['sex','age','drug']
    dynamic_list = ['ascites','hepatomegaly','spiders','edema','serBilir','serChol','albumin','alkaline','SGOT','platelets','prothrombin', 'histologic']

    df_ = make_df_with_baseline(df_.copy(), baseline_list, dynamic_list)
    
    # label 0: censored, 1: death, 2: liver transplant
    if multi == False:
        df_                = df_[df_['label'] != 2]
    else:
        df_                = df_
        
    bin_list           = ['drug', 'sex', 'ascites', 'hepatomegaly', 'spiders']
    cont_list          = ['age', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 'prothrombin', 'histologic']
    feat_list          = cont_list + bin_list
    df_                = df_[['id', 'tte', 'times', 'label']+feat_list]
    df_org_            = df_.copy(deep=True)

    df_[cont_list]     = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    pat_info, data     = f_construct_dataset(df_, feat_list)
    _, data_org        = f_construct_dataset(df_org_, feat_list)

    data_mi                  = np.zeros(np.shape(data))
    data_mi[np.isnan(data)]  = 1
    data_org[np.isnan(data)] = 0
    data[np.isnan(data)]     = 0 

    x_dim           = np.shape(data)[2] # 1 + x_dim_cont + x_dim_bin (including delta)
    x_dim_cont      = len(cont_list)
    x_dim_bin       = len(bin_list) 

    last_meas       = pat_info[:,[3]]  # pat_info[:, 3] contains age at the last measurement
    label           = pat_info[:,[2]]  # two competing risks
    time            = pat_info[:,[1]]  # age when event occurred

    num_Category    = int(np.max(pat_info[:, 1])) 
    num_Event       = len(np.unique(label)) - 1

    if num_Event == 1:
        label[np.where(label!=0)] = 1 # make single risk

    mask1           = f_get_fc_mask1(last_meas, num_Event, num_Category)
    mask2           = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask3           = f_get_fc_mask3(time, last_meas, num_Category)

    DIM             = (x_dim, x_dim_cont, x_dim_bin)
    DATA            = (data, time, label)
    MASK            = (mask1, mask2, mask3)

    return DIM, DATA, MASK, data_mi



def import_dataset_mimic(norm_mode='standard', multi=False):
    # 1. Load Data
    train_df = pd.read_csv('./data/train.csv')
    test_df  = pd.read_csv('./data/test.csv')

    # 2. Generate ID and Separate Indices
    # Assuming 5 rows per patient
    N_train = int(len(train_df) / 5)
    N_test  = int(len(test_df) / 5)

    # Train ID: 0 ~ N_train-1
    train_df['id'] = train_df.index // 5
    # Test ID: N_train ~ (N_train + N_test - 1)
    test_df['id']  = (test_df.index // 5) + N_train

    # Create index lists
    idx_train = list(range(N_train))
    idx_test  = list(range(N_train, N_train + N_test))

    # 3. Concatenate Data
    df_ = pd.concat([train_df, test_df], ignore_index=True)
    df_['time'] = np.tile([1, 2, 3, 4, 5], len(df_) // 5)
    df_['label'] = (df_.groupby('id')['death_reason'].transform('max') > 0).astype(int)

    # 4. Clean Label and Time Columns
    # Create combined label (if label is 1, use death_reason, else 0)
    df_['processed_label'] = np.where(df_['label'] == 1, df_['death_reason'], 0)
    
    # Rename and drop columns
    df_ = df_.rename(columns={'time': 'times', 'ett': 'tte'})
    df_ = df_.drop(columns=['label', 'death_reason', 'Unnamed: 0'], errors='ignore')
    df_ = df_.rename(columns={'processed_label': 'label'})

    # 5. Define Feature List
    bin_list = ['gender']
    exclude_cols = ['id', 'tte', 'times', 'label'] + bin_list
    cont_list = [c for c in df_.columns if c not in exclude_cols]
    
    df_ = make_df_with_baseline(df_.copy(), bin_list, cont_list)

    feat_list = cont_list + bin_list
    df_['tte'] = df_['tte'] + 5
    df_       = df_[['id', 'tte', 'times', 'label'] + feat_list]
    df_org_   = df_.copy(deep=True)

    if norm_mode != 'none':
        df_[cont_list] = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    pat_info, data = f_construct_dataset(df_, feat_list)
    _, data_org    = f_construct_dataset(df_org_, feat_list)

    data_mi                  = np.zeros(np.shape(data))
    data_mi[np.isnan(data)]  = 1
    data_org[np.isnan(data)] = 0
    data[np.isnan(data)]     = 0 

    x_dim       = np.shape(data)[2] 
    x_dim_cont  = len(cont_list)
    x_dim_bin   = len(bin_list) 

    last_meas   = pat_info[:,[3]]  
    label       = pat_info[:,[2]]  
    time        = pat_info[:,[1]]  

    num_Category = int(np.max(pat_info[:, 1])) 
    num_Event    = len(np.unique(label)) - 1

    if num_Event == 1:
        label[np.where(label!=0)] = 1 

    mask1 = f_get_fc_mask1(last_meas, num_Event, num_Category)
    mask2 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask3 = f_get_fc_mask3(time, last_meas, num_Category)

    DIM  = (x_dim, x_dim_cont, x_dim_bin)
    DATA = (data, time, label)
    MASK = (mask1, mask2, mask3)
    idx = (idx_train, idx_test)

    return DIM, DATA, MASK, data_mi, idx



def import_dataset_hypert4(norm_mode = 'standard', multi = False):
    # Anonymized path
    df_ = pd.read_csv('./data/T4_final.csv')

    # 1) Drop patients with tte == 0
    drop_ids = df_.loc[df_['tte'] == 0, 'user_id_Encoder'].unique()
    df_ = df_[~df_['user_id_Encoder'].isin(drop_ids)].copy()
    
    target_cols = ['tot_mg', 'Free T4', 'receptor']

    # 2) Identify dynamic patients (values change)
    group_changes = df_.groupby('user_id_Encoder')[target_cols].nunique(dropna=True)
    dynamic_ids = group_changes[group_changes.max(axis=1) > 1].index

    df_["is_dynamic"] = df_["user_id_Encoder"].isin(dynamic_ids)

    # 3) For static patients, select the row with the most non-NaN values
    df_["_non_na_cnt"] = df_[target_cols].notna().sum(axis=1)

    static_best_idx = (
        df_[~df_["is_dynamic"]]
        .sort_values(["user_id_Encoder", "_non_na_cnt"], ascending=[True, True])
        .groupby("user_id_Encoder")
        .tail(1)          # select the last one (highest non_na_cnt)
        .index
    )

    # 4) Combine all dynamic patients + best static rows
    df_filtered = pd.concat([
        df_[df_["is_dynamic"]],
        df_.loc[static_best_idx]
    ]).sort_index()

    df_filtered = df_filtered.drop(columns=["is_dynamic", "_non_na_cnt"])
    df_ = df_filtered.copy()
    
    # Fill baseline Free T4 if missing at interval 0
    condition = (df_['interval'] == 0) & (df_['Free T4'].isna())
    df_.loc[condition, 'Free T4'] = df_.loc[condition, 'initial T4']

    df_['id'] = df_['user_id_Encoder'].copy()

    df_[['height', 'weight']] = df_.groupby('user_id_Encoder')[['height', 'weight']].transform(lambda x: x.ffill().bfill())
    dynamic_list = ['Free T4', 'receptor', 'tot_mg']

    df_['tte']         = df_['tte'].astype(float)
    df_['times']       = df_['interval'].astype(float)

    bin_list = ['sex']  
    cont_list = ['initial T4', 'initial Receptor', 'Free T4', 'tot_mg', 'receptor','height', 'weight', 'age']
    feat_list          = cont_list + bin_list
    df_                = df_[[f'id', 'tte', 'times', 'label']+ feat_list]
    df_org_            = df_.copy(deep=True)

    df_[cont_list]    = f_get_Normalization(np.asarray(df_[cont_list]).astype(float), norm_mode)

    df_ = make_df_with_baseline(df_.copy(), [], dynamic_list)

    pat_info, data     = f_construct_dataset(df_, feat_list)
    _, data_org        = f_construct_dataset(df_org_, feat_list)

    data_mi = np.zeros_like(data, dtype=np.float32)
    data_mi[np.isnan(data)] = 1.0

    # NaN becomes 0 in model input
    data_org[np.isnan(data_org)] = 0.0
    data[np.isnan(data)] = 0.0

    # Key: Padding intervals are treated as missing
    for i in range(len(pat_info)):
        L = int(pat_info[i, 4])  # number of measurements
        data_mi[i, L:, :] = 1.0  # padding time steps => missing

    x_dim           = np.shape(data)[2] 
    x_dim_cont      = len(cont_list)
    x_dim_bin       = len(bin_list) 

    last_meas       = pat_info[:,[3]] 
    label           = pat_info[:,[2]]  
    time            = pat_info[:,[1]]  
    recommend       = {} 

    num_Category    = int(np.max(pat_info[:, 1]) + 1) 
    num_Event       = len(np.unique(label)) - 1

    if num_Event == 1:
        label[np.where(label!=0)] = 1 

    mask1           = f_get_fc_mask1(last_meas, num_Event, num_Category)
    mask2           = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask3           = f_get_fc_mask3(time, -1, num_Category)

    DIM             = (x_dim, x_dim_cont, x_dim_bin)
    DATA            = (data, time, label)
    MASK            = (mask1, mask2, mask3)

    return DIM, DATA, MASK, data_mi
