_EPSILON = 1e-08

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import pickle

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

import pycox
from pycox.evaluation import EvalSurv

import import_data_attention as impt
from EU_SURV import Model_Longitudinal_Attention

from utils_eval import c_index, brier_score
from utils_log import save_logging, load_logging
from utils_helper import f_get_minibatch, f_get_boosted_trainset, f_get_minibatch_tail

def _f_get_pred(sess, model, data, data_mi, pred_horizon):
    '''
    Predictions based on the prediction time.
    Create new_data and new_mask2 that are available previous or equal to the prediction time (no future measurements are used)
    '''
    new_data    = np.zeros(np.shape(data))
    new_data_mi = np.zeros(np.shape(data_mi))

    meas_time = np.concatenate([np.zeros([np.shape(data)[0], 1]), np.cumsum(data[:, :, 0], axis=1)[:, :-1]], axis=1)

    for i in range(np.shape(data)[0]):
        last_meas = np.sum(meas_time[i, :] <= pred_horizon)

        new_data[i, :last_meas, :]    = data[i, :last_meas, :]
        new_data_mi[i, :last_meas, :] = data_mi[i, :last_meas, :]
    pred, pred_event = model.predict(new_data, new_data_mi)

    return pred, pred_event


def f_get_risk_predictions(sess, model, data_, data_mi_, pred_time, eval_time):
    
    pred, _ = _f_get_pred(sess, model, data_[[0]], data_mi_[[0]], 0)
    _, num_Event, num_Category = np.shape(pred)
        
    risk_all = {}
    for k in range(num_Event):
        risk_all[k+1] = np.zeros([np.shape(data_)[0], len(pred_time), len(eval_time)])
            
    for p, p_time in enumerate(pred_time):
        ### PREDICTION
        pred_horizon = int(p_time)
        pred, pred_event = _f_get_pred(sess, model, data_, data_mi_, pred_horizon)


        for t, t_time in enumerate(eval_time):
            eval_horizon = int(t_time) + pred_horizon # if eval_horizon >= num_Category, output the maximum...

            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:,:,pred_horizon:(eval_horizon+1)], axis=2) # risk score until eval_time
            risk = risk / (np.sum(np.sum(pred[:,:,pred_horizon:], axis=2), axis=1, keepdims=True) +_EPSILON) 

            for k in range(num_Event):
                risk_all[k+1][:, p, t] = risk[:, k] * pred_event[:,k+1].reshape(-1)
                
    return risk_all


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if tf.__version__.startswith('2'):
        tf.random.set_seed(seed)
    else:
        tf.set_random_seed(seed)
        
    print(f"Random Seed set to {seed}")

# Execution parameters
data_mode = 'PBC2_final'  
seed = 1234
seed_everything(seed)

##### IMPORT DATASET
'''
    num_Category    = max event/censoring time * 1.2
    num_Event       = number of evetns i.e. len(np.unique(label))-1
    max_length      = maximum number of measurements
    x_dim           = data dimension including delta (1 + num_features)
    x_dim_cont      = dim of continuous features
    x_dim_bin       = dim of binary features
    mask1, mask2, mask3 = used for cause-specific network (FCNet structure)
'''

if data_mode == 'PBC2_final':
    (x_dim, x_dim_cont, x_dim_bin), (data, time, label), (mask1, mask2, mask3), (data_mi) = impt.import_dataset(norm_mode = 'standard', multi= True)
    
    # This must be changed depending on the datasets, prediction/evaliation times of interest
    pred_time = [1,  3*52, 7*52] # prediction time (in months)
    eval_time = [180, 360, 720] # months evaluation time (for C-index and Brier-Score)

elif data_mode == 'mimic_final':
    (x_dim, x_dim_cont, x_dim_bin), (data, time, label), \
    (mask1, mask2, mask3), (data_mi), (tr_idx, te_idx) = impt.import_dataset_mimic(norm_mode="none", multi=True)
    
    pred_time = [6] # prediction time (in days)
    eval_time = [14] # days evaluation time (for C-index and Brier-Score)

elif data_mode == 'hypert4':
    (x_dim, x_dim_cont, x_dim_bin), (data, time, label), \
    (mask1, mask2, mask3), (data_mi), (tr_idx, te_idx) = impt.import_dataset(norm_mode="none", multi=True)
    
    pred_time = [6] # prediction time (in days)
    eval_time = [9] # days evaluation time (for C-index and Brier-Score)

else:
    print("Data mode error!")

_, num_Event, num_Category  = np.shape(mask1)  # dim of mask3: [subj, Num_Event, Num_Category]
max_length = np.shape(data)[1]

file_path = '{}'.format(data_mode)

if not os.path.exists(file_path):
    os.makedirs(file_path)


def f_get_fc_mask1(meas_time, num_Event, num_Category):
    '''
    mask1 is required to get the contional probability (to calculate the denominator part)
    mask1 size is [N, num_Event, num_Category]. 1's until the last measurement time
    '''
    mask = np.zeros([np.shape(meas_time)[0], num_Event, num_Category]) # for denominator
    for i in range(np.shape(meas_time)[0]):
        mask[i, :, :int(meas_time[i, 0]+1)] = 1 # last measurement time

        
def f_get_fc_mask_tail(time, label, num_Event, num_Category, include_tau=False):
    """
    Tail mask for the new loss term.
    shape: [N, num_Event, num_Category]
    - if event observed (label[i,0] > 0): only the observed cause gets 1s on tau > tau^i
      (or tau >= tau^i if include_tau=True)
    - if censored (label[i,0] == 0): all zeros
    """
    N = np.shape(time)[0]
    mask = np.zeros((N, num_Event, num_Category), dtype=np.float32)
    for i in range(N):
        if label[i, 0] != 0:  # observed event
            e = int(label[i, 0] - 1)       # cause index (0-based)
            t_idx = int(time[i, 0])        # event time bin index (0-based)
            start = t_idx if include_tau else (t_idx + 1)
            if start < num_Category:
                mask[i, e, start:] = 1.0   # only the observed cause gets tail mass
    return mask

mask_tail = f_get_fc_mask_tail(time, label, num_Event, num_Category, include_tau=False)

burn_in_mode = 'OFF' #{'ON', 'OFF'}
boost_mode   = 'OFF' #{'ON', 'OFF'}

##### HYPER-PARAMETERS
new_parser = {'mb_size': 64,
             'iteration_burn_in': 5000,
             'iteration': 5000,
             'keep_prob': 0.6,
             'lr_train': 1e-4,
             'h_dim_RNN': 100,
             'h_dim_FC' : 100,
             'num_layers_RNN':2,
             'num_layers_ATT':2,
             'num_layers_CS' :2,
             'RNN_type':'GRU', #{'LSTM', 'GRU'}
             'FC_active_fn' : tf.nn.relu,
             'RNN_active_fn': tf.nn.tanh,
            'reg_W'         : 1e-5,
            'reg_W_out'     : 0.,
             'alpha' :1.0,
             'beta'  :0.1,
             'gamma' :1.0,
             "tail"  :0.1,
}

# INPUT DIMENSIONS
input_dims = { 'x_dim'         : x_dim,
               'x_dim_cont'    : x_dim_cont,
               'x_dim_bin'     : x_dim_bin,
               'num_Event'     : num_Event,
               'num_Category'  : num_Category,
               'max_length'    : max_length }

# NETWORK HYPER-PARMETERS
network_settings = { 'h_dim_RNN'         : new_parser['h_dim_RNN'],
                     'h_dim_FC'          : new_parser['h_dim_FC'],
                     'num_layers_RNN'    : new_parser['num_layers_RNN'],
                     'num_layers_ATT'    : new_parser['num_layers_ATT'],
                     'num_layers_CS'     : new_parser['num_layers_CS'],
                     'RNN_type'          : new_parser['RNN_type'],
                     'FC_active_fn'      : new_parser['FC_active_fn'],
                     'RNN_active_fn'     : new_parser['RNN_active_fn'],
                     'initial_W'         : tf.contrib.layers.xavier_initializer(),
                     'reg_W'             : new_parser['reg_W'],
                     'reg_W_out'         : new_parser['reg_W_out']
                     }

mb_size           = new_parser['mb_size']
iteration         = new_parser['iteration']
iteration_burn_in = new_parser['iteration_burn_in']

keep_prob         = new_parser['keep_prob']
lr_train          = new_parser['lr_train']

alpha             = new_parser['alpha']
beta              = new_parser['beta']
gamma             = new_parser['gamma']
tail_lambda       = new_parser['tail']

# SAVE HYPERPARAMETERS
log_name = file_path + '/hyperparameters_log.txt'
save_logging(new_parser, log_name)

def fold_data(data, data_mi, time, label, mask1, mask2, mask3, mask_tail, train_idx, test_idx, seed):

    tr_data = data[train_idx]
    tr_data_mi = data_mi[train_idx]
    tr_time = time[train_idx]
    tr_label = label[train_idx]
    tr_mask1 = mask1[train_idx]
    tr_mask2 = mask2[train_idx]
    tr_mask3 = mask3[train_idx]
    tr_mask_tail = mask_tail[train_idx]

    te_data = data[test_idx]
    te_data_mi = data_mi[test_idx]
    te_time = time[test_idx]
    te_label = label[test_idx]
    te_mask1 = mask1[test_idx]
    te_mask2 = mask2[test_idx]
    te_mask3 = mask3[test_idx]
    te_mask_tail = mask_tail[test_idx]
    
    (tr_data,va_data, tr_data_mi, va_data_mi, tr_time,va_time, tr_label,va_label, 
    tr_mask1,va_mask1, tr_mask2,va_mask2, tr_mask3,va_mask3, tr_mask_tail, va_mask_tail) = train_test_split(tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, tr_mask_tail, test_size=0.2, stratify=tr_label, random_state=seed) 

    train_set = (tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, tr_mask_tail)
    val_set   = (va_data, va_data_mi, va_time, va_label, va_mask1, va_mask2, va_mask3, va_mask_tail)
    test_set  = (te_data, te_data_mi, te_time, te_label, te_mask1, te_mask2, te_mask3, te_mask_tail)

    return train_set, val_set, test_set

with open ("pbc_index.pkl", "rb") as f:
    pbc_index = pickle.load(f)


##### CREATE DYNAMIC-DEEPHIT NETWORK

for cv in range(5):

    seed_everything(seed + cv)
    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Model_Longitudinal_Attention(sess, "EU-SURV", input_dims, network_settings)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    train_set, val_set, test_set = fold_data(data, data_mi, time, label, mask1, mask2, mask3, mask_tail, pbc_index[f'tr_idx_{cv}'] -1 , pbc_index[f'te_idx_{cv}'] -1, pbc_index[f'seed_{cv}'] )

    (tr_data, tr_data_mi, tr_time, tr_label, tr_mask1, tr_mask2, tr_mask3, tr_mask_tail) = train_set
    (va_data, va_data_mi, va_time, va_label, va_mask1, va_mask2, va_mask3, va_mask_tail) = val_set
    (te_data, te_data_mi, te_time, te_label, te_mask1, te_mask2, te_mask3, te_mask_tail) = test_set
    
    ### TRAINING - BURN-IN
    if burn_in_mode == 'ON':
        print( "BURN-IN TRAINING ...")
        for itr in range(iteration_burn_in):
            x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = f_get_minibatch(mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3, seed)
            DATA = (x_mb, k_mb, t_mb)
            MISSING = (x_mi_mb)

            _, loss_curr = model.train_burn_in(DATA, MISSING, keep_prob, lr_train)

            if (itr+1)%1000 == 0:
                print('itr: {:04d} | loss: {:.4f}'.format(itr+1, loss_curr))


    ### TRAINING - MAIN
    print( "MAIN TRAINING ...")
    min_valid = 0.1
    loss_list = []
    val_loss_list = []
    min_loss = 1e+10
    for itr in range(iteration):

        x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb, m4_mb = f_get_minibatch_tail(mb_size, tr_data, tr_data_mi, tr_label, tr_time, tr_mask1, tr_mask2, tr_mask3, tr_mask_tail, itr)
        DATA = (x_mb, k_mb, t_mb)
        MASK = (m1_mb, m2_mb, m3_mb, m4_mb)
        MISSING = (x_mi_mb)
        PARAMETERS = (alpha, beta, gamma, tail_lambda)

        _, loss_curr = model.train(DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train)
        
        tr_loss = model.get_cost(DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train)
        val_loss = model.get_cost(DATA, MASK, MISSING, PARAMETERS, keep_prob, lr_train)

        loss_list.append(tr_loss)

        val_loss_list.append(val_loss)
        if (itr+1)%500 == 0:
            print('itr: {:04d} | loss: {:.4f}'.format(itr+1, loss_curr))

        ### VALIDATION  (based on average C-index of our interest)
        if (itr+1)%100 == 0:        
            risk_all = f_get_risk_predictions(sess, model, va_data, va_data_mi, pred_time, eval_time)
            
            for p, p_time in enumerate(pred_time):
                pred_horizon = int(p_time)
                val_result1 = np.zeros([num_Event, len(eval_time)])
                
                for t, t_time in enumerate(eval_time):                
                    eval_horizon = int(t_time) + pred_horizon
                    for k in range(num_Event):
                        val_result1[k, t] = c_index(risk_all[k+1][:, p, t], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                
                if p == 0:
                    val_final1 = val_result1
                else:
                    val_final1 = np.append(val_final1, val_result1, axis=0)

            tmp_valid = np.mean(val_final1)

            if tmp_valid >  min_valid:
                min_valid = tmp_valid
                saver.save(sess, file_path + f'/model_{cv}_pbc')
                print( 'updated.... average c-index = ' + str('%.4f' %(tmp_valid)))

            if val_loss[0] < min_loss:
                min_loss = val_loss[0]
                saver.save(sess, file_path + f'/model_loss_{cv}_pbc')
                print( 'updated (loss)... loss = ' + str('%.4f' %(min_loss)))

result_c_index_list = []
result_brier_list = []

pred_time = [1,  3*52, 7*52] 
eval_time = [180, 360, 720] 

for cv in range(5):
    saver.restore(sess, file_path + f'/model_{cv}_pbc') 

    train_set, val_set, test_set = fold_data(data, data_mi, time, label, mask1, mask2, mask3, mask_tail, pbc_index[f'tr_idx_{cv}'] -1 , pbc_index[f'te_idx_{cv}'] -1, pbc_index[f'seed_{cv}'] )

    (te_data, te_data_mi, te_time, te_label, te_mask1, te_mask2, te_mask3, te_mask_tail) = test_set

    risk_all = f_get_risk_predictions(sess, model, te_data, te_data_mi, pred_time, eval_time)

    all_c_index, all_brier = [], []
    for p, p_time in enumerate(pred_time):
        pred_horizon = int(p_time)
        result1, result2 = np.zeros([num_Event, len(eval_time)]), np.zeros([num_Event, len(eval_time)])

        for t, t_time in enumerate(eval_time):                
            eval_horizon = int(t_time) + pred_horizon
            for k in range(num_Event):
                result1[k, t] = c_index(risk_all[k+1][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon)
                result2[k, t] = brier_score(risk_all[k+1][:, p, t], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon)

        all_c_index.append(result1)
        all_brier.append(result2)

    # shape: [num_pred_time * num_event, num_eval_time]
    all_c_index = np.concatenate(all_c_index, axis=0)
    all_brier = np.concatenate(all_brier, axis=0)

    result_c_index_list.append(all_c_index)
    result_brier_list.append(all_brier)

# Calculate mean and std of results
c_index_array = np.stack(result_c_index_list, axis=0)  # shape: (5, Nrow, Ncol)
brier_array = np.stack(result_brier_list, axis=0)

c_index_mean = np.mean(c_index_array, axis=0)
c_index_std  = np.std(c_index_array, axis=0)

brier_mean = np.mean(brier_array, axis=0)
brier_std  = np.std(brier_array, axis=0)

def format_result(mean_arr, std_arr):
    formatted = np.char.add(
        np.char.add(np.round(mean_arr, 3).astype(str), " ± "),
        np.round(std_arr, 3).astype(str)
    )
    return formatted

c_index_formatted = format_result(c_index_mean, c_index_std)
brier_formatted   = format_result(brier_mean, brier_std)

row_header = []
for p_time in pred_time:
    for t in range(num_Event):
        row_header.append(f'pred_time {p_time}: event_{t+1}')
col_header = [f'eval_time {t}' for t in eval_time]

df_c_index = pd.DataFrame(c_index_formatted, index=row_header, columns=col_header)
df_brier   = pd.DataFrame(brier_formatted, index=row_header, columns=col_header)

print('========================================================')
print('C-INDEX (mean ± std):')
print(df_c_index)
print('--------------------------------------------------------')
print('BRIER-SCORE (mean ± std):')
print(df_brier)
print('========================================================')