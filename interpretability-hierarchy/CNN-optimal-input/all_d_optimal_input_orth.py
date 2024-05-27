# optimize mu_CPR, by keeping mu_GA fixed

import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.rc('font', size=18)
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import os
import sys
sys.path.append('../Climate-Learning')
import general_purpose.utilities as ut
import general_purpose.uplotlib as uplt


sys.path.append('../Climate-Learning/PLASIM')
import Learn2_new as ln

import probabilistic_regression as pr
pr.enable()
from committor_projection_NN import GradientRegularizer
import optimal_input as oi

# log to stdout
import logging
logging.getLogger().level = logging.INFO
logging.getLogger().handlers = [logging.StreamHandler(sys.stdout)]

HOME = '/ClimateDynamics/MediumSpace/ClimateLearningFR/alovo'

def get_arg(run, key, config_dict):
    return run['args'].get(key, ut.extract_nested(config_dict, key))

def get_completed(folder):
    return {k:v for k,v in ut.json2dict(f'{folder}/runs.json').items() if v['status'] == 'COMPLETED'}

def get_checkpoints(run, folder, monitor=None):
    config_dict = ut.json2dict(f'{folder}/config.json')
    nfolds = get_arg(run, 'nfolds', config_dict)
    opt_ckp, fold_subfolder = ln.optimal_checkpoint(f"{folder}/{run['name']}", nfolds, collective=False, metric=monitor or ut.extract_nested(config_dict,'metric'))
    return opt_ckp

def get_model(run, fold, folder, monitor=None):
    opt_ckp = get_checkpoints(run, folder, monitor=monitor)
    model = ln.load_model(f"{folder}/{run['name']}/fold_{fold}/cp-{opt_ckp[fold]:04d}.ckpt")
    return model


# inherit from oi.OptimalInput to implement orthogonality condition

class OrthOptimalInput(oi.OptimalInput):
    def __init__(self, *args, orth_coef=0, orth_model=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.orth_coef = orth_coef
        self.orth_model = orth_model if orth_coef else None
        if self.orth_model is None:
            self.orth_coef = 0

        if not self.orth_coef:
            print('Deprecated: if you do not need the orthogonal regularization, just use the native OptimalInput class')

    def init(self, seed):
        super().init(seed)
        if self.orth_model is not None:
            self.ori_orth = self.orth_model(self.seed_)

    def loss_function(self, x=None):
        loss = super().loss_function(x)
        if self.orth_coef:
            u = self.orth_model(self.inim_)
            assert loss.shape == u.shape == self.ori_orth.shape, 'Shape mismatch!'
            if self.info is not None:
                self.info['orth'] = u.numpy()
            loss = loss + (u - self.ori_orth)**2

        return loss

    

lon, lat = np.load('lon.npy'), np.load('lat.npy')

start_time = time.time()

dataset_filename = 'CPR-oi_d_orth_1.nc'
if os.path.exists(dataset_filename):
    raise FileExistsError()

## load test data

folder = 'CPR/r2/'
config_dict = ut.json2dict(f'{folder}/config.json')

load_data_kwargs = ut.extract_nested(config_dict, 'load_data_kwargs')
prepare_XY_kwargs = ut.extract_nested(config_dict, 'prepare_XY_kwargs')

ut.set_values_recursive(load_data_kwargs, {'year_list': 'range(800,1000)'}, inplace=True)
ut.set_values_recursive(prepare_XY_kwargs, {'do_balance_folds': False}, inplace=True)

trainer = ln.Trainer(config=config_dict)

_ = trainer.load_data(**load_data_kwargs)
_ = trainer.prepare_XY(trainer.fields, **prepare_XY_kwargs)

X_te = trainer.X
A_te = trainer.Y

threshold = np.load('Va-threshold.npy').item()
A_te_ = np.load('A_te.npy')
assert np.allclose(A_te, A_te_)

## get a model

runs = get_completed(folder)
for k,run in runs.items():
    print(k, run['args'], run['score'])

run = runs['0']
run_folder = f"{folder}/{run['name']}"

fold = 0

X_mean = np.load(f'{run_folder}/fold_{fold}/X_mean.npy')
X_std = np.load(f'{run_folder}/fold_{fold}/X_std.npy')
# Y_pred_te_ = np.load(f'{run_folder}/fold_{fold}/Y_pred_te.npy')

model = get_model(run, fold, folder)
model.summary()

## get gaussian model

pert_folder = 'GApCPR/t4'
pert_run = get_completed(pert_folder)['2']
pert_run_folder = f"{pert_folder}/{pert_run['name']}"

assert (X_mean == np.load(f'{pert_run_folder}/fold_{fold}/X_mean.npy')).all()
assert (X_std == np.load(f'{run_folder}/fold_{fold}/X_std.npy')).all()
full_model = get_model(pert_run, fold, pert_folder)
# split the model
ga_model = tf.keras.Sequential(full_model.layers[:2])
ga_model.summary()


## prepare input

X_norm = (X_te - X_mean)/X_std
print(f'{X_norm.shape = }')

# Y_pred_te = model.predict(X_norm)
# assert np.allclose(Y_pred_te, Y_pred_te_)


## optimal input

physical_mask = X_std != 1
geosep = ut.Reshaper(physical_mask)
gr = GradientRegularizer(weights='sphere', lat=np.array(lat, dtype=np.float32))

area_weights = (np.ones_like(X_std).T * np.cos(np.deg2rad(lat))).T
area_weights = geosep.reshape(area_weights)
area_weights /= np.mean(area_weights)
area_weights = geosep.inv_reshape(area_weights)
area_weights = tf.convert_to_tensor(area_weights, dtype=tf.float32)

reg_kwargs = dict(l1coef=0, l2coef=100, target_l2=0.7, rough_coef=0.1, target_roughness=28)
oir_kwargs = dict(maxiter=400, lr=0.02, ori_coef=30, orth_coef=10)

reg = oi.Regularizer(gradient_regularizer=gr, area_weights=area_weights, **reg_kwargs)
oir = OrthOptimalInput(lambda x: model(x)[...,0], reg, orth_model = lambda x: ga_model(x)[...,0], physical_mask=physical_mask, weights=area_weights, **oir_kwargs)


## compute optimal input for all* data
time_indices = np.arange(X_norm.shape[0], dtype=int)[::4]
subset = X_norm[time_indices]

batch_size = 64
nbatches = subset.shape[0]//batch_size
if nbatches*batch_size < subset.shape[0]:
    nbatches += 1
all_info = None
all_optim = []
for b in range(nbatches):
    print(f'batch {b+1}/{nbatches}')
    all_optim.append(oir(subset[batch_size*b:batch_size*(b+1)]) - subset[batch_size*b:batch_size*(b+1)])
    oir.info['ga_output'] = ga_model(oir.info['input'])[...,0].numpy()
    if all_info is None:
        all_info = {k:v for k,v in oir.info.items() if k != 'input'}
    else:
        for k in all_info:
            if k == 'input':
                continue
            all_info[k] = np.concatenate([all_info[k], oir.info[k]], axis=0)

all_optim = np.concatenate(all_optim, axis=0)

# create xarray dataset
optim_da = xr.DataArray(all_optim, coords={
    'time': time_indices,
    'lat': lat,
    'lon': lon,
    'field': np.arange(X_norm.shape[-1], dtype=int)
})

ds = xr.Dataset({k: xr.DataArray(v, coords={'time': time_indices}) for k,v in all_info.items()})

ds['delta_input'] = optim_da

ds.attrs = {**reg_kwargs, **oir_kwargs}

print('Saving to netcdf')
ds.to_netcdf(dataset_filename)

end_time = time.time()

print(f'{start_time = }, {end_time = }, run_time = {ut.pretty_time(end_time - start_time)}')