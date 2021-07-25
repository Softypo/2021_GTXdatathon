import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import json
import pickle
from sklearn import metrics as skmetrics
from tensorflow import function as tfunc
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
import tensorflow.keras.losses as klosses
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from os import listdir, remove
from os.path import isfile, islink, isdir
from shutil import rmtree
from plotly.subplots import make_subplots

# a simple custom activation
def mapping_to_target_range(x, target_min=0, target_max=500):
    from tensorflow.keras import backend as BK
    x02 = BK.tanh(x)+1 # x in range(0,2)
    scale = (target_max-target_min)/2.
    return  x02*scale+target_min

# graph training result using mathplotlib
def graph_training_mathplot(histories, tight_layout=False, holdout_scores=None):
    fig, axs = plt.subplots(nrows=1, ncols=len(histories), figsize=(20, 5))
    n = 0
    for ax, kmodel in zip(axs, histories.items()):
        epochs = range(1, kmodel[1]['params']['epochs']+1)
        ax.plot(epochs, kmodel[1]['history']['loss'])
        ax.plot(epochs, kmodel[1]['history']['val_loss'])
        if holdout_scores!=None: ax.axhline(y=holdout_scores[n][0], color='r', linestyle='-')
        ax.set_yscale('log')
        ax.set_ylabel('loss')
        ax.set_xlabel('epochs')
        ax.legend(['loss', 'val_loss', 'holdout_loss'], loc='upper right')
        if holdout_scores!=None: ax.set_title(f'{kmodel[0]}, holdout loss: {round(holdout_scores[n][0], 2)}')
        else: ax.set_title(f'{kmodel[0]} model loss')
        n += 1
    fig.suptitle(f'Kfolds training results -> holdout mean loss: {round(holdout_scores[-1][0], 2)}, mean first metric:{round(holdout_scores[-1][1], 2)}')
    if tight_layout==True: plt.tight_layout()
    plt.show()

# graph training result using plotly
def graph_training_plotly(histories, holdout_scores=None, width=None, height=None, template='plotly_dark', renderer=None):
    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=len(histories), subplot_titles=tuple(histories.keys()), shared_yaxes=True)
    c = 1
    if holdout_scores==None: holdout_scores = [None for k in histories.keys()]
    for holdout, kmodel in zip(holdout_scores, histories.items()):
        epochs = list(range(1, kmodel[1]['params']['epochs']+1))
        #figures
        fig.add_trace(go.Scatter(x=epochs, y=kmodel[1]['history']['loss'], name = f'{kmodel[0]} loss', line=dict(color='#64AEEB', width=3)), secondary_y=False, row=1, col=c)

        fig.add_trace(go.Scatter(x=epochs, y=kmodel[1]['history']['val_loss'], name = f'{kmodel[0]} val_loss', line=dict(color='#2700FF', width=3)), secondary_y=False, row=1, col=c)

        if holdout!=None:
            fig.add_hline(y=holdout[0], line_width=3, line_dash="dash", line_color='#FF5733', row=1, col=c)
            # text annotation
            fig.add_annotation(xref='x domain', yref='y domain', x=0.02, y=0.04,
                text=f'holdout loss: {round(holdout[0], 2)}',
                showarrow=False,
                row=1, col=c)
            fig.add_annotation(xref='x domain', yref='y domain', x=0.02, y=0.0,
                text=f'holdout metric: {round(holdout[1], 2)}',
                showarrow=False,
                row=1, col=c)
        # Axes
        fig.update_traces(opacity=0.6)
        fig.update_layout(hovermode='x')
        fig.update_yaxes(type="log")
        fig.update_yaxes(showgrid=True)
        c += 1

    # Add titles
    fig.update_layout(
        title=f'Kfolds training results:',
        yaxis_title='loss',
        xaxis_title='epochs',
        xaxis_rangeslider_visible=False,
        template=template)
    if holdout_scores[0]!=None:
        fig.update_layout(title=f'Kfolds training results -> holdout mean loss: {round(holdout_scores[-1][0], 2)}, mean first metric:{round(holdout_scores[-1][1], 2)}')
    if isinstance(width, int) and isinstance(height, int): fig.update_layout(width=width, height=height)
    fig.show(renderer=renderer)

def graph_results_plotly(y, predictions, scores=None, width=None, height=None, template='plotly_dark', renderer=None):
    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=len(predictions), subplot_titles=tuple(predictions.keys()), shared_yaxes=True)
    c = 1
    if scores==None: scores = {k:None for k in predictions.keys()}
    for score, yhat in zip(scores.items(), predictions.items()):
        # poly1d_fn is a function which takes in yhat values and returns an estimate for y
        coef = np.polyfit(yhat[1], y,1)
        poly1d_fn = np.poly1d(coef)
        #point error
        error = ['point error: '+str(round(abs(yy-yh), 2)) for yy, yh in zip(y, yhat[1])]
        #figures
        fig.add_shape(type="line", x0=0, y0=0, x1=max(y), y1=max(y), line=dict(color="#2700FF", width=3, dash="dashdot",), row=1, col=c)

        fig.add_trace(go.Scatter(x=yhat[1], y=y, name = f'{yhat[0]}', text=error, mode='markers'), secondary_y=False, row=1, col=c)

        fig.add_trace(go.Scatter(x=yhat[1], y=poly1d_fn(yhat[1]), name = f'{yhat[0]} fit', line=dict(color='#FF5733', width=3)), secondary_y=False, row=1, col=c)

        if scores!=None:
            # text annotation
            fig.add_annotation(xref='x domain', yref='y domain', x=0.02, y=1.0,
                text=f'loss: {round(score[1][0], 2)}',
                showarrow=False,
                row=1, col=c)
            fig.add_annotation(xref='x domain', yref='y domain', x=0.02, y=0.96,
                text=f'metric: {round(score[1][1], 2)}',
                showarrow=False,
                row=1, col=c)
        # Axes
        fig.update_traces(opacity=0.6)
        fig.update_yaxes(showgrid=True)
        c += 1

    # Add titles
    fig.update_layout(
        title=f'Kfolds prediction results:',
        yaxis_title='target',
        xaxis_title='predictions',
        xaxis_rangeslider_visible=False,
        template=template)
    if isinstance(width, int) and isinstance(height, int): fig.update_layout(width=width, height=height)
    fig.show(renderer=renderer)

# keras model callbacks
def get_callbacks(monitor='val_loss', lr_patience=250, factor=0.1, min_lr=0.000001, stop_patience=1000, min_delta=0, mode='auto', baseline=None, restore_best_weights=False):
    return [callbacks.ReduceLROnPlateau(monitor=monitor, factor=factor, patience=lr_patience, min_lr=min_lr, verbose=1),
            callbacks.EarlyStopping(monitor=monitor, min_delta=min_delta, patience=stop_patience, verbose=1, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights)]

# model builder
def model_builder_kfolds(model, x, num_folds, random_state=42, shuffle=True):
    #setting up model
    models = {}
    n = 0
    if isinstance(model, str):
        try:
            for k in range(0, num_folds):
                models[f'k{k+1}_model'] = load_model(model)
                n+=1
        except: print ('No model found for \\',model)
    elif isinstance(model, object):
        try:
            for k in range(0, num_folds):
                models[f'k{k+1}_model'] = model
                n+=1
        except: print ('No model found for \\',model)
    #setting up ksets
    try:
        # define the K-fold Cross Validator
        kfold = KFold(n_splits=num_folds, random_state=random_state, shuffle=shuffle)
        
        # storing ksets
        ksets = {}
        count = 1
        for test, train in kfold.split(x):
            ksets['k{}'.format(count)] = {}
            ksets['k{}'.format(count)]['train'] = train.tolist()
            ksets['k{}'.format(count)]['test'] = test.tolist()
            count += 1
    except: print ('error during ksets creation')
    return models, ksets

def remove(path):
    # param <path> could either be relative or absolute
    if isfile(path) or islink(path):
        remove(path)  # remove the file
    elif isdir(path):
        rmtree(path)  # remove dir and all contains
    else:
        raise ValueError(f'file {path} is not a file or dir.')

# save models to disk
def model_saver_kfolds(models, histories=None, ksets=None, path='_Kfolds_models', save_models_after_training=False):
    if save_models_after_training==True: val = 'Y'
    else: val = input("Save results (y/n): ")
    if val.startswith(('Y', 'y')):
        # remove previous saved files
        remove(path)
        # save new files
        try:
            for model_name, model in models.items():
                # save models
                model.save(f'./{path}/{model_name}.h5')
            if histories!=None:
                # save histories
                with open(f'./{path}/histories.data', 'wb') as fp: pickle.dump(histories, fp)
            if ksets!=None:
                # dump ksets to json
                with open(f'./{path}/ksets.json', 'w') as fp: json.dump(ksets, fp)
        except Exception as e:
            print ("Exception: %s." % (e))

# loads models from disk
def model_loader_kfolds(path='_Kfolds_models'):
    try:
        models = {}
        n = 0
        for filename in listdir(path):
            name = filename.rpartition('.')[0].replace(' ', '_')
            if filename.endswith('.h5'):
                models[name] = load_model(path+'\\'+filename)
                n+=1
        return models
    except:
        print ('No models loaded for \\',path)

# load training histories from disk
def histories_loader_kfolds(path='_Kfolds_models/histories.data'):
    try:
        histories = pickle.load(open(f'./{path}', 'rb'))
        return histories
    except:
        print ('No histories loaded for \\',path)

# load ksets from disk
def ksets_loader_kfolds(path='_Kfolds_models/ksets.json'):
    try:
        with open(f'./{path}') as f:
            ksets = json.load(f)
        return ksets
    except:
        print ('No ksets loaded for \\',path)

# kfolds implementation for keras
def model_kfolds(model, x, y, holdout=None, num_folds=5, shuffle=True, random_state=42, batch_size=None, steps_per_epoch=10, max_epochs=100, ind_epochs=None, monitor='val_loss', lr_patience=250, factor=0.2, min_lr=0.000001, stop_patience=1000, min_delta=0, mode='auto', baseline=None, restore_best_weights=False, verbosity=1, workers=6, use_multiprocessing=True, continue_training=False, save_models_after_training=False, plot_results=True, plot_width=None, plot_height=None, path='_Kfolds_models'):
    
    # setting local variables
    path = path if path!=None else '_Kfolds_models'
    if continue_training==True:
        # load models
        models = model_loader_kfolds(path=path)
        # load ksets oder
        ksets = ksets_loader_kfolds(path=f'{path}/ksets.json')
        # load histories
        histories = histories_loader_kfolds(path=f'{path}/histories.data')
    else:
        models, ksets = model_builder_kfolds(model=model, x=x, num_folds=num_folds, random_state=random_state, shuffle=shuffle)
        histories = {}

    # K-fold Cross Validation model evaluation
    fold_no = 1
    h = {}
    ind_epochs = [max_epochs for k in range(0, num_folds)] if ind_epochs==None else ind_epochs
    holdout_prediction =[]
    holdout_scores =[]
    for k, kset in ksets.items():
        if ind_epochs[fold_no-1]!=None: epochs = ind_epochs[fold_no-1] if ind_epochs[fold_no-1]>=1 and ind_epochs[fold_no-1]<max_epochs else max_epochs
        else: epochs = None
        model_key = str(f'{k}_model')
        train = kset['train']
        test = kset['test']
        # generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        # running models
        if epochs!=None:
            h[model_key] = models[model_key].fit(x[train], y[train],
                                                batch_size = batch_size,
                                                steps_per_epoch = steps_per_epoch,
                                                epochs = epochs,
                                                validation_data = (x[test],y[test]),
                                                callbacks = get_callbacks(monitor=monitor, lr_patience=lr_patience, factor=factor, min_lr=min_lr, stop_patience=stop_patience,
                                                                        min_delta=min_delta, mode=mode, baseline=baseline, restore_best_weights=restore_best_weights),
                                                verbose = verbosity,
                                                workers = workers,
                                                use_multiprocessing = use_multiprocessing)
            histories.update({model_key: {'history':dict(h[model_key].history.items()), 'params':dict(h[model_key].params.items())}})
        # holdout prediction and metrics by fold
        if holdout!= None:
            hpred = np.squeeze(models[model_key].predict(holdout[0], verbose=0), axis=1).tolist()
            hscore = models[model_key].evaluate(holdout[0], holdout[1], verbose=0)
            print(f'Holdout score for fold {fold_no}: loss: {hscore[0]}, metrics-{models[model_key].metrics_names[1]}: {hscore[1]}')
            holdout_prediction.append(hpred)
            holdout_scores.append(hscore)
        # increase fold number
        fold_no = fold_no + 1
    # holdout prediction and metrics overall
    if holdout!= None:
        yhat = np.array(holdout_prediction).mean(axis=0).tolist()
        holdout_prediction.append(yhat)
        holdout_scores.append([float(models[model_key].loss(holdout[1], yhat)) if metric=='loss' else float(getattr(klosses, metric)(holdout[1], yhat)) for metric in models[model_key].metrics_names])

    # plot results
    if plot_results==True: graph_training_plotly(histories, width=plot_width, height=plot_height, holdout_scores=holdout_scores)

    # save trained models
    model_saver_kfolds(models, histories, ksets, path=path, save_models_after_training=save_models_after_training)

    if holdout!= None:
        #output trained model with holdoutscores
        return models, histories, holdout_scores
    else:
        #output trained model
        return models, histories

# kfolds predictor function
def predict_kfolds(x, y, models, todrop=None, output=None, plot_results=True, plot_width=None, plot_height=None):
    # calculating predictions
    predictions = {}
    scores = {}
    if todrop==None: todrop = []
    for model_name, model in models.items():
        if model_name not in todrop:
            predictions.update({model_name:np.squeeze(model.predict(x), axis=1).tolist()})
            scores.update({model_name:model.evaluate(x, y)})
    
    yhat = np.array(list(predictions.values())).mean(axis=0)
    predictions.update({'kfolds_mean':yhat.tolist()})
    
    yhat_score = [float(model.loss(y, yhat)) if metric=='loss' else float(getattr(klosses, metric)(y, yhat)) for metric in model.metrics_names]
    scores.update({'kfolds_mean':yhat_score})

    #output predictions and selected yhat
    if isinstance(output, int):
        yhat = predictions[f'k{output}_model']
        yhat_score = scores[f'k{output}_model']
    else:
        yhat = predictions['kfolds_mean']
        yhat_score = scores['kfolds_mean']
    
    # plot results
    if plot_results==True: graph_results_plotly(y, predictions, scores=scores, width=plot_width, height=plot_height, template='plotly_dark')

    #output prediction results
    return yhat, yhat_score, predictions, scores

def main ():
    # loading train and validation sets in binary format
    train_x = np.load('train_x.npy')
    holdout_x = np.load('holdout_x.npy')
    train_y = np.load('train_y.npy')
    holdout_y = np.load('holdout_y.npy')
    
    # run kfolds models
    models, histories, holdout_score = model_kfolds(model='.\.kerastunner\GTX_dataset_second_bayesian\second_step_bayesian_best_model.h5', x=train_x, y=train_y, holdout=[holdout_x,holdout_y], num_folds=5, shuffle=True, random_state=42, batch_size=None, steps_per_epoch=10, max_epochs=10000, ind_epochs=None, monitor='val_loss', lr_patience=250, factor=0.1, min_lr=0.000001, stop_patience=1000, min_delta=0, mode='auto', baseline=None, restore_best_weights=True, verbosity=1, workers=6, use_multiprocessing=True, continue_training=False, save_models_after_training=False, plot_results=True, plot_width=None, plot_height=None, path=None)
    
    # [600,100,200,175,250,69,575,450,400,600]
    # [300,100,200,175,250,69,575,165,127,600]

    models = model_loader_kfolds()

    # run k folds predictions
    yhat, yhat_score, predictions, scores = predict_kfolds(holdout_x, holdout_y, models, todrop=None, output=None, plot_results=True, plot_width=None, plot_height=None)
    
    pass

if __name__ == "__main__":
    main()

