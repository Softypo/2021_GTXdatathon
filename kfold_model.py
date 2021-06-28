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
from os import listdir
from os import path as ospath
from plotly.subplots import make_subplots

# a simple custom activation
def mapping_to_target_range(x, target_min=0, target_max=500):
    from tensorflow.keras import backend as BK
    x02 = BK.tanh(x)+1 # x in range(0,2)
    scale = (target_max-target_min)/2.
    return  x02*scale+target_min

# graph training result using mathplotlib
def graph_training_mathplot(models, tight_layout=False, holdout_scores=None):
    fig, axs = plt.subplots(nrows=1, ncols=len(models), figsize=(20, 5))
    n = 0
    for ax, kmodel in zip(axs, models.items()):
        epochs = range(1, len(kmodel[1].history['val_loss'])+1)
        ax.plot(epochs, kmodel[1].history['loss'])
        ax.plot(epochs, kmodel[1].history['val_loss'])
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
def graph_training_plotly(histories, holdout_scores=None, width=None, height=None, template='plotly_dark'):
    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=len(histories), subplot_titles=tuple(histories.keys()), shared_yaxes=True)
    c = 1
    if holdout_scores==None: holdout_scores = [None for k in histories.keys()]
    for holdout, kmodel in zip(holdout_scores, histories.items()):
        epochs = list(range(1, len(kmodel[1].history['val_loss'])+1))
        #figures
        fig.add_trace(go.Scatter(x=epochs, y=kmodel[1].history['loss'], name = f'{kmodel[0]} loss', line=dict(color='#64AEEB', width=3)), secondary_y=False, row=1, col=c)

        fig.add_trace(go.Scatter(x=epochs, y=kmodel[1].history['val_loss'], name = f'{kmodel[0]} val_loss', line=dict(color='#2700FF', width=3)), secondary_y=False, row=1, col=c)

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
    fig.show()

def graph_results_plotly(y, predictions, scores=None, width=None, height=None, template='plotly_dark'):
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
    fig.show()

# keras model callbacks
def get_callbacks():
    return [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=250, min_lr=0.00001, verbose=1),
            callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=0, mode='auto', baseline=None, restore_best_weights=False)]

# model builder
def model_builder_kfolds(xdim, ydim, model=None, num_folds=None, activation=None, loss_function=None,
                        optimizer=None, metrics=None):
    # define the model architecture
    for k in range(0, num_folds):
        #shallow model
        if model == 'v1':
            model = Sequential()
            model.add(Dense(270, activation=activation, input_dim=xdim))
            model.add(Dropout(0.2))
            model.add(Dense(135, activation=activation))
            #model.add(LeakyReLU())
            model.add(Dropout(0.1))
            #model.add(Dense(1, activation=mapping_to_target_range))
            #model.add(Lambda(lambda x: x * [2]))
            model.add(Dense(ydim, activation='linear'))
        # deep model
        elif model == 'v2':
            model = Sequential([
            Dense(2048, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0),
                        activation=activation, input_dim=xdim),
            Dropout(0.5),
            Dense(1024, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0),
                        activation=activation),
            Dropout(0.25),
            Dense(512, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0),
                        activation=activation),
            Dropout(0.2),
            Dense(256, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0),
                        activation=activation),
            Dropout(0.1),
            Dense(ydim, activation='linear')])
        # compile model
        model.compile(loss = loss_function, optimizer = optimizer, metrics=metrics)
        # save model
        path = '_Kfolds_models'
        model.save(f'./{path}/k{k+1}_model.h5')
    return path

# save model to disk
def model_saver_kfolds(models, histories=None, path=None):
    #setting up working directory
    if path==None: path = '_Kfolds_models'
    # load saved model from file
    val = input("Saver trained models (y/n): ")
    if val.startswith(('Y', 'y')):
        for model_name, model in models.items():
            # save model
            model.save(f'./{path}/{model_name}.h5')
        if histories!=None:
            h = {}
            for model_name, val in histories.items():
                h.update({model_name: {'history':dict(val.history.items()), 'params':dict(val.params.items())}})
            with open(f'./{path}/histories.data', 'wb') as fp: pickle.dump(h, fp)

# loads model from disk
def model_loader_kfolds(path=None, ext=False, num_folds=None):
    try:
        models = {}
        n = 0
        for filename in listdir(path):
            if ext==False: name = filename.rpartition('.')[0].replace(' ', '_')
            else: name =filename.replace(' ', '_')
            if filename.endswith('.h5') and n<num_folds:
                models[name] = load_model(path+'\\'+filename)
                n+=1
        return models
    except:
        print ('No models loaded for \\',path)

# load training histories from disk
def histories_loader_kfolds(path=None):
    try:
        histories = pickle.load(open(f'./{path}/histories.data', 'rb'))
        return histories
    except:
        print ('No models histories for \\',path)

# kfolds implementation for keras
def model_kfolds(x, y, holdout=None, model='v1', num_folds=5, batch_size=None, steps_per_epoch=10,
                activation='relu', loss_function='mean_squared_error', metrics=['mean_absolute_error'], optimizer='adam',
                max_epochs=100, verbosity=1, workers=6, use_multiprocessing=True, continue_training=False, save_models_afte_training=True,
                reshuffle=False, random_state=42, path=None, plot_results=True, plot_width=None, plot_height=None):
    # setting local variables
    if 'histories' not in locals(): histories = {}

    #setting up working directory
    if path==None: path = '_Kfolds_models'

    # define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)
    
    # storing ksets
    ksets = {}
    count = 1
    for test, train in kfold.split(x):
        ksets['k{}'.format(count)] = {}
        ksets['k{}'.format(count)]['train'] = train.tolist()
        ksets['k{}'.format(count)]['test'] = test.tolist()
        count += 1
    print(len(ksets) == num_folds)#assert we have the same number of splits
    
    # create fesh modelsn
    if continue_training==False:
        path = model_builder_kfolds(x.shape[1], len(y.shape), model=model, num_folds=num_folds, activation=activation, loss_function=loss_function,
                            optimizer=optimizer, metrics=metrics)
        if reshuffle==False:
            #dump ksets to json
            with open(f'./{path}/ksets.json', 'w') as fp: json.dump(ksets, fp)

    # load models
    if ospath.exists(f'./{path}')==True:
        models = model_loader_kfolds(path=path, ext=False, num_folds=num_folds)
    # loading previous ksets oder
    if ospath.exists(f'./{path}/ksets.json')==True and reshuffle==False:
        with open(f'./{path}/ksets.json') as f: ksets = json.load(f)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    holdout_prediction =[]
    holdout_scores =[]
    for k, kset in ksets.items():
        model_key = str(f'{k}_model')
        train = kset['train']
        test = kset['test']
        # generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        # running models
        histories[model_key] = models[model_key].fit(x[train], y[train],
                                batch_size = batch_size,
                                steps_per_epoch = steps_per_epoch,
                                epochs = max_epochs,
                                validation_data = (x[test],y[test]),
                                callbacks = get_callbacks(),
                                verbose = verbosity,
                                workers = workers,
                                use_multiprocessing = use_multiprocessing)
        # holdout prediction and metrics by fold
        if holdout!= None:
            hpred = np.squeeze(models[model_key].predict(holdout[0], verbose=0), axis=1).tolist()
            hscore = models[model_key].evaluate(holdout[0], holdout[1], verbose=0)
            print(f'Holdout score for fold {fold_no}: loss-{loss_function}: {hscore[0]}, metrics-{models[model_key].metrics_names[1]}: {hscore[1]}')
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
    if save_models_afte_training==True: model_saver_kfolds(models, histories, path=path)

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
    models, histories, holdout_score = model_kfolds(train_x, train_y, holdout=[holdout_x,holdout_y], max_epochs=2, save_models_afte_training=False, plot_results=False)
    # run k folds predictions
    yhat, yhat_score, predictions, scores = predict_kfolds(holdout_x, holdout_y, models, output=3)
    pass

if __name__ == "__main__":
    main()

