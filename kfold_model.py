import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from os import listdir
import json
from os import path as ospath

# a simple custom activation
def mapping_to_target_range(x, target_min=0, target_max=500):
    from tensorflow.keras import backend as BK
    x02 = BK.tanh(x)+1 # x in range(0,2)
    scale = (target_max-target_min)/2.
    return  x02*scale+target_min

def graph(histories, tight_layout=False, holdoutscores=None):
    fig, axs = plt.subplots(nrows=1, ncols=len(histories), figsize=(20, 5))
    n = 0
    for ax, kmodel in zip(axs, histories.items()):
        epochs = range(1, kmodel[1].params['epochs']+1)
        ax.plot(epochs, kmodel[1].history['loss'])
        ax.plot(epochs, kmodel[1].history['val_loss'])
        if holdoutscores!=None: ax.axhline(y=holdoutscores[n], color='r', linestyle='-')
        ax.set_yscale('log')
        ax.set_ylabel('loss')
        ax.set_xlabel('epochs')
        ax.legend(['loss', 'val_loss', 'holdoutscore'], loc='upper left')
        if holdoutscores!=None: ax.set_title(f'{kmodel[0]} model loss, holdout: {round(holdoutscores[n], 2)}')
        else: ax.set_title(f'{kmodel[0]} model loss')
        n += 1
    if tight_layout==True: plt.tight_layout()
    plt.show()

def get_callbacks():
    return [callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=100, min_lr=0.0001, verbose=1),
            callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=0, mode='auto', baseline=None, restore_best_weights=False)]

def model_duilder_kfold(xdim, ydim, model='v1', num_folds=5, loss_function='mean_squared_error',
                        optimizer='adam', reshuffle=False):
    # define the model architecture
    for k in range(0, num_folds):
        #shallow model
        if model == 'v1':
            model = Sequential()
            model.add(Dense(270, activation='relu', input_dim=xdim))
            model.add(Dropout(0.2))
            model.add(Dense(135, activation='relu'))
            model.add(Dropout(0.1))
            #model.add(Dense(1, activation=mapping_to_target_range))
            #model.add(Lambda(lambda x: x * [2]))
            model.add(Dense(ydim))
        
        # deep model
        elif model == 'v2':
            model = Sequential([
            Dense(2048, kernel_regularizer=regularizers.l2(0.001),
                        activation='relu', input_dim=xdim),
            Dropout(0.2),
            Dense(1024, kernel_regularizer=regularizers.l2(0.001),
                        activation='relu'),
            Dropout(0.1),
            Dense(512, kernel_regularizer=regularizers.l2(0.001),
                        activation='relu'),
            Dropout(0.01),
            Dense(ydim)])
        model.compile(loss = loss_function, optimizer = optimizer)
        # save model
        path = '_Kfolds_models'
        model.save(f'./{path}/k{k+1}_model.h5')
    return path

def model_saver_kfold(models):
    # load saved model from file
    val = input("Saver trained models (y/n): ")
    if val.startswith(('Y', 'y')):
        for model_name, model in models.items():
            # save model
            model.save(f'./_Kfolds_models/{model_name}.h5')

def model_loader_kfold(path=None, ext=False):
    try:
        models = {}
        for filename in listdir(path):
            if ext==False: name = filename.rpartition('.')[0].replace(' ', '_')
            else: name =filename.replace(' ', '_')
            if filename.endswith('.h5'):
                models[name] = load_model(path+'\\'+filename)
        return models
    except:
        print ('No models loaded for \\',path)

def model_kfold(x, y, holdout=None, model='v1', num_folds=5, batch_size=None, steps_per_epoch=20, loss_function='mean_squared_error', optimizer='adam', max_epochs=100, verbosity=1, workers=6,
                use_multiprocessing=True, continue_training=False, save_models_afte_training=True, plot_results=True, reshuffle=False, random_state=42, path=None):
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
    #dump ksets to json
    

    #loading previous ksets oder
    if ospath.exists(f'./{path}/ksets.json')==True and reshuffle==False:
        with open(f'./{path}/ksets.json') as f: ksets = json.load(f)

    # create fesh modelsn
    if continue_training==False:
        path = model_duilder_kfold(x.shape[1], len(y.shape), model=model, num_folds=num_folds, loss_function=loss_function,
                            optimizer=optimizer)
        if reshuffle==False:
            with open(f'./{path}/ksets.json', 'w') as fp: json.dump(ksets, fp)

    #load models
    models = model_loader_kfold(path=path, ext=False)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    holdoutscores =[]
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
        # generate generalization metrics
        if holdout!= None:
            hscore = models[model_key].evaluate(holdout[0], holdout[1], verbose=0)
            print(f'Holdout score for fold {fold_no}: {models[model_key].loss}: {hscore}')
            holdoutscores.append(hscore)
        # increase fold number
        fold_no = fold_no + 1
    
    # plot results
    if plot_results==True: graph(histories, tight_layout=True, holdoutscores=holdoutscores)

    # save trained models
    if save_models_afte_training==True: model_saver_kfold(models)

    #output trained model
    return models, histories, holdoutscores

def main ():
    # loading train and validation sets in binary format
    train_x = np.load('train_x.npy')
    holdout_x = np.load('holdout_x.npy')
    train_y = np.load('train_y.npy')
    holdout_y = np.load('holdout_y.npy')
    # run kfolds
    models, histories, holdoutscores = model_kfold(train_x, train_y, holdout=[holdout_x,holdout_y])
    pass

if __name__ == "__main__":
    main()

