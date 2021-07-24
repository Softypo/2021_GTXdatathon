import numpy as np
from tensorflow.keras import Sequential, optimizers
from tensorflow.keras import metrics as kerasmetrics
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
from keras_tuner import HyperModel
from keras_tuner import HyperParameters
from keras_tuner import Tuner, oracles

class HyperModel_MLP(HyperModel):
    def __init__(self, xdim, ydim, activation_hidden, activation_output, loss_function, metrics=None, hidden_layers=None, units=None, initializers=None, l1_reg=None, l2_reg=None, dropout=None, optimizers=None, learning_rates=None):
        self.xdim = xdim
        self.ydim = ydim
        self.activation_hidden = activation_hidden
        self.activation_output = activation_output
        self.loss_function = loss_function
        self.metrics = metrics
        self.hidden_layers = hidden_layers if hidden_layers!=None else [2, 23]
        self.units = units if units!=None else [32, 1024, 32]
        self.initializers = initializers if initializers!=None else ['constant', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'identity', 'lecun_normal', 'lecun_uniform', 'ones', 'orthogonal', 'random_normal', 'random_uniform', 'truncated_normal', 'variance_scaling', 'zeros']
        self.l1_reg = l1_reg if l1_reg!=None else [0.0, 1e-1, 1e-2, 1e-3, 1e-4]
        self.l2_reg = l2_reg if l2_reg!=None else [0.0, 1e-1, 1e-2, 1e-3, 1e-4]
        self.dropout = dropout if dropout!=None else [0.0, 0.5, 0.05]
        self.optimizers = optimizers if optimizers!=None else ['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'Ftrl', 'Nadam', 'RMSprop']
        self.learning_rates = learning_rates if learning_rates!=None else [1e-2, 1e-3, 1e-4]
        self.hp_space = int((self.hidden_layers[1]+1-self.hidden_layers[0])*((self.units[1]-self.units[0])/self.units[2] if (self.units[1]-self.units[0])/self.units[2]>0 else 1)*(len(self.initializers))*(len(self.l1_reg))*(len(self.l2_reg))*((self.dropout[1]-self.dropout[0])/self.dropout[2] if (self.dropout[1]-self.dropout[0])/self.dropout[2]>0 else 1)*(len(self.learning_rates)))

    def build(self, hp):
        model = Sequential()
        #input layer
        model.add(InputLayer(input_shape=self.xdim))
        #hidden layers
        for i in range(hp.Int('num_hidden_layers', self.hidden_layers[0], self.hidden_layers[1])):
            model.add(
                Dense(
                    units=hp.Int(f'units_{str(i)}', min_value=self.units[0], max_value=self.units[1], step=self.units[2]),
                    kernel_initializer=hp.Choice(f'kernel_initializer_{str(i)}', self.initializers),
                    kernel_regularizer=L1L2(l1=hp.Choice(f'l1_{str(i)}', self.l1_reg), l2=hp.Choice(f'l2_{str(i)}', self.l2_reg)),
                    activation=self.activation_hidden,
                )
            )
            model.add(
                Dropout(rate=hp.Float(f'dropout_{str(i)}', min_value=self.dropout[0], max_value=self.dropout[1], step=self.dropout[2]))
            )
        # output layer
        model.add(Dense(self.ydim, activation=self.activation_output))
        # compiling model
        model.compile(
            loss = self.loss_function,
            optimizer = getattr(optimizers, hp.Choice('optimizer', self.optimizers))(hp.Choice('learning_rate', self.learning_rates)),
            metrics=self.metrics
        )
        return model

def two_steps(train_x, train_y, holdout_x, holdout_y, project_name, path=None, oracle_1st='random', oracle_2nd='bayesian', epochs_1st=50, epochs_2nd=5, max_trials_1st=None, max_trials_2nd=None, seed=42, activation_hidden=None, activation_output=None, loss_function=None, metrics=None, hidden_layers=None, units=None, initializers=None, l1_reg=None, l2_reg=None, dropout=None, optimizers=None, learning_rates=None):
    
    path = path if path!=None else '.kerastunner'
    hypermodel_mlp_init_op = HyperModel_MLP((train_x.shape[1], ), len(train_y.shape), activation_hidden=activation_hidden, activation_output=activation_output, loss_function=loss_function, metrics=metrics, hidden_layers=[1,1], units=[32,32,32], initializers=initializers, l1_reg=[0.0], l2_reg=[0.0], dropout=[0.0,0.0,0.5], optimizers=optimizers, learning_rates=[0.001])
    hypermodels = (('first', oracle_1st, hypermodel_mlp_init_op, epochs_1st, max_trials_1st), ['second', oracle_2nd, None, epochs_2nd, max_trials_2nd])

    for hyperm in hypermodels:
        epochs = hyperm[3]
        max_trials = hyperm[4] if hyperm[4]!=None else hyperm[2].hp_space
        if hyperm[1]=='random':
            #oracle random search instance
            oracle = oracles.RandomSearchOracle(
                objective=hyperm[2].metrics[0],
                max_trials=max_trials,
                seed=seed,
                hyperparameters=None,
                allow_new_entries=True,
                tune_new_entries=True,
                )
        elif hyperm[1]=='bayesian':
            #oracle bayesian search instance
            oracle = oracles.BayesianOptimizationOracle(
                objective=hyperm[2].metrics[0],
                max_trials=max_trials,
                num_initial_points=None,
                alpha=0.0001,
                beta=2.6,
                seed=seed,
                hyperparameters=None,
                allow_new_entries=True,
                tune_new_entries=True
            )   
        elif hyperm[1]=='hyperband':
            #oracle hyperband search instance
            oracle = oracles.HyperbandOracle(
                objective=hyperm[2].metrics[0],
                max_epochs=epochs,
                factor=3,
                hyperband_iterations=1,
                seed=seed,
                hyperparameters=None,
                allow_new_entries=True,
                tune_new_entries=True
            )
        
        #tunner instance
        tuner = Tuner(
            oracle,
            hyperm[2],
            max_model_size=None,
            optimizer=None,
            loss=None,
            metrics=None,
            distribution_strategy=None,
            directory=path,
            project_name=f'{project_name}_{hyperm[0]}_{hyperm[1]}',
            logger=None,
            tuner_id=f'{hyperm[0]}_{hyperm[1]}',
            overwrite=True
            )
        
        print (f'hp_space: {hyperm[2].hp_space}')

        tuner.search(
            train_x, train_y,
            epochs=epochs,
            validation_data=(holdout_x, holdout_y),
            # Use the TensorBoard callback.
            # The logs will be write to "/tmp/tb_logs".
            callbacks=[TensorBoard(f'./{path}/{project_name}_{hyperm[0]}_{hyperm[1]}/tb_logs', histogram_freq=1, update_freq='epoch')]
            )

        tuner.results_summary(num_trials=1)
        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.hypermodel.build(best_hp)
        best_model.save(f'./{path}/{project_name}_{hyperm[0]}_{hyperm[1]}/{hyperm[0]}_step_{hyperm[1]}_best_model.h5')
        best_model.summary()
        
        if hyperm[0]=='first':
            #hp = HyperParameters()
            hypermodel_mlp_params = HyperModel_MLP((train_x.shape[1], ), len(train_y.shape), activation_hidden=activation_hidden, activation_output=activation_output, loss_function=loss_function, metrics=metrics, hidden_layers=hidden_layers, units=units, initializers=[best_hp.values['kernel_initializer_0']], l1_reg=l1_reg, l2_reg=l2_reg, dropout=dropout, optimizers=[best_hp.values['optimizer']], learning_rates=learning_rates)
            hypermodels[1][2] = hypermodel_mlp_params
            first_tuner = tuner
        else: second_tuner = tuner

    return first_tuner, second_tuner, best_model


def main ():
    # loading train and validation sets in binary format
    train_x = np.load('train_x.npy')
    holdout_x = np.load('holdout_x.npy')
    train_y = np.load('train_y.npy')
    holdout_y = np.load('holdout_y.npy')

    first_tuner, second_tuner, best_model = two_steps(train_x, train_y, holdout_x, holdout_y, project_name='GTX_dataset', path=None, oracle_1st='random', oracle_2nd='bayesian', epochs_1st=15, epochs_2nd=5, max_trials_1st=None, max_trials_2nd=69, seed=42, activation_hidden='relu', activation_output='linear', loss_function='mean_squared_error', metrics=['mean_squared_error'], hidden_layers=None, units=None, initializers=['constant', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'identity', 'lecun_normal', 'lecun_uniform', 'orthogonal', 'random_normal', 'random_uniform', 'truncated_normal', 'variance_scaling', 'zeros'], l1_reg=[0.0], l2_reg=[0.0], dropout=None, optimizers=None, learning_rates=None)

if __name__ == "__main__":
    main()