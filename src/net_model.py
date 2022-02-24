import sys
import os
from string_utils import *
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchsummary import summary
import pandas as pd
from string_utils import *
import argparse
import subprocess
import json
import copy
from collections import Counter as C


class BinaryClassification(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=[11], output_dim=1):
        """ A Neural Network which can be adaptively parameterised for tabulra data predictions.

        :param input_dim: the number of input units (features), defaults to 12
        :type input_dim: int, optional.
        
        :param hidden_dim: each element value gives the number of neurons per layer,
                           while the len(hidden_sim) signifies the number of layers, defaults to [64]
        :type hidden_dim: list, optional.
        
        :param output_dim: the number of outputs expected (1 for binary classification), defaults to 1
        :type output_dim: int, optional.
        """
        super(BinaryClassification, self).__init__()       
        
        # Define field attributes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()


        # If we are only interested in a network with a single hidden layer
        if len(self.hidden_dim) == 1:
          self.hidden_dim = hidden_dim[0]
          self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
          self.relu_1 = nn.ReLU()
          self.batch_norm_1 = nn.BatchNorm1d(self.hidden_dim)
          self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim) 
          self.relu_2 = nn.ReLU()
          self.batch_norm_2 = nn.BatchNorm1d(self.hidden_dim)
          self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
          
          # Chain all layers together (except output)
          all_layers = [self.input_layer, self.relu_1, self.batch_norm_1,
                        self.hidden_layer, self.relu_2, self.batch_norm_2]
          self.layers += [x for x in all_layers]
 
        elif len(self.hidden_dim) > 1:
          self.chained_dims = pairwise([self.hidden_dim[0]] + self.hidden_dim)
          self.input_layer = nn.Linear(self.input_dim, self.hidden_dim[0])
          self.relu_1 = nn.ReLU()
          self.batch_norm_1 = nn.BatchNorm1d(self.hidden_dim[0])
          self.output_layer = nn.Linear(self.hidden_dim[-1], self.output_dim)
          
          # Chain all hidden_dims sequentially - add to ModuleList
          self.layers.append(self.input_layer)
          self.layers.append(self.relu_1)
          self.layers.append(self.batch_norm_1)

          for _, (h_in, h_out) in enumerate(self.chained_dims):
            hidden_layer = nn.Linear(h_in, h_out)
            self.layers.append(hidden_layer)
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(h_out))
          
        # Activation and relu layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.02)
       
    def forward(self, inputs):
      x = inputs
      for layer in self.layers:
        x = layer(x)
      
    #   x = self.dropout(x)
      x = self.output_layer(x)
      return x

def train_network(params, x_train, y_train):
    
    # Grab parameters
    epochs = params['epochs']
    criterion = params['criterion']
    net = params['net']
    optimizer = params['optimizer']
    to_shuffle = params['to_shuffle']

    balance = False
    if balance:
        d_train = dict(C(y_train.values))
        x_train['labels'] = y_train.values
        x_train = x_train.groupby('labels').apply(lambda x: x.sample(n=d_train[1], replace=False)).reset_index(drop=True)
        y_train = x_train['labels']
        x_train.drop(columns='labels', inplace=True)
    
    # Generate train tensors
    y_tensor_train = torch.FloatTensor(y_train.values)
    x_tensor_train = torch.FloatTensor(x_train.values)

    # Establish train data-loader
    to_shuffle = True
    train_tensor = data_utils.TensorDataset(x_tensor_train, y_tensor_train)
    train_loader = data_utils.DataLoader(train_tensor, batch_size=len(y_train), shuffle=to_shuffle)
    
    # Train loop
    net.train()
    for e in range(1, epochs+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch, y_batch
            optimizer.zero_grad()
            
            y_pred = net(X_batch)
            
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
    return net

def predict(net, x_test, y_test):

    # Establish train data-loader
    to_shuffle = True
        # Generate train tensors
    y_tensor_train = torch.FloatTensor(y_test.values)
    x_tensor_train = torch.FloatTensor(x_test.values)
        
    # Create the data loader
    to_shuffle = False
    test_tensor = data_utils.TensorDataset(x_tensor_train, y_tensor_train)
    test_loader = data_utils.DataLoader(test_tensor, batch_size=len(y_test), shuffle=to_shuffle)


    # Make predicitions
    y = []
    y_hat = []
    y_probas = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            # calculate outputs by running images through the network
            logits = net(inputs)
            probas = torch.sigmoid(logits)
            outputs = torch.round(probas)
            y_hat.append(outputs.numpy())
            y.append(labels.numpy())
            y_probas.append(probas.numpy())

    y_hat = [a.squeeze().tolist() for a in y_hat][0]
    y = [a.squeeze().tolist() for a in y][0]
    return net, y, y_hat, y_probas

def run_pipeline(x, params, scale=False, weights=None, 
                cogs=True, train_ratio=0.8, noise=False, n_runs=3):
                
    """Runs the entire modeling process, pre-processing has now been migrated to src/pre_process.py.

    :param data: x-data containing 'labels' and 'cogs' columns
    :type data: pandas DataFrame object

    :param params: model hyper-parameter dictionary
    :type param: dict

    :param scale: if True,  scales inputs in range [0,1], defaults to False
    :type scale: bool, optional

    :param weights: if provided, upscales the positive class importance during training, defaults to None
    :type weights: float, optional

    :param cogs: if True, train and test are split on COG observations, defaults to True
    :type cogs: bool, optional

    :param train_ratio: the proportion of data used for training, defaults to 0.8
    :type train_ratio: float, optional

    :param noise: if True, injects noise term to specified features, defaults to False
    :type noise: bool, optional
    
    :return: Returns an output dict containing key information.
    :rtype: dict
    """

    print("Beginning pipeline...")
    test_ratio = 1-train_ratio

    # Split the data
    train_splits  = []
    test_splits = []
    models = []
    predictions = []
    probabilities = []
    accuracies = []

    # Pre-allocate the datasets
    for i in range(1, n_runs+1):

        if cogs:
            # Stratify data on the ortholog groups
            print('Generating COG splits for sampling run {}'.format(i))
            x_train, x_test = split_on_cogs_alt(x=x, test_size=test_ratio)

            # Shuffle the data
            x_train = x_train.sample(frac=1)
            x_test = x_test.sample(frac=1)

            # Split on labels
            y_train = x_train.labels
            y_test = x_test.labels

        else:
            # Don't stratify on orthologs and sample uniformly
            x_train, x_test, y_train, y_test = model_splits(
                x, x.labels, test_ratio=test_ratio)   

        # Drop the labels from x-train and x-test
        x_train.drop(columns=['labels', 'cogs'], inplace=True)
        x_test.drop(columns=['labels', 'cogs'], inplace=True)

        # Store all of the unique splits
        train_splits.append([x_train, y_train])
        test_splits.append([x_test, y_test])

    # CML message
    print("Complete with no errors")
    print('Done\n')


    # Train across n-unique subsets of the data
    for i in range(len(train_splits)):
        print("Training on sampling run {}".format(i+1))
        x_train, y_train = train_splits[i]
        x_test, y_test = test_splits[i]
        
        # Scale the data if necessary
        if scale:
            col_names = x_train.columns
            x_train, x_test, mms = scale_features(x_train, x_test, method='standard')
            x_train.columns = col_names
            x_test.columns = col_names

        if noise:
            # Add normally distributed noise to following features
            perturb = [
                'neighborhood_transferred',
                'experiments_transferred',
                'textmining',
                'textmining_transferred',
                'experiments',
                'experiments_transferred',
                'coexpression_transferred']

            # Define guassian noise argumnets
            mu = 0
            sigma = 0.005

            x_train = x_train.apply(lambda x: inject_noise(
                x, mu=mu, sigma=sigma) if x.name in perturb else x)

            x_test = x_test.apply(lambda x: inject_noise(
                x, mu=mu, sigma=sigma) if x.name in perturb else x)
        

        # Make a one time prediction for each of the splits
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        output_size = params['output_size']
    
        net = BinaryClassification(input_dim=input_size, hidden_dim=[hidden_size], output_dim=output_size)
        print('Network Architecture: \n',net)
        net = train_network(params=params, x_train=x_train, y_train=y_train)
        params['net'] = net
        print("Predicting on test data")
        net, y, y_hat, y_probas = predict(net=net, x_test=x_test, y_test=y_test)
        
        # Collect the model specific data
        models.append(net)
        probabilities.append(y_probas)
        predictions.append(y_hat)

    output_dict = {
            'probabilities': probabilities,
            'classifier': models,
            'train_splits': train_splits,
            'test_splits': test_splits
            }

    return output_dict

def mean_probas(x_test, y_test, models):

        # Initialise final results (mean_probas)
        n_rows = np.shape(y_test)[0]
        n_cols = len(models)
        mean_probas = np.zeros(shape=(n_rows, n_cols))
        
        # Loop over all models, make predictions across K model, compute average.
        for i in range(len(models)):
            net, y, y_hat, probas = predict(models[i], x_test, y_test)
            mean_probas[:, i] = probas

        # Returns the mean predictions for all K models
        mean_probas = mean_probas.mean(axis=1)
        mean_probas = [(x, x) for x in mean_probas]
        return mean_probas

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

###############################################################################################
# START SCRIPT
###############################################################################################
   
# Extract input variables from Argparse
USE_ARGPASE = True
if USE_ARGPASE:
    parser = argparse.ArgumentParser(description='bambi')
    parser.add_argument('-n', '--model_name', type=str, metavar='',
                        required=True, default='model_0', help='name of the model')

    parser.add_argument('-c', '--cogs', type=str, metavar='',
                        required=True, default=True, help='to split on cogs or not')

    parser.add_argument('-cw', '--class_weight', type=float, metavar='',
                        required=True, default=4, help='factor applied to positive predictions')

    parser.add_argument('-un', '--use_noise', type=str, metavar='',
                        required=True, default=False, help='if True, injects noise to X')

    parser.add_argument('-nr', '--neg_ratio', type=int, metavar='',
                        required=True, default=4, help='factor increase in neg obs compared to pos obs')

    parser.add_argument('-dh', '--drop_homology', type=str, metavar='',
                        required=True, default=True, help='if True, drops homology feature')

    parser.add_argument('-sid', '--species_id', type=str, metavar='',
                        required=True, default='511145 9606 4932', help='ids of species to include sepr=' '')

    parser.add_argument('-o', '--output_dir', type=str, metavar='',
                        required=True, default='benchmark/cog_predictions', help='directory to save outputs to')

    parser.add_argument('-foi', '--use_foi', type=str, metavar='',
                        required=True, default='False', help='make dot-plot on feature of interest')
    
    parser.add_argument('-ns', '--n_runs', type=int, metavar='',
                        required=True, default=3, help='number of randomised samplings')
    
    
    parser.add_argument('-pp', '--pre_process', type=str, metavar='',
                    required=True, default='False', help='to pre-process train and test splits')
    
    
    # Parse agrs
    FORMAT = True
    args = parser.parse_args()
    model_name = args.model_name
    use_cogs = True if args.cogs == 'True' else False
    weights = args.class_weight
    use_noise = True if args.use_noise == 'True' else False
    neg_ratio = args.neg_ratio
    drop_homology = True if args.drop_homology == 'True' else False
    species_id = args.species_id
    output_dir = os.path.join(args.output_dir, model_name)
    use_foi = True if args.use_foi == 'True' else False
    n_runs = args.n_runs
    pre_process = True if args.pre_process == 'True' else False
    print('Running script with the following args:\n', args)
    print('\n')

else:
    # Define defaults without using Argparse
    model_name = 'nn_model_0'
    use_cogs = False
    weights = 1
    use_noise = True
    neg_ratio = 1
    drop_homology = True
    species_id = '511145'
    output_dir = os.path.join('benchmark/cog_predictions', model_name)
    use_foi = False
    n_runs = 1
    pre_process = False

# Just to hide ugly code
HIDE = True
if HIDE:
    # Check whether the specified path exists or not
    isExist = os.path.exists(os.path.join(output_dir, 'ensemble'))
    if not isExist:
        # Create it
        os.makedirs(os.path.join(output_dir, 'ensemble'))
        print("{} directory created.".format(os.path.join(output_dir, 'ensemble')))

    # Specify link paths
    full_kegg_path = 'data/kegg_benchmarking.CONN_maps_in.v11.tsv'
    full_kegg = pd.read_csv(full_kegg_path, header=None, sep='\t')

    # Map species ID to  name
    species_dict = {'511145': 'ecoli', '9606': 'human', '4932': 'yeast'}
    full_kegg_path = 'data/kegg_benchmarking.CONN_maps_in.v11.tsv'
    full_kegg = pd.read_csv(full_kegg_path, header=None, sep='\t')

    # Define model parameters
    to_shuffle = True
    batch_size = 50
    epochs = 100
    input_size = 12 if drop_homology else 13
    hidden_size = 200
    output_size = 1
    learning_rate = 0.001 # <-- best: 0.001
    

    # Store parameters
    params = {
        'epochs':epochs,
        'input_size':input_size,
        'output_size':output_size,
        'hidden_size':hidden_size,
        'to_shuffle': to_shuffle
        }

    for (species, species_name) in species_dict.items():
        if species in species_id:

            print("Computing for {}".format(species))
            spec_path = 'data/{}.protein.links.full.v11.5.txt'.format(species)
            kegg_data = pd.read_csv(spec_path, header=0, sep=' ', low_memory=False)

            # Load in pre-defined train and validate sets
            train_path = "pre_processed_data/scaled/{}_train.csv".format(species_name)
            valid_path = "pre_processed_data/scaled/{}_valid.csv".format(species_name)
            all_path = 'pre_processed_data/scaled/{}_all.csv'.format(species_name)    

            
            # Load train, test, valid data
            train_data = pd.read_csv(train_path, header=0, low_memory=False, index_col=0)
            valid_data = pd.read_csv(valid_path, header=0, low_memory=False, index_col=0)
            all_data = pd.read_csv(all_path, header=0, low_memory=False, index_col=0)

            # Load in all data even without KEGG memberships
            spec_path = 'data/{}.protein.links.full.v11.5.txt'.format(species)
            x_data = pd.read_csv(spec_path, header=0, sep=' ', low_memory=False)

            # Remove regference to the original data 
            x = copy.deepcopy(train_data)
            a = copy.deepcopy(all_data)
            v = copy.deepcopy(valid_data)

            t1 = time.time()
            output = run_pipeline(x=x,cogs=use_cogs,
                                params=params, weights=weights, noise=use_noise, n_runs=n_runs)
            t2 = time.time()
            print("Finished training in {}".format(t2-t1))


        ###############################################################################################
            # Make predictions
        ###############################################################################################

            print("Making inference")
            # Grab classifier(s)
            classifiers = output['classifier'][-1]

            # Remove COG labels from the data 
            # x.drop(columns=['labels', 'cogs'], inplace=True)
            x = a
            x_labels = x['labels']
            v_labels = v['labels']

            x.drop(columns=['labels'], inplace=True)
            v.drop(columns=['labels', 'cogs'], inplace=True)

            # Get probabilities
            _, xy, xy_hat, x_probas = predict(net=classifiers, x_test=x, y_test=x_labels)
            x_probas = [float(x) for x in x_probas[0]]
            x_probas = [[x,x] for x in x_probas]

            _, vy, vy_hat, v_probas = predict(net=classifiers, x_test=v, y_test=v_labels)
            v_probas = [float(x) for x in v_probas[0]]
            v_probas = [[x,x] for x in v_probas]

            
            # Need to import data/spec_id.combinedv11.5.tsv for filtering on hold-out
            combined_score_file = 'data/{}.combined.v11.5.tsv'.format(species)
            combined_scores = pd.read_csv(combined_score_file, header=None, sep='\t')


            # Save data compatible for Damaians benchmark script (all data)
            x_outs = save_outputs_benchmark(x=x, probas=x_probas,  sid=species,
                                            direc=output_dir, model_name=model_name + '.train_data')
            
            v_outs = save_outputs_benchmark(x=v, probas=v_probas,  sid=species,
                                            direc=output_dir, model_name=model_name + '.hold_out_data')


            # Get the intersection benchmark plot 
            filtered_string_score_x = get_interesction(target=x_outs, reference=combined_scores)
            filtered_string_score_v = get_interesction(target=v_outs, reference=combined_scores)

            data_intersections = {
            'train_data': filtered_string_score_x,
            'hold_out_data': filtered_string_score_v}

            t2 = time.time()
            print("Finished predictions in {}".format(t2-t1))

            print('Saving model(s)')
            for i, model in enumerate(output['classifier']):
                torch.save(model.state_dict(), os.path.join(output_dir, 'ensemble', 'model_{}_{}'.format(i, species)))


            for i, (file_name, filtered_file) in enumerate(data_intersections.items()):
                
                # Save data compatible for Damaians benchmark script (all data)
                save_dir = os.path.join(
                        output_dir, '{}.{}.combined.v11.5.tsv'.format(file_name, species))

                filtered_file.to_csv(
                        save_dir, header=False, index=False, sep='\t')

                                                
                json_report = generate_quality_json(
                        model_name=model_name, direct=output_dir, sid=species, alt=file_name)


                # Call Damians benchmark script on all of train - test - valid
                print("Computing summary statistics for {} data.".format(file_name))
                command = ['perl'] + ['compute_summary_statistics_for_interact_files.pl'] + \
                    ["{}/quality_full_{}.{}.{}.json".format(
                        output_dir, model_name, file_name, species)]
                out = subprocess.run(command)
        

        
                