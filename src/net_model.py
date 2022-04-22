import os
from string_utils import *
import numpy as np
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
import copy
from collections import Counter as C
from itertools import tee


def pairwise(iterable):
    # Chains sequences: 'ABCDEFG' --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return list(zip(a, b))
    
class BinaryClassification(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=[11], output_dim=1):
        """ A Neural Network which can be adaptively parameterised for tabular data predictions.

        :param input_dim: the number of input units (features), defaults to 11
        :type input_dim: int, optional.

        :param hidden_dim: each element value gives the number of neurons per layer,
                           while the len(hidden_sim) signifies the number of layers, defaults to [11]
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
            self.chained_dims = pairwise(
                [self.hidden_dim[0]] + self.hidden_dim)
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

        # Optional dropout usage
        self.dropout = nn.Dropout(p=0.02)

    def forward(self, inputs):
        """Computes the forward pass of the network

        :param inputs: batched data comin from dataloader
        :type inputs: numpy array
        :return: neural network logit output
        :rtype: float tensor
        """

        dropout = False
        x = inputs
        for layer in self.layers:
            x = layer(x)

        # Final outputs
        if dropout:
            x = self.dropout(x)
        x = self.output_layer(x)
        return x

def train_network(params, x_train, y_train):
    """ Train the neural networok

    :param params: Parameters used for the neural network model see definitions below
    :type params: dictionary
    :param x_train: features to train on 
    :type x_train: pandas.core.DataFrame
    :param y_train: grounth truth observations
    :type y_train: 1-dimensional array
    :return: a fully trained neural network 
    :rtype: pytorch neural network model
    """

    # Grab parameters
    epochs = params['epochs']
    criterion = params['criterion']
    net = params['net']
    optimizer = params['optimizer']
    to_shuffle = params['to_shuffle']

    # Balance the dataset
    balance = False
    if balance:
        d_train = dict(C(y_train.values))
        x_train['labels'] = y_train.values
        x_train = x_train.groupby('labels').apply(
            lambda x: x.sample(
                n=d_train[1], replace=False)).reset_index(drop=True)
        y_train = x_train['labels']
        x_train.drop(columns='labels', inplace=True)

    # Generate train tensors
    y_tensor_train = torch.FloatTensor(y_train.values)
    x_tensor_train = torch.FloatTensor(x_train.values)

    # Establish train data-loader
    to_shuffle = True
    train_tensor = data_utils.TensorDataset(x_tensor_train, y_tensor_train)
    train_loader = data_utils.DataLoader(
        train_tensor, batch_size=len(y_train), shuffle=to_shuffle)

    # Train loop
    net.train()
    for e in range(1, epochs + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:

            # Grab data
            X_batch, y_batch = X_batch, y_batch
            optimizer.zero_grad()

            # Predict
            y_pred = net(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = binary_acc(y_pred, y_batch.unsqueeze(1))
            loss.backward()

            # Step
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(
            f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')
    return net

def model_predict(net, x_test, y_test):
    """Computes neural network predictions on test data

    :param net: neural network object
    :type net: model object
    :param x_test: test features
    :type x_test: pandas.core.DataFra
    :param y_test: test labels
    :type y_test: pandas.core.series or np.array
    :return: network, labels, predictions, predicted probabilities
    :rtype: object, list, list, list
    """

    # Establish data-loader
    to_shuffle = True

    # Generate train tensors
    y_tensor_test = torch.FloatTensor(y_test.values)
    x_tensor_test = torch.FloatTensor(x_test.values)

    # Create the test data loader
    to_shuffle = False
    test_tensor = data_utils.TensorDataset(
        x_tensor_test, y_tensor_test)

    test_loader = data_utils.DataLoader(
        test_tensor, batch_size=len(y_test), shuffle=to_shuffle)

    # Make predicitions
    y = []
    y_hat = []
    y_probas = []

    # Don't backpropogate loss during eval
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            # calculate outputs by running images through the network
            logits = net(inputs)
            probas = torch.sigmoid(logits)
            outputs = torch.round(probas)

            # Collect the outputs across all baches
            y_hat.append(outputs.numpy())
            y.append(labels.numpy())
            y_probas.append(probas.numpy())

    # Reformat the batched predictions
    y_hat = [a.squeeze().tolist() for a in y_hat][0]
    y = [a.squeeze().tolist() for a in y][0]
    return net, y, y_hat, y_probas

def run_pipeline(x, params, cogs=True, train_ratio=0.8, noise=False, n_runs=3):
    """Runs the entire modeling process, pre-processing has now been migrated to src/pre_process.py.

    :param data: x-data containing 'labels' and 'cogs' columns
    :type data: pandas DataFrame object

    :param params: model hyper-parameter dictionary
    :type param: dict

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
    test_ratio = 1 - train_ratio

    # Split the data
    train_splits = []
    test_splits = []
    models = []
    predictions = []
    probabilities = []
    accuracies = []

    # Pre-allocate the datasets
    for i in range(1, n_runs + 1):

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
        print("Training on sampling run {}".format(i + 1))
        x_train, y_train = train_splits[i]
        x_test, y_test = test_splits[i]

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

        # Define key model args
        net = BinaryClassification(input_dim=input_size, hidden_dim=
                                   hidden_size, output_dim=output_size)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(net.parameters(), lr=params['learning_rate'], momentum=0.9)
        params['net'] = net
        params['criterion'] = criterion
        params['optimizer'] = optimizer 

        print('Network Architecture: \n', net)
        net = train_network(params=params, x_train=x_train, y_train=y_train)
        print("Predicting on test data")
        net, y, y_hat, y_probas = model_predict(
            net=net, x_test=x_test, y_test=y_test)

       
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

def combine_ensemble_reports(df_list, protein_names):
    """ Combines the each dataset and lists each results under multi-index

    :param df_list: list containing pandas DF
    :type df_list: list

    :param protein_names: index list
    :type protein_names: list
    """

    x = copy.deepcopy(df_list[0])
    pn = [x.split("and") for x in protein_names]
    pn1, pn2 = list(zip(*pn))
    x['run'] = ['run_{}'.format(0)] * np.shape(x)[0]
    x['protein1'] = pn1
    x['protein2'] = pn2
    x.reset_index(inplace=True, drop=True)

    # Extract each subsequent pandas DatFrame and modify it
    for i in range(1, len(df_list)):
        xi = df_list[i]
        xi['protein1'] = pn1
        xi['protein2'] = pn2
        xi['run'] = ['run_{}'.format(i)] * np.shape(xi)[0]
        xi.reset_index(inplace=True, drop=True)

        # Concatenate the 2 pandas DataFrames
        x = pd.concat([x, xi], axis=0)

    # x.drop(columns=['index'], inplace=True)
    x.sort_index()

    # Define multi-indices
    inda = x.index.values
    indb = x['run'].values

    # Set as tuple object & create MultiIndex (MI) obj
    tuples = list(zip(inda, indb))
    index = pd.MultiIndex.from_tuples(tuples, names=["id", "run"])

    # Assign indicies using MI object
    x = pd.DataFrame(x.values, columns=x.columns, index=index)
    x = x.sort_index(level='id')
    x.drop(columns=['run'], inplace=True)
    return x

def mean_probas(x_test, y_test, models, compute_summary=False):

    # Initialise final results (mean_probas)
    n_rows = np.shape(y_test)[0]
    n_cols = len(models)
    mean_probas = np.zeros(shape=(n_rows, n_cols))

    # Loop over all models, make predictions across K model, compute average.
    summaries = []
    for i in range(len(models)):
        _, _, _, probas = model_predict(models[i], x_test, y_test)
        flattend_probas = probas[0].flatten()

        if compute_summary:
            df = pd.DataFrame(flattend_probas, columns=['positive'])
            summaries.append(df)
        mean_probas[:, i] = flattend_probas

    # Returns the mean predictions for all K models
    mean_probas = mean_probas.mean(axis=1)
    mean_probas = [(x, x) for x in mean_probas]
    return mean_probas, summaries


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

###############################################################################################
# START SCRIPT
###############################################################################################


# Extract input variables from Argparse
parser = argparse.ArgumentParser(description='bambi')
parser.add_argument('-n', '--model_name', type=str, metavar='',
                    required=True, default='model_0', help='name of the model')

parser.add_argument('-c', '--cogs', type=str, metavar='',
                    required=True, default=True, help='to split on cogs or not')

parser.add_argument('-un', '--use_noise', type=str, metavar='',
                    required=True, default=False, help='if True, injects noise to X')

parser.add_argument('-dh', '--drop_homology', type=str, metavar='',
                    required=True, default=True, help='if True, drops "homology" feature')

parser.add_argument('-sid', '--species_id', type=str, metavar='',
                    required=True, default='511145 9606 4932', help='ID of species to include sepr=' '')

parser.add_argument('-i', '--input_dir', type=str, metavar='',
                    required=True, default='../pre_processed_data/', help='directory to load data from')

parser.add_argument('-o', '--output_dir', type=str, metavar='',
                    required=True, default='benchmark/cog_predictions', help='directory to save outputs to')

parser.add_argument('-ns', '--n_sampling_runs', type=int, metavar='',
                    required=True, default=3, help='number of randomised samplings on COG splits')

parser.add_argument('-bs', '--batch_size', type=int, metavar='',
                    required=True, default=50, help='number of batches to use for model training')

parser.add_argument('-e', '--epochs', type=int, metavar='',
                    required=True, default=100, help='number of epochs to complete for model training')

parser.add_argument('-hs', '--hidden_size', type=str, metavar='',
                    required=True, default='200', help='amount of neurons per hidden layer e.g. "200 100 10" generates 3 hidden layers with the corresponding elements representing the number of neurons per layer. ')

parser.add_argument('-lr', '--learning_rate', type=float, metavar='',
                    required=True, default=0.001, help='learning rate to apply to gradient update rule in back propogation')

parser.add_argument('-gr', '--generate_report', type=str, metavar='',
                    required=True, default='False', help='generates ensemble report and saves each model in the ensemble (warning - very slow')

# Parse agrs
FORMAT = True
args = parser.parse_args()
model_name = args.model_name
use_cogs = True if args.cogs == 'True' else False
use_noise = True if args.use_noise == 'True' else False
drop_homology = True if args.drop_homology == 'True' else False
generate_report = True if args.generate_report == 'True' else False
species_id = args.species_id
output_dir = os.path.join(args.output_dir, model_name)
input_dir = os.path.join(args.input_dir)
n_runs = args.n_sampling_runs

# ML args
input_size = 12 if drop_homology else 13
output_size = 1
to_shuffle = True
batch_size = args.batch_size
epochs = args.epochs
hidden_size = [int(x) for x in args.hidden_size.split(' ')]
learning_rate = args.learning_rate
print('Running script with the following args:\n', args)
print('\n')


# Check whether the specified path exists or not
isExist = os.path.exists(os.path.join(output_dir, 'ensemble'))
if not isExist:
    # Create it
    os.makedirs(os.path.join(output_dir, 'ensemble'))
    print("{} directory created.".format(
        os.path.join(output_dir, 'ensemble')))

# Specify link paths
full_kegg_path = '../data/kegg_benchmarking.CONN_maps_in.v11.tsv'
full_kegg = pd.read_csv(full_kegg_path, header=None, sep='\t')

# Map species ID to  name
species_dict = {'511145': 'ecoli', '9606': 'human', '4932': 'yeast'}
full_kegg_path = '../data/kegg_benchmarking.CONN_maps_in.v11.tsv'
full_kegg = pd.read_csv(full_kegg_path, header=None, sep='\t')


# Store parameters
params = {
    'epochs': epochs,
    'learning_rate':learning_rate,
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'to_shuffle': to_shuffle
}

for (species, species_name) in species_dict.items():
    if species in species_id:

        print("Computing for {}".format(species))
        spec_path = '../data/{}.protein.links.full.v11.5.txt'.format(species)
        kegg_data = pd.read_csv(
            spec_path, header=0, sep=' ', low_memory=False)

       # Load in pre-defined train and validate sets
        train_path = os.path.join(input_dir, "{}_train.csv".format(
            species_name))
        valid_path = os.path.join(input_dir,"{}_valid.csv".format(
            species_name))
        all_path =  os.path.join(input_dir,"{}_all.csv".format(
            species_name))

        # Load train, test, valid data
        train_data = pd.read_csv(
            train_path, header=0, low_memory=False, index_col=0)
        valid_data = pd.read_csv(
            valid_path, header=0, low_memory=False, index_col=0)
        all_data = pd.read_csv(
            all_path, header=0, low_memory=False, index_col=0)

        # Load in all data even without KEGG memberships
        spec_path = '../data/{}.protein.links.full.v11.5.txt'.format(species)
        x_data = pd.read_csv(spec_path, header=0,
                             sep=' ', low_memory=False)

        # Remove regference to the original data
        x = copy.deepcopy(train_data)
        a = copy.deepcopy(all_data)
        v = copy.deepcopy(valid_data)

        # Run and time the model
        t1 = time.time()
        output = run_pipeline(x=x, cogs=use_cogs, params=params,noise=use_noise, n_runs=n_runs)
        t2 = time.time()
        print("Finished training in {}".format(t2 - t1))

    ###############################################################################################
        # Make predictions
    ###############################################################################################

        # Grab classifier(s)
        print("Making inference")
        classifiers = output['classifier']
        # Remove COG labels from the data
        # x.drop(columns=['labels', 'cogs'], inplace=True)              # <-- uncomment to run ony on train data, else runs on all data
        x = a                                                           # <-- comment to run ony on train data, else runs on all data

        x_labels = x['labels']
        v_labels = v['labels']

        x.drop(columns=['labels'], inplace=True)
        v.drop(columns=['labels', 'cogs'], inplace=True)

        # Get ensemble probabilities
        ensemble_probas_x, summaries_x = mean_probas(
            x_test=x, y_test=x_labels, models=classifiers, compute_summary=generate_report)
        
        ensemble_probas_v, summaries_v = mean_probas(
            x_test=v, y_test=v_labels, models=classifiers, compute_summary=generate_report)

        if generate_report:
            # Get ensemble reports
            c_x = combine_ensemble_reports(
                summaries_x, protein_names=x.index.values)
            c_v = combine_ensemble_reports(
                summaries_v, protein_names=v.index.values)

            # Save ensemble reports
            c_x.to_csv(os.path.join(output_dir, 'ensemble',
                    'ensemble_report_x_{}.csv'.format(species)))
            c_v.to_csv(os.path.join(output_dir, 'ensemble',
                    'ensemble_report_v_{}.csv'.format(species)))


        # Need to import data/spec_id.combinedv11.5.tsv for filtering on hold-out
        combined_score_file = '../data/{}.combined.v11.5.tsv'.format(species)
        combined_scores = pd.read_csv(
            combined_score_file, header=None, sep='\t')

        # Save data compatible for Damaian's benchmark script (all data)
        x_outs = save_outputs_benchmark(x=x, probas=ensemble_probas_x, sid=species,
                                        direc=output_dir, model_name=model_name + '.train_data')

        v_outs = save_outputs_benchmark(x=v, probas=ensemble_probas_v, sid=species,
                                        direc=output_dir, model_name=model_name + '.hold_out_data')

        # Get the test intersection with benchmark plot - to make sure all ROC curves have equal points
        filtered_string_score_x = get_interesction(
            target=x_outs, reference=combined_scores)

        # Get the validation intersection with benchmark plot - to make sure all ROC curves have equal points
        filtered_string_score_v = get_interesction(
            target=v_outs, reference=combined_scores)

        # Collect all subset data
        data_intersections = {
            'train_data': filtered_string_score_x,
            'hold_out_data': filtered_string_score_v}

        # Save all the models
        t2 = time.time()
        print("Finished predictions in {}".format(t2 - t1))
        print('Saving model(s)')
        for i, model in enumerate(output['classifier']):
            torch.save(model.state_dict(), os.path.join(
                output_dir, 'ensemble', 'model_{}_{}'.format(i, species)))

        for i, (file_name, filtered_file) in enumerate(data_intersections.items()):

            # Save above and generate JSON files (all data)
            save_dir = os.path.join(
                output_dir, '{}.{}.combined.v11.5.tsv'.format(file_name, species))

            filtered_file.to_csv(
                save_dir, header=False, index=False, sep='\t')

            json_report = generate_quality_json(
                model_name=model_name, direct=output_dir, sid=species, alt=file_name)

            # Call Damians benchmark script on data splits
            print("Computing summary statistics for {} data.".format(file_name))
            command = ['perl'] + ['../compute_summary_statistics_for_interact_files.pl'] + \
                ["{}/quality_full_{}.{}.{}.json".format(
                    output_dir, model_name, file_name, species)]
            out = subprocess.run(command)
