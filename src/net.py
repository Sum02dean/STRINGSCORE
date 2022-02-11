import sys
import os
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


class NN(nn.Module):	
    def __init__(self, input_size, hidden_size, output_size):
        """Barebones fully connected neural network with a single hidden layer.

        :param input_size: Number of features
        :type input_size: int
        :param hidden_size: number of neurons in the hidden layers
        :type hidden_size: int
        :param output_size: Number of classes (1 for binary)
        :type output_size: int
        """
        super(NN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Input layer
        self.layer_1 = nn.Linear(input_size, self.hidden_size)
        # Hidden layer
        self.layer_2 = nn.Linear(self.hidden_size, self.hidden_size)
        # Output layer
        self.out = nn.Linear(self.hidden_size, self.output_size)
        # Sigmoid to squish data to probability distribution
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        """Forward method used for forward pass during training.

        :param x: data (features without outcome variabel)
        :type x: nd-array or pandas.core.DataFrame
        :return: probability of class 1 scaled between 0-1
        :rtype: tensor.float32
        """
        grad_val = 0.01
        x = nn.LeakyReLU(grad_val)(self.layer_1(x))
        x = nn.LeakyReLU(grad_val)(self.layer_2(x))
        output = self.out(x)
        return self.sig(output)


def train_network(params, x_train, y_train):
    
    # Grab parameters
    epochs = params['epochs']
    criterion = params['criterion']
    net = params['net']
    optimizer = params['optimizer']
    batch_size = params['batch_size']
    to_shuffle = params['to_shuffle']
    
    # Generate data tensors
    features_tensor = torch.tensor(x_train, dtype=torch.float32)
    labels_tensor = torch.tensor(y_train, dtype=torch.float32)
    labels_tensor = torch.unsqueeze(labels_tensor, 1)

    # Build data-loader
    train_tensor = data_utils.TensorDataset(features_tensor, labels_tensor)
    train_loader = data_utils.DataLoader(train_tensor, batch_size=batch_size, shuffle=to_shuffle)
    

    # Train loop
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for i, data, in enumerate(train_loader, 0):
            # Zero the gradients
            optimizer.zero_grad()

            # Extract the inputs
            inputs, labels = data

            # Compute the neural network outputs
            outputs = net(inputs)

            # Compute the loss between inputs and outputs ans tep the optimizer
            loss = criterion(outputs, labels)
            
            # print(outputs[0], labels[0])
            loss.backward()

            # Step optimizer
            optimizer.step()

            # Print the statisticcs
            running_loss += loss.item()
            mini_epochs = 1000
    
    if i % mini_epochs == mini_epochs-1:    # print every n mini-batches
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / mini_epochs:.3f}')
        running_loss = 0.0
print("Finished training")




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

    :param neg_ratio: the proportion of negative-positive samples, defaults to 1
    :type neg_ratio: int, optional

    
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
                x, y, test_ratio=test_ratio)   

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
        print("Computing predictions for sampling run {}".format(i+1))
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
        clf = build_model(params, class_ratio=weights)
        clf = fit(clf, x_train, y_train, x_test, y_test)
        clf, preds, probas, acc, _ = predict(clf, x_test, y_test)

        # Collect the model specific data
        models.append(clf)
        predictions.append(preds)
        probabilities.append(probas)
        accuracies.append(acc)

    output_dict = {
            'predictions': predictions,
            'probabilities': probabilities,
            'classifier': models,
            'train_splits': train_splits,
            'test_splits': test_splits
            }

    return output_dict

def mean_probas(x, clfs):
    probabilities = 0
    for clf in clfs:
        probas = clf.predict_proba(x)
        probabilities += probas
    probas = probabilities/len(clfs)
    return probas

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
    
    parser.add_argument('-nc', '--n_chains', type=int, metavar='',
                        required=True, default=1000, help='number of chains')

    parser.add_argument('-nd', '--n_draws', type=int, metavar='',
                        required=True, default=100, help='number of draws per chain')
    
    parser.add_argument('-nt', '--n_tune', type=int, metavar='',
                        required=True, default=100, help='number of iterations to tune in NUTS')
    
    parser.add_argument('-fam', '--family', type=str, metavar='',
                    required=True, default='bernoulli', help='prior family to use')
    
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
    n_chains= args.n_chains
    n_draws= args.n_draws
    n_tune= args.n_tune
    family = args.family
    pre_process = True if args.pre_process == 'True' else False
    print('Running script with the following args:\n', args)
    print('\n')

else:
    # Define defaults without using Argparse
    model_name = 'nn_model_0'
    use_cogs = False
    weights = 4
    use_noise = True
    neg_ratio = 4
    drop_homology = True
    species_id = '511145'
    output_dir = os.path.join('benchmark/cog_predictions', model_name)
    use_foi = False
    n_runs = 1
    n_chains= 4
    n_draws= 100
    n_tune= 300
    family = 'bernoulli'
    pre_process = False


# Check whether the specified path exists or not
isExist = os.path.exists(output_dir)
if not isExist:
    # Create it
    os.makedirs(output_dir)
    print("{} directory created.".format(output_dir))

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
input_size = features.shape[1]
hidden_size = 200
output_size = 1
learning_rate = 0.0002 # <-- best: 0.001
criterion = nn.BCELoss(reduction='mean')
net = NN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=learning_rate)
print('Network Architecture: \n',net)

params = {
    'net': net,
    'criterion': criterion,
    'optimizer': optimizer, 
    'batch_size': batch_size, 
    'to_shuffle': to_shuffle
    }

for (species, species_name) in species_dict.items():
    if species in species_id:

        print("Computing for {}".format(species))
        spec_path = 'data/{}.protein.links.full.v11.5.txt'.format(species)
        kegg_data = pd.read_csv(spec_path, header=0, sep=' ', low_memory=False)

        # Load in pre-defined train and validate sets
        train_path = "pre_processed_data/script_test/{}_train.csv".format(species_name)
        valid_path = "pre_processed_data/script_test/{}_valid.csv".format(species_name)
        all_path = 'pre_processed_data/script_test/{}_all.csv'.format(species_name)    

        
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
                            params=params, weights=weights, noise=use_noise, run_cv=False, n_runs=n_runs)
        t2 = time.time()
        print("Finished training in {}".format(t2-t1))


       ###############################################################################################
        # Make predictions
        ###############################################################################################
        