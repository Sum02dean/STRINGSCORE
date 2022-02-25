import sys
import os
from matplotlib import transforms
import pandas as pd
src_path = os.getcwd().replace('/ideas', '')
sys.path.append(src_path)
from string_utils import *
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm
import theano as thno
import theano.tensor as T
from scipy import integrate
from scipy.optimize import fmin_powell
import seaborn as sns
import bambi as bmb
import argparse
import subprocess
import time
from sklearn.decomposition import TruncatedSVD

# Useful functions.
def get_formula(feature_names):
    template = ['{}'] * (len(feature_names))
    template = " + ".join(template)
    template = template.format(*list(feature_names))
    f = 'y ~ ' +  template 
    return f

def mean_probas(x, models, classifiers):
    mp = np.zeros(np.shape(x)[0])

    for i in range(len(models)):
        idata = models[i].predict(classifiers[i], data=x, inplace=False)
        mp += np.mean(idata.posterior['y_mean'].values, axis=(0, 1))
        
    
    # Get the man proabilities
    ensemble_probas = mp / len(models)
    # This is a hack to guarentee that the probas work with the generate_output() script
    ensemble_probas = [(x,x) for x in ensemble_probas]
    return ensemble_probas

def select_pcs(x, n_pcs=None, use_cogs=True, svd=None):
    """Selects the appropriate amount of principle components based on variance explained       

    :param x: Design matrix including labels and COGs
    :type x: pandas dataframe
    """
    print('Transformatin data with SVD')
    # Don't take eigen decompositon for labels or COGs
    y = x['labels'].values
    
    if use_cogs:
        cogs = x['cogs'].values
        x = x.drop(columns=['cogs'], inplace=False)
    x = x.drop(columns=['labels'], inplace=False)

    # Intantiate svd and compute PCs of x
    _, n_cols = np.shape(x)
    n_comp = n_cols-1

    if svd == None:
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        x_pc = pd.DataFrame(svd.fit_transform(x))
    else:
        x_pc = pd.DataFrame(svd.transform(x))

    # Set the index
    x_pc.index = x.index

    if n_pcs == None:
        # Get vairance explained
        ev = np.array(svd.explained_variance_ratio_)
        ev_cumsum = np.cumsum(ev)
        pcs = np.array([x for x in range(1, n_comp + 1)])
        best_pcs = pcs[ev_cumsum <= 0.95]
        n_comp = best_pcs[-1]
    else:
        n_comp = n_pcs
    # Subset the dataframe
    x_pc = x_pc.iloc[:, :n_comp]
    x_pc['labels'] = y

    if use_cogs:
        x_pc['cogs'] = cogs

    return x_pc, n_pcs, svd

def run_pipeline(x, params, scale=False, weights=None, cogs=True, 
                train_ratio=0.8, noise=False, n_runs=3, run_cv=False, verbose_eval=True):

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
    train_splits  = []
    test_splits = []
    models = []
    classifiers = []
    predictions = []
    

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
        x_train.drop(columns=['labels', 'cogs', 'neighborhood'], inplace=True)
        x_test.drop(columns=['labels', 'cogs', 'neighborhood'], inplace=True)

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
        
        # Add normally distributed noise to following features
        if noise:
            dont_perturb = ['labels', 'cogs']
            # Define guassian noise argumnets
            mu = 0
            sigma = 0.05

            x_train = x_train.apply(lambda x: inject_noise(
                x, mu=mu, sigma=sigma) if x.name not in dont_perturb else x)

            x_test = x_test.apply(lambda x: inject_noise(
                x, mu=mu, sigma=sigma) if x.name not in dont_perturb else x)
        

        # Run probabilistic LR bambi model
        x_train['y'] = y_train
        
        # Get the function formula
        f = get_formula(x_train.columns[:-1])
        model = bmb.Model(f, x_train, family=params['family'])
        clf = model.fit(draws=params['draws'], tune=params['tune'], chains=params['chains'], init='adapt_diag')
        models.append(model)
        classifiers.append(clf)
        
        # Run predictions
        idata = model.predict(clf, data=x_test, inplace=False)
        mean_preds = idata.posterior["y_mean"].values
        predictions.append(mean_preds)
        
        # Collect outputs
        output_dict = {
        'predictions': predictions,
        'models': models,
        'classifier': classifiers,
        'train_splits': train_splits,
        'test_splits': test_splits
        }
    return output_dict

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
    model_name = 'bambi_model_0'
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
params = {
    'family': family,
    'chains': n_chains,
    'draws': n_draws, 
    'tune': n_tune}

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
        
        t1 = time.time()
        print("Making inference")

        # Grab classifier(s)
        classifiers = output['classifier']
        models = output['models']

        # Remove COG labels from the data 
        # x.drop(columns=['labels', 'cogs'], inplace=True)
        x = a
        x.drop(columns=['labels', 'neighborhood'], inplace=True)
        v.drop(columns=['labels', 'cogs', 'neighborhood'], inplace=True)

        # Get ensemble probabilities
        ensemble_probas_x = mean_probas(x, models=models, classifiers=classifiers)
        ensemble_probas_v = mean_probas(v, models=models, classifiers=classifiers)
        
        # Need to import data/spec_id.combinedv11.5.tsv for filtering on hold-out
        combined_score_file = 'data/{}.combined.v11.5.tsv'.format(species)
        combined_scores = pd.read_csv(combined_score_file, header=None, sep='\t')


        # Save data compatible for Damaians benchmark script (all data)
        x_outs = save_outputs_benchmark(x=x, probas=ensemble_probas_x,  sid=species,
                                        direc=output_dir, model_name=model_name + '.train_data')
        
        v_outs = save_outputs_benchmark(x=v, probas=ensemble_probas_v,  sid=species,
                                        direc=output_dir, model_name=model_name + '.hold_out_data')


        # Get the intersection benchmark plot 
        filtered_string_score_x = get_interesction(target=x_outs, reference=combined_scores)
        filtered_string_score_v = get_interesction(target=v_outs, reference=combined_scores)

        data_intersections = {
        'train_data': filtered_string_score_x,
        'hold_out_data': filtered_string_score_v}

        t2 = time.time()
        print("Finished predictions in {}\n\n".format(t2-t1))

        print('Saving model(s)')
        for i, model in enumerate(output['classifier']):
            az.to_netcdf(model, filename=os.path.join(output_dir, 'ensemble', 'model_{}_{}'.format(i, species)))

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