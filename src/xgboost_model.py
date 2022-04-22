import os
import pandas as pd
import xgboost as xgb
import time
import argparse
import subprocess
import copy
import warnings
from string_utils import *
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_pipeline(x, params, scale=False, weights=None,
                 cogs=True, train_ratio=0.8, noise=False,
                 n_runs=3, run_cv=False, verbose_eval=True):

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
        print("Computing predictions for sampling run {}".format(i + 1))
        x_train, y_train = train_splits[i]
        x_test, y_test = test_splits[i]

        # Scale the data if necessary
        if scale:
            col_names = x_train.columns
            x_train, x_test, mms = scale_features(
                x_train, x_test, method='standard')
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

        if run_cv:
            # Perform cross-validation on each of the differential splits
            dtrain = xgb.DMatrix(x_train, label=y_train)
            _ = xgb.cv(dtrain=dtrain, params=params, nfold=5,
                                metrics="auc", as_pandas=True, stratified=True,
                                verbose_eval=verbose_eval)

        # Make a one time prediction for each of the splits
        clf = build_model(params, class_ratio=weights)
        clf = fit(clf, x_train, y_train, x_test, y_test)
        clf, preds, probas, acc, _ = model_predict(clf, x_test, y_test)

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


def mean_probas(x, clfs, compute_summary=False):
    """Compute the mean ensemble predictions

    :param x: feates
    :type x: pandas.core.DataFrame

    :param clfs: fitten models
    :type clfs: xgboost XGBClassifier object

    :param compute_summary: computes full ensemble prediction reports, defaults to False
    :type compute_summary: bool, optional

    :return: mean ensemble probabilities, list of pandas.core.DaraFrames
    :rtype: _type_
    """
    probabilities = 0
    summaries = []
    for clf in clfs:
        probas = clf.predict_proba(x)
        probabilities += probas
        if compute_summary:
            df = pd.DataFrame(probas, columns=['negative', 'positive'])
            summaries.append(df)
    probas = probabilities / len(clfs)
    return probas, summaries


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


###############################################################################################
# START SCRIPT
###############################################################################################


# Extract input variables from Argparse
parser = argparse.ArgumentParser(description='XGBoost')
parser.add_argument('-n', '--model_name', type=str, metavar='',
                    required=True, default='model_0', help='name of the model')

parser.add_argument('-c', '--cogs', type=str, metavar='',
                    required=True, default=True, help='to split on cogs or not')

parser.add_argument('-un', '--use_noise', type=str, metavar='',
                    required=True, default=False, help='if True, injects noise to X')

parser.add_argument('-cw', '--class_weight', type=float, metavar='',
                    required=True, default=4, help='factor applied to positive predictions')

parser.add_argument('-nr', '--neg_ratio', type=int, metavar='',
                    required=True, default=4, help='factor increase in number of negative obs compared to positive observations')

parser.add_argument('-dh', '--drop_homology', type=str, metavar='',
                    required=True, default=True, help='if True, drops "homology" feature')

parser.add_argument('-sid', '--species_id', type=str, metavar='',
                    required=True, default='511145 9606 4932', help='ID of species to include sepr=' '')

parser.add_argument('-i', '--input_dir', type=str, metavar='',
                    required=True, default='pre_processed_data/scaled/', help='directory for pre-processed data')

parser.add_argument('-o', '--output_dir', type=str, metavar='',
                    required=True, default='benchmark/cog_predictions', help='directory to save outputs to')

parser.add_argument('-ns', '--n_sampling_runs', type=int, metavar='',
                    required=True, default=3, help='number of randomised samplings for COG splits')

parser.add_argument('-gr', '--generate_report', type=str, metavar='',
                    required=True, default='False', help='generates ensemble report and saves each model in the ensemble (warning - very slow')


# To format data
FORMAT = True

# Parse args
args = parser.parse_args()
model_name = args.model_name
use_cogs = True if args.cogs == 'True' else False
weights = args.class_weight
use_noise = True if args.use_noise == 'True' else False
neg_ratio = args.neg_ratio
drop_homology = True if args.drop_homology == 'True' else False
species_id = args.species_id
output_dir = os.path.join(args.output_dir, model_name)
input_dir = os.path.join(args.input_dir)
n_samples = args.n_sampling_runs
generate_report = True if args.generate_report == 'True' else False

print('Running script with the following args:\n', args)
print('\n')


# Check whether the specified path exists or not
isExist = os.path.exists(os.path.join(output_dir, 'ensemble'))
if not isExist:
    # Create it
    os.makedirs(os.path.join(output_dir, 'ensemble'))
    print("{} directory created.".format(os.path.join(output_dir, 'ensemble')))

# Specify link paths
full_kegg_path = '../data/kegg_benchmarking.CONN_maps_in.v11.tsv'
full_kegg = pd.read_csv(full_kegg_path, header=None, sep='\t')

# Run the full pipeline (these values have been optimised, don't change!)
params = {'max_depth': 15,
          'eta': 0.1,
          'objective': 'binary:logistic',
          'alpha': 0.1,
          'lambda': 0.01,
          'subsample': 0.9,
          'colsample_bynode': 0.2}

# Map species ID to  name
species_dict = {'511145': 'ecoli', '9606': 'human', '4932': 'yeast'}

# Run code for each species given in bash file: do this 'n_runs' times to reduce stochasticity
predictions = []
probabilities = []

for (species, species_name) in species_dict.items():
    if species in species_id:
        print("Computing for {}".format(species))
        spec_path = '../data/{}.protein.links.full.v11.5.txt'.format(species)
        kegg_data = pd.read_csv(spec_path, header=0, sep=' ', low_memory=False)

        # Load in pre-defined train and validate sets
        train_path = os.path.join(input_dir, "{}_train.csv".format(
            species_name))
        valid_path = os.path.join(input_dir,"{}_valid.csv".format(
            species_name))
        all_path =  os.path.join(input_dir,"{}_all.csv".format(
            species_name))

        # Load train, test, valid data
        train_data = pd.read_csv(train_path, header=0,
                                 low_memory=False, index_col=0)
        valid_data = pd.read_csv(valid_path, header=0,
                                 low_memory=False, index_col=0)
        all_data = pd.read_csv(
            all_path, header=0, low_memory=False, index_col=0)

        # Load in all data even without KEGG memberships
        spec_path = '../data/{}.protein.links.full.v11.5.txt'.format(species)
        x_data = pd.read_csv(spec_path, header=0, sep=' ', low_memory=False)

        # Remove reference to the original data  (uncomment as necessary)
        x = copy.deepcopy(train_data)
        a = copy.deepcopy(all_data)
        v = copy.deepcopy(valid_data)

        t1 = time.time()
        output = run_pipeline(x=x, cogs=use_cogs,
                              params=params, weights=weights, noise=use_noise, run_cv=False, n_runs=n_samples, scale=False)
        t2 = time.time()
        print("Finished training in {}".format(t2 - t1))

        ###############################################################################################
        # Make predictions
        ###############################################################################################

        # Grab classifier(s)
        classifiers = output['classifier']

        # Remove COG labels from the data
        v.drop(columns=['labels', 'cogs'], inplace=True)
        # x.drop(columns=['labels', 'cogs'], inplace=True)              # <-- uncomment to run ony on train data, else runs on all data
        x = a                                                           # <-- comment to run ony on train data, else runs on all data
        x.drop(columns=['labels'], inplace=True)

        # Get ensemble probabilities
        ensemble_probas_x, summaries_x = mean_probas(
            x, classifiers, compute_summary=generate_report)
        ensemble_probas_v, summaries_v = mean_probas(
            v, classifiers, compute_summary=generate_report)

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

        # Save data compatible for Damaians benchmark script (all data)
        x_outs = save_outputs_benchmark(x=x, probas=ensemble_probas_x, sid=species,
                                        direc=output_dir, model_name=model_name + '.train_data')

        v_outs = save_outputs_benchmark(x=v, probas=ensemble_probas_v, sid=species,
                                        direc=output_dir, model_name=model_name + '.hold_out_data')

        # Get the intersection benchmark plot
        filtered_string_score_x = get_interesction(
            target=x_outs, reference=combined_scores)
        filtered_string_score_v = get_interesction(
            target=v_outs, reference=combined_scores)

        # Store all splits
        data_intersections = {
            'train_data': filtered_string_score_x,
            'hold_out_data': filtered_string_score_v}

        print('Saving model(s)')
        for i, model in enumerate(output['classifier']):
            model.save_model(os.path.join(output_dir, 'ensemble',
                             'model_{}_{}.json'.format(i, species)))

        # Save data compatible for Damaians benchmark script (all data)
        for i, (file_name, filtered_file) in enumerate(data_intersections.items()):

            save_dir = os.path.join(
                output_dir,
                '{}.{}.combined.v11.5.tsv'.format(file_name, species))

            filtered_file.to_csv(
                save_dir, header=False, index=False, sep='\t')

            json_report = generate_quality_json(
                model_name=model_name, direct=output_dir, sid=species, alt=file_name)

            # Call Damians benchmark script on all data splits
            print("Computing summary statistics for {} data.".format(file_name))
            command = ['perl'] + ['../compute_summary_statistics_for_interact_files.pl'] + \
                ["{}/quality_full_{}.{}.{}.json".format(
                    output_dir,
                    model_name,
                    file_name,
                    species)]
            # Run Perl summary statistics script from witin python
            out = subprocess.run(command)
