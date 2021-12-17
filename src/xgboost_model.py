import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from collections import Counter as C
import seaborn as sns
import time
import argparse
import subprocess
from string_utils import *


def run_pipeline(data, labels, spec_kegg, params, scale=False, weights=None,
                 cogs=True, species_id=None, train_ratio=0.8, noise=False, neg_ratio=1, plot=False):
    """Runs the entire piplie: COG splits --> data preprocessing --> model outputs.

    :param data: x-data
    :type data: pandas DataFrame object
    :param labels: y-labels
    :type labels: iterable (e.g. list)
    :param spec_kegg: species KEGG paths
    :type spec_kegg: pandas DataFrame object
    :param params: model hyper-parameter dictionary
    :type param: dict
    :param scale: if True,  scales inputs in range [0,1], defaults to False
    :type scale: bool, optional
    :param weights: if provided, upscales the positive class importance during training, defaults to None
    :type weights: float, optional
    :param cogs: if True, train and test are split on COG observations, defaults to True
    :type cogs: bool, optional
    :param species_id: species identifier, defaults to None
    :type species_id: string, optional
    :param train_ratio: the proportion of data used for training, defaults to 0.8
    :type train_ratio: float, optional
    :param noise: if True, injects noise term to specified features, defaults to False
    :type noise: bool, optional
    :param neg_ratio: the proportion of negative-positive samples, defaults to 1
    :type neg_ratio: int, optional
    param plot: it True, plots outputs
    :type plot: bool, defauts to False
    :return: Returns an output dict containing key information.
    :rtype: dict
    """

    print(" Beginning pipeline...")
    # Format the data
    x, y = pre_process_data(data, labels)
    test_ratio = 1-train_ratio

    if cogs:
        print('Generating COG splits')
        # split data on the cogs
        cog_map, _ = create_cog_map(spec_kegg=spec_kegg, species_id=species_id)
        pos_tr, pos_te, neg_tr, neg_te = split_on_cogs(x=data, y=labels, cog_map=cog_map,
                                                       neg_ratio=neg_ratio, train_ratio=train_ratio)

        # Generate hold-out set
        ho_pos = pos_te.sample(frac=test_ratio)
        ho_neg = neg_te.sample(frac=test_ratio)
        x_ho = pd.concat([ho_pos, ho_neg])
        y_ho = x_ho.labels
        x_ho.drop(columns=['labels', 'cogs'], inplace=True)

        # Drop hold-out samples from the test set
        pos_te = pos_te.drop(ho_pos.index)
        neg_te = neg_te.drop(ho_neg.index)

        # Define the splits
        x_train = pd.concat([pos_tr, neg_tr])
        x_test = pd.concat([pos_te, neg_te])

        # Shuffle the data
        x_train = x_train.sample(frac=1)
        x_test = x_test.sample(frac=1)
        y_train = x_train.labels
        y_test = x_test.labels

        # Drop the labels from x-train and x-test
        x_train.drop(columns=['labels', 'cogs'], inplace=True)
        x_test.drop(columns=['labels', 'cogs'], inplace=True)
        print('Done')

    else:
        # Train-test splits
        x_train, x_test, y_train, y_test = model_splits(
            x, y, test_ratio=test_ratio)
        x_test, x_ho, y_test, y_ho = model_splits(
            x_test, y_test, test_ratio=test_ratio)

    # Scale the data if necessary
    if scale:
        x_train, x_test, mms = scale_features(x_train, x_test)

    if noise:

        # Add normally distributed noise to following features
        perturb = [
            'neighborhood_transferred',
            'experiments_transferred',
            'textmining',
            'textmining_transferred',
            'experiments',
            'experiments_transferred',
            'coexpression_transferred'
        ]

        # Define guassian noise parameters
        mu = 0
        sigma = 0.005

        x_train = x_train.apply(lambda x: inject_noise(
            x, mu=mu, sigma=sigma) if x.name in perturb else x)
        x_test = x_test.apply(lambda x: inject_noise(
            x, mu=mu, sigma=sigma) if x.name in perturb else x)
        x_ho = x_ho.apply(lambda x: inject_noise(
            x, mu=mu, sigma=sigma) if x.name in perturb else x)

    # Model init
    # increased weight applied to the positive class
    print('Building and Fitting model')
    clf = build_model(params, class_ratio=weights)

    # Predict
    clf = fit(clf, x_train, y_train, x_test, y_test)
    clf, preds, probas, acc, _ = predict(clf, x_test, y_test)
    print('Done')

    # Perform cross validation scoring
    print('Generating CV results')
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=5,
                        metrics="auc", as_pandas=True, seed=123)
    print(cv_results)
    print('Done')

    # plot ROC curve
    score = plot_roc(y_test=y_test, probas=probas, plot=False)
    acc = accuracy_score(y_test, preds)

    output_dict = {
        'preds': preds,
        'probas': probas,
        'accuracy': acc,
        'score': score,
        'classifier': clf,
        'y_train': y_train,
        'y_test': y_test,
        'X_train': x_train,
        'X_test': x_test,
        'X_HO': x_ho,
        'y_HO': y_ho
    }
    return output_dict


###############################################################################################
# START SCRIPT
###############################################################################################


# Extract input variables from Argparse
USE_ARGPASE = True

if USE_ARGPASE:
    parser = argparse.ArgumentParser(description='XGBoost')
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
    use_foi = True if args.use_foi == 'True' else False
    print('Running script with the following args:', args)

else:
    # Define defaults without using Argparse
    model_name = 'model_0'
    use_cogs = True
    weights = 4
    use_noise = True
    neg_ratio = 4
    drop_homology = True
    species_id = '511145'
    output_dir = os.path.join('benchmark/cog_predictions', model_name)
    use_foi = False

# Check whether the specified path exists or not
isExist = os.path.exists(output_dir)
if not isExist:
    # Create it
    os.makedirs(output_dir)
    print("{} directory created.".format(output_dir))

# Specify link paths
full_kegg_path = 'data/kegg_benchmarking.CONN_maps_in.v11.tsv'
full_kegg = pd.read_csv(full_kegg_path, header=None, sep='\t')

# Run the full pipeline (these values have been optimised, don't change!)
params = {'max_depth': 15,
          'eta': 0.1,
          'objective': 'binary:logistic',
          'alpha': 0.1,
          'lambda': 0.01}

# Map species ID to  name
species_dict = {'511145': 'ecoli', '9606': 'human', '4932': 'yeast'}

# Run code for each species given in bash file
for (species, species_name) in species_dict.items():
    if species in species_id:
        print("Computing for {}".format(species))
        spec_path = 'data/{}.protein.links.full.v11.5.txt'.format(species)
        label_path = 'data/{}_labels.csv'.format(species_name)
        data = pd.read_csv(spec_path, header=0, sep=' ', low_memory=False)
        labels = pd.read_csv(label_path, index_col=False, header=None)

        # Format the data
        if FORMAT:
            x, y, idx = format_data(
                data, labels, drop_homology=drop_homology)

        t1 = time.time()
        output = run_pipeline(data=x, labels=y, spec_kegg=data, cogs=use_cogs,
                              params=params, weights=weights, species_id='{}.'.format(
                                  species),
                              noise=use_noise, neg_ratio=neg_ratio)
        t2 = time.time()
        print("Finished training in {}".format(t2-t1))

        # Make classifications
        clf = output['classifier']
        probas = clf.predict_proba(x)
        preds = clf.predict(x)
        hold_out_probas = clf.predict_proba(output['X_HO'])
        hold_out_preds = clf.predict(output['X_HO'])

        # Save data compatible for Damaians benchmark script
        x_outs = save_outputs_benchmark(x=x, probas=probas,  sid=species,
                                        direc=output_dir, model_name=model_name)

        # Need to import data/spec_id.combinedv11.5.tsv for filtering on hold-out
        combined_score_file = 'data/{}.combined.v11.5.tsv'.format(species)
        combined_scores = pd.read_csv(
            combined_score_file, header=None, sep='\t')

        hol_out_preds = save_outputs_benchmark(x=output['X_HO'], probas=hold_out_probas,
                                               sid=species, direc=output_dir,
                                               model_name=model_name + '.hold_out')

        # Filter STRING score set to contain only observations in hold-out
        filtered_string_score = get_interesction(
            target=hol_out_preds, reference=combined_scores)

        # Resave the intersect file to the model directory
        save_dir = os.path.join(
            output_dir, 'hold_out.{}.combined.v11.5.tsv'.format(species))
        filtered_string_score.to_csv(
            save_dir, header=False, index=False, sep='\t')
        # This requires to change the quality json function if dataset is a hold_out set

        # Generate quality reports
        json_report = generate_quality_json(
            model_name=model_name, direct=output_dir, sid=species)

        hold_out_json_report = generate_quality_json(
            model_name=model_name, direct=output_dir, sid=species, hold_out=True)

        # Call Damian benchark script here
        print('Running benchmark')
        command = ['perl'] + ['compute_summary_statistics_for_interact_files.pl'] + \
            ["{}/quality_full_{}.{}.json".format(
                output_dir, model_name, species)]
        out = subprocess.run(command)

        command = ['perl'] + ['compute_summary_statistics_for_interact_files.pl'] + \
            ["{}/quality_full_{}.hold_out.{}.json".format(
                output_dir, model_name, species)]
        out = subprocess.run(command)
