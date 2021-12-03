import sys
import seaborn as sns
import os
import pandas as pd
from sklearn.preprocessing import Normalizer
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, average_precision_score
from sklearn.metrics import accuracy_score
from collections import Counter as C
import seaborn as sns
from tqdm import tqdm
import time
import copy
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
import argparse
import subprocess
import json


def get_mask(x, substring='9606.'):
    """Simple boolean search, return True if substring matched, else False.

    :param x: annotation data
    :type x: pandas Series
    :param substring: organism to query, defaults to '9606.'
    :type substring: str, optional
    :return: True if match found, else False
    :rtype: Bool
    """
    spec = x['proteins']
    if substring in spec:
        return True
    else:
        return False


def create_cog_map(spec_kegg, species_id='9606.'):
    """Creates a mapping between all the protein IDs in a dataset
    and their respective COG IDs.

    :param spec_kegg: species KEGG data
    :type spec_kegg: pandas dataframe
    :param species_id: organism of interest, defaults to '9606.'
    :type species_id: str, optional
    :return: coder, decoder
    :rtype: dict, dict
    """

    # Filter the COGs for a specific organism
    cog_path = '/mnt/mnemo2/damian/dean/ogs.luca.core.tsv'
    cog_df = pd.read_csv(cog_path, sep='\t', header=0)
    cog_df.columns = ['idk', 'cog_group', 'proteins']
    mask = cog_df.apply(lambda x: get_mask(x, substring=species_id), axis=1)
    cogs = cog_df[mask]

    # For all data in species subset create a mapping
    spec_proteins = list(spec_kegg.protein1.values) + \
        list(spec_kegg.protein2.values)
    spec_proteins = list(set(spec_proteins))

    # Extract the cogs that are relevant for the dataset
    spec_mask = cogs['proteins'].isin(spec_proteins)
    sample_cogs = cogs[spec_mask]

    # Make a coder and decoder between protein-IDs and COG-IDs
    prot_id = list(sample_cogs['proteins'].values)
    cog_vals = list(sample_cogs['cog_group'].values)
    cog_map = dict(zip(prot_id, cog_vals))
    rev_cog_map = dict(zip(cog_vals, prot_id))
    return cog_map, rev_cog_map


def generate_cog_labels(x, cog_map):
    """Appends COG labels as column to x

    :param x: x-data
    :type x: pandas DataFrame object
    :param cog_map: cog encoder
    :type cog_map: dict
    :return: original x annotated with COG group
    :rtype: pandas DataFrame object
    """
    # # Check input variables are correct types (must go above any local var defs!)
    # for i, (k, v) in enumerate(locals().items()):
    #     input_types = [pd.core.frame.Series, str]
    #     assert(isinstance(v, input_types[i])), 'Input {} must be of type {}. Got {} instead'.format(
    #         k, input_types[i], type(v))
    # # End check

    x_cogs = []
    for proteins in x.index:
        try:
            p1, p2 = proteins.split('and')
            cog_pair = sorted([cog_map[p1], cog_map[p2]])
            x_cogs.append(cog_pair)
        except Exception as e:
            # print(e)
            x_cogs.append('?')

    # Populate the dataframe with the COGs
    x_cogs = [tuple(x) for x in x_cogs]
    x['cogs'] = x_cogs
    # Remove any unidentified cogs
    x = x[x.cogs != ('?',)]
    return x


def sample_cogs(x, class_label=1, shuffle=True):
    """Samples from x without replacement.

    :param x: x-data
    :type x: pandas DataFrame object
    :param class_label: positive class (1) or negative class (0), defaults to 1
    :type class_label: int, optional
    :param shuffle: to shuffle, defaults to True
    :type shuffle: bool, optional
    :return: a sample of the original x
    :rtype: pandas DataFrame object
    """
    # Shuffle and sort
    if shuffle:
        x = x.sample(frac=1)

    x = x.sort_values('cogs', ascending=False)
    # Drop duplicates
    x = x.drop_duplicates(subset='cogs', keep='first')
    # Append label
    x['labels'] = [class_label] * np.shape(x)[0]
    return x


def split_on_cogs(x, y, cog_map, neg_ratio=4, train_ratio=0.8):
    """Split the data to guarentee no overlap in COG groups.

    :param x: data
    :type x: pandas DataFrame

    :param y: labels
    :type y: pandas DataFrame or pandas Series
    :param cog_map: COG encoder
    :type cog_map: dict
    :param neg_ratio: ratio of desired negative examples, defaults to 4
    :type neg_ratio: int, optional
    :param train_ratio: ratio of desired train observations, defaults to 0.8
    :type train_ratio: float, optional
    :return: positive-train, positive-test, negative-train, negative-test
    :rtype: tulple of pandas DataFrame objects
    """

    # Split into positive and negative sets
    x_pos = deepcopy(x[y.labels == 1])
    x_neg = deepcopy(x[y.labels == 0])

    # Add COG labels to x dataframe
    x_pos = generate_cog_labels(x_pos, cog_map)
    x_neg = generate_cog_labels(x_neg, cog_map)

    # Sample for positive train cogs
    x_pos = sample_cogs(x_pos, class_label=1)
    x_neg = sample_cogs(x_neg, class_label=0)

    # Subset the positive data into train-test
    k_train = int(train_ratio * np.shape(x_pos)[0])
    pos_train = x_pos.iloc[:k_train, :]
    pos_test = x_pos.iloc[k_train+1:, :]

    # Subset the negative data into train-test
    neg_train = x_neg.iloc[:neg_ratio*k_train, :]
    neg_test = x_neg.iloc[-k_train-1:, :]

    return pos_train, pos_test, neg_train, neg_test


def format_data(data, labels, drop_homology=True):
    """Makes certain that only observations with class labels are present,
        sets index as protein pair names.

    :param data: x_data
    :type data: pandas DataFrame
    :param labels: class labels {0:negatives, 1:positives, 2:no-membership}
    :type labels: int
    :param drop_homology: to drop the homology column, defaults to True
    :type drop_homology: bool, optional
    :return: returns filtered x
    :rtype: pandas DataFrame object
    """
    x = data.copy()
    y = labels.copy()

    # Remove verbose labels (rows with no pathway membership: 2)
    x['labels'] = y
    x = x[x['labels'] != 2]
    y = pd.DataFrame(x.labels, columns=['labels'])

    # Drop the labels and other non-appropriate columns
    idx = ["and".join([x.protein1.values[i], x.protein2.values[i]])
           for i in range(0, len(x.protein1))]
    cols_to_drop = ['protein1', 'protein2', 'combined_score', 'labels']
    x.drop(columns=cols_to_drop, axis=1, inplace=True)

    # Drop homology column
    if drop_homology:
        x.drop(columns=['homology'], axis=1, inplace=True)

    # Re-index
    x.index = idx
    y.index = idx
    return x, y, idx


def pre_process_data(data, labels, balance=True):
    """Balances positive and negative classes, drop labels from x-data

    :param data: x-data
    :type data: pandas DataFrame
    :param labels: y-labels
    :type labels: iterable
    :param balance: to balance class ratios, defaults to True
    :type balance: bool, optional
    :return: processed x-data, processed y-labels
    :rtype: tuple of pandas DataFrame object and array
    """
    x = data.copy()
    labels = labels.copy()
    x['labels'] = labels

    # Split into class sets
    x_pos = x[x.labels == 1.0]
    x_neg = x[x.labels == 0.0]

    if balance:
        # Balance the negative dataset via sampling
        n_pos = int(np.shape(x_pos)[0])
        x_neg = x_neg.sample(n=n_pos)

        # Concatenate the DFs
        x = pd.concat([x_pos, x_neg])

    # Extract and drop the labels column
    labels = pd.concat([x_pos.labels, x_neg.labels])
    labels = x.labels
    x.drop(columns=['labels'], inplace=True)
    return x, labels


def combine_datasets(Xs=[], ys=[], idxs=[]):
    """Combine all of the organism datasets

    :param Xs: all x-datas, defaults to []
    :type Xs: list, optional
    :param ys: all y-labels, defaults to []
    :type ys: list, optional
    :param idxs: protein pair names as index list, defaults to []
    :type idxs: list, optional
    :return: x-data of concatenated Xs, y-labels of concatenated ys
    :rtype: tuple of pandas DataFrame objects
    """
    # Concatenate all of the data
    x = pd.concat([i.copy() for i in Xs], axis=0)
    y = pd.concat([j.copy() for j in ys], axis=0)

    # What is this line doing?s
    idxs = np.sum(idxs, dtype=object)
    x.index = idxs
    y.index = idxs
    return x, y


def model_splits(x, labels, test_ratio):
    """Splits each x and y set into train and test data respectively

    :param x: x-data
    :type x: pandas DataFrame
    :param labels: y-labels
    :type labels: iterable
    :param test_ratio: proportion of observations for testing
    :type test_ratio: float
    :return: train-test splits for both x-data and y-data
    :rtype: tuple of pandas DataFrame objects
    """

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        labels, test_size=test_ratio,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def scale_features(dtrain, dtest):
    """Scale the datasets

    :param dtrain: train x-data
    :type dtrain: pandas DataFrame
    :param dtest: test x-data
    :type dtest: pandas DataFrame
    :return: scaled datasets
    :rtype: tuple 2x pandas DataFrames, and the scaler object
    """
    mms = MinMaxScaler()
    x_train_sc = pd.DataFrame(mms.fit_transform(dtrain))
    x_test_sc = pd.DataFrame(mms.transform(dtest))
    return x_train_sc, x_test_sc, mms


def build_model(param, class_ratio=1):
    """Build XBoost model

    :param param: param dict
    :type param: dict
    :param class_ratio: how many times to upscale positive class, defaults to 1
    :type class_ratio: int, optional
    :return: model
    :rtype: sklearn-wrapped XGBoost model
    """
    # Define the model
    clf = xgb.XGBClassifier(
        **param, verbosity=0, scale_pos_weight=class_ratio, use_label_encoder=False)

    return clf


def fit(clf, x_train, y_train, x_test, y_test):
    """Fit the model

    :param clf: the model
    :type clf: model object
    :param x_train: x-train data
    :type x_train: pandas DataFrame
    :param y_train: y-test
    :type y_train: pandas DataFrame object
    :return: fitted model
    :rtype: model object
    """
    # Cast labels as ints
    y_train = [int(x) for x in y_train]
    y_test = [int(x) for x in y_test]

    # Fit the model
    clf.fit(x_train, y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            eval_metric='auc',
            verbose=True)
    return clf


def predict(clf, x_test, y_test):
    """Predict on the unknown observations,

    :param clf: XGBoost model
    :type clf: model object
    :param x_test: test-data
    :type x_test: pandas DataFrame
    :param y_test: test-labels
    :type y_test: iterable
    :return: predictions, probabilities, accuracy, and the model
    :rtype: mixed tuple.
    """

    # Evaluation
    eval_result = clf.evals_result()
    probas = clf.predict_proba(x_test)
    preds = clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy: {}".format(acc))
    return clf, preds, probas, acc, eval_result


def plot_roc(y_test, probas, plot=False):
    """Plots an ROC curve

    :param y_test: y-test labels
    :type y_test: iterable
    :param probas: class probabilities
    :type probas: list
    :return: auc score and visualize plots
    :rtype: float
    """

    # Get the positive class probabilities of the classifier
    fpr, tpr, _ = roc_curve(y_test, probas[:, 1])

    # Get auc stats
    roc_auc = auc(fpr, tpr)

    if plot:
        # Plot AUC
        lw = 1
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.show()
    return roc_auc


def inject_noise(x, mu=0.0, sigma=0.5):
    """Injects noise to feature vectors

    :param x: feature vector
    :type x: np.array
    :param mu: noise mean, defaults to 0.0
    :type mu: float, optional
    :param sigma: noise variance, defaults to 0.5
    :type sigma: float, optional
    :return: returns positive valued x with added noise
    :rtype:
    """
    if ~isinstance(x, np.ndarray):
        x = np.array(x)
    noise = np.random.normal(mu, sigma, np.shape(x))
    return np.abs(x + noise)


def save_outputs_benchmark(x, probas, sid='511145', direc='benchmark/cog_predictions', model_name='.single'):
    """Reformats model outputs for compatibility with Damians benchmark script - saves to disk.

    :param x: x-data to predict on
    :type x: pandas DataFrame
    :param probas: class probabilies pf x
    :type probas: numpy array
    :param sid: species identifier, defaults to '511145'
    :type sid: str, optional
    :param direc: directory to save predictions to, defaults to 'benchmark/cog_predictions'
    :type direc: str, optional
    :param model_name: name of the model, defaults to '.single'
    :type model_name: str, optional
    :return: DataFrame with columns useful for benchmark script
    :rtype: pandas DataFrame
    """
    p1 = [xi.split('and')[0] for xi in list(x.index)]
    p2 = [xi.split('and')[-1] for xi in list(x.index)]
    proba_1 = [x[1] for x in probas]
    df = pd.DataFrame({'model': 'xgboost', 'spec': sid, 'protein_1': [x.split(
        '.')[-1] for x in p1], 'protein_2': [x.split('.')[-1] for x in p2], 'proba_1': proba_1})
    path = os.path.join(direc, '{}.{}.{}.tsv'.format(
        model_name, sid, 'xgboost'))
    df.to_csv(path, index=False, header=False, sep='\t')
    return df


def generate_quality_json(model_name, direct, sid='9606'):

    json_report = {
        "run_title": "xgboost_v11.5.{}.{}".format(model_name, sid),
        "output_filename_plots": "{}/{}.{}.scores.pdf".format(direct, model_name, sid),
        "output_filename_data": "{}/{}.{}.scores.data.tsv".format(direct, model_name, sid),
        "output_filename_errors": "{}/{}.{}.scores.error.tsv".format(direct, model_name, sid),
        "valid_proteins_file": "data/valid_{}.tsv".format(sid),
        "organisms_to_report": [sid],
        "benchmarking_file": "data/kegg_benchmarking.CONN_maps_in.v11.tsv",
        "samples": [
            {
                "name": "combined_v11.5",
                "color": "red",
                "line": "solid",
                "data_file": "data/{}.combined.v11.5.tsv".format(sid)
            },

            {
                "name": "xgboost_single",
                "color": "blue",
                "line": "solid",
                "data_file": "{}/{}.{}.xgboost.tsv".format(direct, model_name, sid)
            }
        ]
    }

    file_name = os.path.join(
        direct, "quality_full_{}.{}.json".format(model_name, sid))
    with open(file_name, 'w') as f:
        json.dump(json_report, f)
    return json_report


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
    clf = build_model(params, class_ratio=weights)

    # Predict
    clf = fit(clf, x_train, y_train, x_test, y_test)
    clf, preds, probas, acc, _ = predict(clf, x_test, y_test)

    # Perform cross validation scoring
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    cv_results = xgb.cv(dtrain=dtrain, params=params, nfold=5,
                        metrics="auc", as_pandas=True, seed=123)
    print(cv_results)

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

    parser.add_argument('-c', '--cogs', type=bool, metavar='',
                        required=True, default=True, help='to split on cogs or not')

    parser.add_argument('-cw', '--class_weight', type=float, metavar='',
                        required=True, default=4, help='factor applied to positive predictions')

    parser.add_argument('-un', '--use_noise', type=bool, metavar='',
                        required=True, default=False, help='if True, injects noise to X')

    parser.add_argument('-nr', '--neg_ratio', type=int, metavar='',
                        required=True, default=4, help='factor increase in neg obs compared to pos obs')

    parser.add_argument('-dh', '--drop_homology', type=bool, metavar='',
                        required=True, default=True, help='if True, drops homology feature')

    parser.add_argument('-sid', '--species_id', type=str, metavar='',
                        required=True, default='511145 9606 4932', help='ids of species to include sepr=' '')

    parser.add_argument('-o', '--output_dir', type=str, metavar='',
                        required=True, default='benchmark/cog_predictions', help='directory to save outputs to')

    # To format data
    FORMAT = True

    # Parse args
    args = parser.parse_args()
    model_name = args.model_name
    use_cogs = args.cogs
    weights = args.class_weight
    use_noise = args.use_noise
    neg_ratio = args.neg_ratio
    drop_homology = args.drop_homology
    species_id = args.species_id
    output_dir = os.path.join(args.output_dir, model_name)


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

# Ecoli
if '511145' in species_id:
    print("Computing for Ecoli")
    ecoli_path = 'data/511145.protein.links.full.v11.5.txt'
    ecoli_label_path = 'data/ecoli_labels.csv'
    ecoli = pd.read_csv(ecoli_path, header=0, sep=' ', low_memory=False)
    ecoli_labels = pd.read_csv(ecoli_label_path, index_col=False, header=None)

    # Format the data
    if FORMAT:
        x_ecoli, y_ecoli, ecoli_idx = format_data(
            ecoli, ecoli_labels, drop_homology=drop_homology)

    t1 = time.time()

    ecoli_output = run_pipeline(data=x_ecoli, labels=y_ecoli, spec_kegg=ecoli, cogs=use_cogs,
                                params=params, weights=weights, species_id='511145.', noise=use_noise, neg_ratio=neg_ratio)
    t2 = time.time()

    print("Finished training in {}".format(t2-t1))
    clf = ecoli_output['classifier']
    ecoli_probas = clf.predict_proba(x_ecoli.iloc[:, :])
    ecoli_preds = clf.predict(x_ecoli.iloc[:, :])
    hold_out_ecoli_probas = clf.predict_proba(ecoli_output['X_HO'])
    hold_out_ecoli_preds = clf.predict(ecoli_output['X_HO'])

    # Save benchmarks
    ecoli_outs = save_outputs_benchmark(x=x_ecoli, probas=ecoli_probas,  sid='511145',
                                        direc=output_dir, model_name=model_name)

    hold_out_ecoli_outs = save_outputs_benchmark(x=ecoli_output['X_HO'], probas=hold_out_ecoli_probas,
                                                 sid='511145', direc=output_dir,
                                                 model_name=model_name + '.hold_out')
    # Generate quality reports
    json_report = generate_quality_json(
        model_name=model_name, direct=output_dir, sid='511145')

    hold_out_json_report = generate_quality_json(
        model_name=model_name + '.hold_out', direct=output_dir, sid='511145')

    # Call Damian benchark script here
    print('Running benchmark')
    command = ['perl'] + ['compute_summary_statistics_for_interact_files.pl'] + \
        ["{}/quality_full_{}.{}.json".format(
            output_dir, model_name, '511145')]
    out = subprocess.run(command)

    command = ['perl'] + ['compute_summary_statistics_for_interact_files.pl'] + \
        ["{}/quality_full_{}.hold_out.{}.json".format(
            output_dir, model_name, '511145')]
    out = subprocess.run(command)


# Human
if '9606' in species_id:
    print("Computing for Human")
    human_path = 'data/9606.protein.links.full.v11.5.txt'
    human_label_path = 'data/human_labels.csv'
    human = pd.read_csv(human_path, header=0, sep=' ', low_memory=False)
    human_labels = pd.read_csv(human_label_path, index_col=False, header=None)

    # Format the data
    if FORMAT:
        x_human, y_human, human_idx = format_data(
            human, human_labels, drop_homology=drop_homology)

    t1 = time.time()
    human_output = run_pipeline(data=x_human, labels=y_human, spec_kegg=human, cogs=use_cogs,
                                params=params, weights=weights, species_id='9606.', noise=use_noise,  neg_ratio=neg_ratio)
    t2 = time.time()

    clf = human_output['classifier']
    human_probas = clf.predict_proba(x_human.iloc[:, :])
    human_preds = clf.predict(x_human.iloc[:, :])
    hold_out_human_probas = clf.predict_proba(human_output['X_HO'])
    hold_out_preds = clf.predict(human_output['X_HO'])

    # Save benchmarks
    human_outs = save_outputs_benchmark(x=x_human, probas=human_probas,  sid='9606',
                                        direc=output_dir, model_name=model_name)

    hold_out_human_outs = save_outputs_benchmark(x=human_output['X_HO'], probas=hold_out_human_probas,
                                                 sid='9606', direc=output_dir, model_name=model_name + '.hold_out')

    # Generate quality reports
    json_report = generate_quality_json(
        model_name=model_name, direct=output_dir, sid='9606')

    hold_out_json_report = generate_quality_json(
        model_name=model_name + '.hold_out', direct=output_dir, sid='9606')

    print('Running benchmark')
    command = ['perl'] + ['compute_summary_statistics_for_interact_files.pl'] + \
        ["{}/quality_full_{}.{}.json".format(
            output_dir, model_name, '9606')]
    out = subprocess.run(command)

    command = ['perl'] + ['compute_summary_statistics_for_interact_files.pl'] + \
        ["{}/quality_full_{}.hold_out.{}.json".format(
            output_dir, model_name, '9606')]
    out = subprocess.run(command)


# Yeast
if '4932' in species_id:
    print("Computing for Yeast")
    yeast_path = 'data/4932.protein.links.full.v11.5.txt'
    yeast_label_path = 'data/yeast_labels.csv'
    yeast = pd.read_csv(yeast_path, header=0, sep=' ', low_memory=False)
    yeast_labels = pd.read_csv(yeast_label_path, index_col=False, header=None)

    # Format the data
    if FORMAT:
        x_yeast, y_yeast, yeast_idx = format_data(
            yeast, yeast_labels, drop_homology=drop_homology)

    t1 = time.time()
    yeast_output = run_pipeline(data=x_yeast, labels=y_yeast, spec_kegg=yeast, cogs=use_cogs,
                                params=params, weights=weights, species_id='4932.', noise=use_noise,  neg_ratio=neg_ratio)
    t2 = time.time()

    clf = yeast_output['classifier']
    yeast_probas = clf.predict_proba(x_yeast.iloc[:, :])
    yeast_preds = clf.predict(x_yeast.iloc[:, :])
    hold_out_yeast_probas = clf.predict_proba(yeast_output['X_HO'])
    hold_out_yeast_preds = clf.predict(yeast_output['X_HO'])

    # Save benchmarks
    yeast_outs = save_outputs_benchmark(x=x_yeast, probas=yeast_probas,  sid='4932',
                                        direc=output_dir, model_name=model_name)

    hold_out_yeast_outs = save_outputs_benchmark(x=yeast_output['X_HO'], probas=hold_out_yeast_probas,
                                                 sid='4932', direc=output_dir, model_name=model_name + '.hold_out')

    # Generate quality reports
    json_report = generate_quality_json(
        model_name=model_name, direct=output_dir, sid='4932')

    hold_out_json_report = generate_quality_json(
        model_name=model_name + '.hold_out', direct=output_dir, sid='4932')

    # Call benchark script here
    print('Running benchmark')
    command = ['perl'] + ['compute_summary_statistics_for_interact_files.pl'] + \
        ["{}/quality_full_{}.{}.json".format(
            output_dir, model_name, '4932')]
    out = subprocess.run(command)

    command = ['perl'] + ['compute_summary_statistics_for_interact_files.pl'] + \
        ["{}/quality_full_{}.hold_out.{}.json".format(
            output_dir, model_name, '4932')]
    out = subprocess.run(command)
