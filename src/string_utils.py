""" This is where I keep all of my functions for the STRING project """
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, recall_score, precision_score, average_precision_score
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from sklearn.metrics import accuracy_score
from collections import Counter as C
import seaborn as sns
from tqdm import tqdm
import copy
import json


def generate_random_hash(hash_list):
    """Generates a random hash if a COG has no formal grouping

    :param hash_list: list of hash's already used
    :type hash_list: list
    :return: random hash, new appended hash_list containing generated hash
    :rtype: string, list.
    """
    r = ''.join(random.choices(string.ascii_lowercase, k=10))
    while r in hash_list:
        r = ''.join(random.choices(string.ascii_lowercase, k=10))
    hash_list.append(r)
    return r, hash_list


def create_cog_map(spec_kegg, species_id='9606.'):
    """Creates a mapping between all the protein IDs in a dataset
    and their respective COG IDs.

    :param spec_kegg: species specific KEGG data
    :type spec_kegg: pandas.core.Dataframe
    :param species_id: organism of interest(don't forget the '.'), defaults to '9606.'
    :type species_id: str, optional
    :return: coder, decoder
    :rtype: dict, dict
    """

    # Filter the COGs for a specific organism
    cog_path = '/mnt/mnemo2/damian/dean/ogs.luca.core.tsv'
    cog_df = pd.read_csv(cog_path, sep='\t', header=0)
    cog_df.columns = ['idk', 'cog_group', 'proteins']
    cogs = cog_df[cog_df['proteins'].str.contains(species_id)]

    # For all data in species subset create a mapping
    spec_proteins = list(spec_kegg.protein1.values) + \
        list(spec_kegg.protein2.values)
    spec_proteins = list(set(spec_proteins))

    # Extract the cogs that are relevant for the dataset
    sample_cogs = cogs[cogs['proteins'].isin(spec_proteins)]

    # Make a coder and decoder between protein-IDs and COG-IDs
    prot_id = list(sample_cogs['proteins'].values)
    cog_vals = list(sample_cogs['cog_group'].values)

    cog_map = dict(zip(prot_id, cog_vals))
    rev_cog_map = dict(zip(cog_vals, prot_id))
    return cog_map, rev_cog_map


def generate_cog_labels(x, cog_map):
    """Appends COG labels as column to x. each COG pairs is sorted list [COGA-COGB]

    :param x: x-data - x.index must be protein names - else run format_data()
    :type x: pandas.core.DataFrame object
    :param cog_map: COG encoder which maps proteins to respective COG groups
    :type cog_map: dict
    :return: original x annotated with COG group, were COGS are formated as tuple
    :rtype: pandas DataFrame object
    """

    x_cogs = []
    hash_list = []
    for proteins in x.index:
        try:
            p1, p2 = proteins.split('and')
            cog_pair = sorted([cog_map[p1], cog_map[p2]])
            x_cogs.append(cog_pair)

        # If COG group not listed - create random unique GOC hash
        except Exception as e:
            r1, hash_list = generate_random_hash(hash_list)
            r2, hash_list = generate_random_hash(hash_list)
            r1 = "".join('hash-' + r1)
            r2 = "".join('hash-' + r2)
            x_cogs.append([r1, r2])

    # Populate the dataframe with the COGs
    x_cogs = [tuple(x) for x in x_cogs]
    x['cogs'] = x_cogs
    return x


def sample_cogs(x, class_label=1, shuffle=True, test_ratio=0.5):
    """Samples from x without replacement.

    :param x: x-data
    :type x: pandas.core.DataFrame object
    :param class_label: positive class (1) or negative class (0), defaults to 1
    :type class_label: int, optional
    :param shuffle: to shuffle, defaults to True
    :type shuffle: bool, optional
    :return: a sample of the original x
    :rtype: pandas.core.DataFrame object.
    """
    x = copy.deepcopy(x)
    # Shuffle and sort
    if shuffle:
        x = x.sample(frac=1)

    # Collect and remove duplicates from train set
    dup_idx = x.cogs.duplicated()
    x_duplicates = x[dup_idx]
    x_filtered = x.drop_duplicates(subset='cogs', keep='first')

    # Sample test set from set of train observations
    x_test = x_filtered.sample(frac=test_ratio)

    # Recombine duplicates with filtered x after test split
    x_train = pd.concat([x_filtered, x_duplicates])

    # Drop any instanes where test COGs are present in train COGs
    x_train_new = x_train[~x_train.cogs.isin(x_test.cogs)].dropna()

    # # Append label
    x_train_new['labels'] = [class_label] * np.shape(x_train_new)[0]
    return x_train_new, x_test


def split_on_cogs(x, y, cog_map, neg_ratio=4, test_ratio=0.2):
    """Split the data to guarentee no overlap in COG groups.

    :param x: data
    :type x: pandas.core.DataFrame

    :param y: labels
    :type y: pandas.core.DataFrame or pandas.core.Series
    :param cog_map: COG encoder mapping proteins to COGs
    :type cog_map: dict
    :param neg_ratio: ratio of desired negative examples, defaults to 4
    :type neg_ratio: int, optional
    :param train_ratio: ratio of desired train observations, defaults to 0.8
    :type train_ratio: float, optional
    :return: positive-train, positive-test, negative-train, negative-test
    :rtype: tulple of pandas.core.DataFrame objects
    """

    # Split into positive and negative sets
    x_pos = copy.deepcopy(x[y.labels == 1])
    x_neg = copy.deepcopy(x[y.labels == 0])

    # Add COG labels to x dataframe
    x_pos = generate_cog_labels(x_pos, cog_map)
    x_neg = generate_cog_labels(x_neg, cog_map)

    # Upscale based on neg-pos class ratios
    k_pos = 1
    k_neg = neg_ratio * k_pos

    # Sample from respective datasets
    x_pos_sample = x_pos.sample(k_pos*np.shape(x_pos)[0])
    x_neg_sample = x_neg.sample(k_neg*np.shape(x_pos)[0])

    # Generate train-test splits
    pos_train, pos_test = sample_cogs(
        x_pos_sample, class_label=1, test_ratio=test_ratio)
    neg_train, neg_test = sample_cogs(
        x_neg_sample, class_label=0, test_ratio=test_ratio)

    return pos_train, pos_test, neg_train, neg_test


def format_data(data, labels, drop_homology=True):
    """Makes certain that only observations with class labels are present,
        sets index as protein pair names.

    :param data: x_data
    :type data: pandas.core.DataFrame
    :param labels: class labels {0:negatives, 1:positives, 2:no-membership}
    :type labels: int
    :param drop_homology: to drop the homology column, defaults to True
    :type drop_homology: bool, optional
    :return: returns filtered x with protein pair names as index
    :rtype: pandas.core.DataFrame object
    """
    x = data.copy()
    y = labels.copy()

    # Remove verbose labels (rows with no pathway membership: 2)
    x['labels'] = y
    x = x[x['labels'] != 2]
    y = pd.DataFrame(x.labels, columns=['labels'])

    # Create row indices
    idx = ["and".join([x.protein1.values[i], x.protein2.values[i]])
           for i in range(0, len(x.protein1))]

    # Drop the labels and other non-appropriate columns
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
    :type data: pandas.core.DataFrame
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
    :type Xs: list of pandas.core.Dataframes, optional
    :param ys: all y-labels, defaults to []
    :type ys: list of pandas.core.DataFrame of pandas.core.Series, optional
    :param idxs: protein pair names as index list, defaults to []
    :type idxs: list of index lists, optional
    :return: x-data of concatenated Xs, y-labels of concatenated ys
    :rtype: tuple of pandas.core.DataFrame objects
    """
    # Concatenate all of the data
    x = pd.concat([i.copy() for i in Xs], axis=0)
    y = pd.concat([j.copy() for j in ys], axis=0)

    # Quick way of concaenating lists of strings
    idxs = np.sum(idxs, dtype=object)
    x.index = idxs
    y.index = idxs
    return x, y


def model_splits(x, labels, test_ratio):
    """Splits each x and y set into train and test data respectively

    :param x: x-data
    :type x: pandas.core.DataFrame
    :param labels: y-labels
    :type labels: iterable e.g list, or pandas.core.Series
    :param test_ratio: proportion of observations for testing
    :type test_ratio: float
    :return: train-test splits for both x-data and y-data
    :rtype: tuple of pandas DataFrame objects
    """

    # Split the dataset using scikit learn implimentation
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        labels, test_size=test_ratio,
                                                        random_state=42)
    return x_train, x_test, y_train, y_test


def scale_features(dtrain, dtest):
    """Scale the datasets

    :param dtrain: train x-data
    :type dtrain: pandas.core.DataFrame
    :param dtest: test x-data
    :type dtest: pandas.core.DataFrame
    :return: scaled dtrain and dtest datasets
    :rtype: tuple 2x pandas.core.DataFrames, sklearn Scaler() object
    """
    mms = MinMaxScaler()
    x_train_sc = pd.DataFrame(mms.fit_transform(dtrain))
    x_test_sc = pd.DataFrame(mms.transform(dtest))
    return x_train_sc, x_test_sc, mms


def build_model(param, class_ratio=1):
    """Build XBoost model

    :param param: param dict containing all of the model parameters
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
    :type x_train: pandas.core.DataFrame object
    :param y_test: y-test
    :type y_test: pandas.core.DataFrame object
    :return: fitted clf model
    :rtype: clf model object
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
    """Predict on the unknown observations

    :param clf: XGBoost model
    :type clf: model object
    :param x_test: test-data
    :type x_test: pandas.core.DataFrame object
    :param y_test: test-labels
    :type y_test: iterable e.g. list or pandas.core.Series object
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
    :type y_test: iterable e.g list
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


def save_outputs_benchmark(x, probas, sid='511145', direc='benchmark/cog_predictions', model_name='.single', hold_out=False):
    """Reformats model outputs for compatibility with Damians benchmark script - saves to disk.

    :param x: x-data to predict on
    :type x: pandas.core.DataFrame object
    :param probas: class probabilies of x
    :type probas: numpy.array()
    :param sid: species identifier, defaults to '511145'
    :type sid: str, optional
    :param direc: directory to save predictions to, defaults to 'benchmark/cog_predictions'
    :type direc: str, optional
    :param model_name: name of the model, defaults to '.single'
    :type model_name: str, optional
    :return: DataFrame with columns useful for benchmark script
    :rtype: pandas.core.DataFrame object
    """
    # Not implimented yet
    if hold_out:
        pass

    p1 = [xi.split('and')[0] for xi in list(x.index)]
    p2 = [xi.split('and')[-1] for xi in list(x.index)]
    proba_1 = [x[1] for x in probas]
    df = pd.DataFrame({'model': 'xgboost', 'spec': sid, 'protein_1': [x.split(
        '.')[-1] for x in p1], 'protein_2': [x.split('.')[-1] for x in p2], 'proba_1': proba_1})
    path = os.path.join(direc, '{}.{}.{}.tsv'.format(
        model_name, sid, 'xgboost'))
    df.to_csv(path, index=False, header=False, sep='\t')
    return df


def generate_quality_json(model_name, direct, sid='9606', hold_out=False):
    """Gerneates the quality json file to use for Damaians benchmark script

    :param model_name: Name of the model
    :type model_name: str
    :param direct: directory to save the json to
    :type direct: str
    :param sid: specied identifier, defaults to '9606'
    :type sid: str, optional
    :param hold_out: weather the report is about a hold-out set of not, defaults to False
    :type hold_out: bool, optional
    :return: json report and saves json to specifies directory path
    :rtype: dictionary
    """
    if hold_out:
        model_name += '.hold_out'
        benchmark_file = "{}/hold_out.{}.combined.v11.5.tsv".format(
            direct, sid)
    else:
        benchmark_file = "data/{}.combined.v11.5.tsv".format(sid)

    json_report = {
        "run_title": "xgboost_v11.5.{}.{}".format(model_name, sid),
        "output_filename_plots": "{}/{}.{}.scores.pdf".format(direct, model_name, sid),
        "output_filename_data": "{}/{}.{}.scores.data.tsv".format(direct, model_name, sid),
        "output_filename_errors": "{}/{}.{}.scores.error.tsv".format(direct, model_name, sid),
        "valid_proteins_file": "data/valid_{}.tsv".format(sid),
        "organisms_to_report": [int(sid)],
        "benchmarking_file": "data/kegg_benchmarking.CONN_maps_in.v11.tsv",
        "samples": [
            {
                "name": "combined_v11.5",
                "color": "red",
                "line": "solid",
                "data_file": benchmark_file
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


def get_interesction(target, reference):
    """Returns a new STRING benchmark dataset where the obersations are those mutual refenence data (formated
       using save_output_benchmark() as expected for Damians benchmark script). 
       This allows the combined STRING score dataset to be filtered to include only observations 
       mutual to the computed hold-out set.

    :param target: the hold-out model dataset
    :type target: pandas.core.DataFrame
    :param reference: the STRING score dataset
    :type reference: pandas.core.DataFrame
    :return: New refernce dataset containing only observations mutual to hold-out set
    :rtype: pandas.core.DataFrame object
    """
    # Add a new column which combines the string names to both dataframes
    target = copy.deepcopy(target)
    reference = copy.deepcopy(reference)
    names_target = []
    for row in target.values:
        _, _, p1, p2, _ = row
        p_name = ".".join([p1, p2])
        names_target.append(p_name)

    names_ref = []
    for row in ref.values:
        model, org, p1, p2, score = row
        p_name = ".".join([p1, p2])
        names_ref.append(p_name)

    col_names = ['model', 'org', 'p1', 'p2', 'proba']
    target.columns = col_names
    reference.columns = col_names
    target['names'] = np.array(names_target)
    reference['names'] = np.array(names_ref)
    intersect = reference[reference.names.isin(target.names)]
    intersect.reset_index(inplace=True, drop=True)
    return intersect


def isolate_non_zero_feature(data, predictions, foi='experiments'):
    """Sample data which only contains non-zero elements for a given feature of interest (FOI)

    :param data: organism data
    :type data: pandas.core.DataFrame object
    :param predictions: predicted probability scores for positive class
    :type predictions: iterable e.g. list
    :param foi: feature of interest, only feature with non-zero elements, defaults to 'experiments'
    :type foi: str, optional
    :return: data where all rows are zero except for FOI column
    :rtype: pandas.core.DataFrame object
    """
    # Make deepcopy of data
    data_copy = copy.deepcopy(data)
    # Filter dataset to exclude feature of interest (foi)
    data_exc = data_copy.loc[:, (data_copy.columns != foi)]
    # Extract indices were all row elements evaluate to zero
    idxes = data_exc.loc[(data_exc == 0).all(axis=1)].index
    # # Use these indices to extract all zero elements where foi is non-zero in original data_copy
    data_copy['labels'] = predictions
    foi_data = data_copy.loc[idxes, :]
    return foi_data
