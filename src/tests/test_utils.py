import sys
import os
import pytest
import numpy as np
src_path = os.getcwd().replace('/tests', '')
sys.path.append(src_path)
from string_utils import *

""" This testing suite will test most of the essential functions for the STRING score script
"""

# Load in example data
species = '511145'
spec_path = '../data/{}.protein.links.full.v11.5.txt'.format(species)
label_path = '../data/{}_labels.csv'.format('ecoli')
data = pd.read_csv(spec_path, header=0, sep=' ', low_memory=False)
labels = pd.read_csv(label_path, index_col=False, header=None)

def test_generate_random_hash():
    """ Test that all generated hashes are unique."""
    hash_list = []
    for _ in range(100):
        _, hash_list = generate_random_hash(hash_list)
        assert len(np.unique(hash_list)) == len(hash_list)

def test_format_data(data=data, labels=labels):
    """ Test that only zeros and ones are present in the list"""
    x, y, idx = format_data(data, labels)
    assert (np.shape(x)[0] == np.shape(y)[0] == np.shape(idx)[0])
    assert (2 not in y.values)
    assert (list(x.index) == idx)

def test_create_cog_map(data=data, species_id=species):
    """ Test that the cogs are mapped correctly"""
    cog_map = create_cog_map(spec_kegg=data, species_id=species_id)   
    cog_path = '/mnt/mnemo2/damian/dean/ogs.luca.core.tsv'
    cog_df = pd.read_csv(cog_path, sep='\t', header=0)
    cog_df.columns = ['idk', 'cog_group', 'proteins']
    cogs = cog_df[cog_df['proteins'].str.contains(species_id)]

    # For all proteins in the cog_map, check if mapping is correct
    for i, (k,v) in enumerate(cog_map.items()):
        prot = cogs['proteins'].values[i]
        cog = cogs['cog_group'].values[i]
        if prot in cog_map.items() and cog in cog_map.items():
            assert cog_map[prot] == cog
    
def test_generate_cog_labels(data=data, labels=labels, species=species):
        """ Test that each protein in a pair has a COG membership"""
        species_id = '{}'.format(species)
        cog_map = create_cog_map(spec_kegg=data, species_id=species_id)
        x, y, idx = format_data(data, labels)

        # Generate labels for proteins with known KEGG pathways
        x = generate_cog_labels(x=x, cog_map=cog_map)
        sampled_data = x.sample(frac=0.25)
        for i in range(len(sampled_data)):
            x = sampled_data.cogs[i]
            assert np.shape(x)[0] == 2

def test_sample_cogs(data=data, labels=labels, species=species):
    """ Test for COG intersection between train and test sets"""
    species_id = '{}'.format(species)
    cog_map = create_cog_map(spec_kegg=data, species_id=species_id)   
    x, _, _ = format_data(data, labels)
    x = generate_cog_labels(x=x, cog_map=cog_map)

    # Split data into train test sets with no overlap in COGS
    train, test = sample_cogs(x=x)
    a = train.cogs.values.tolist()
    b = test.cogs.values.tolist()
    assert bool(set(a) & set(b)) == False


def test_split_on_cogs(data=data, labels=labels, species=species):
    """ Test for COG intersection between train and test sets"""
    species_id = '{}'.format(species)
    cog_map = create_cog_map(spec_kegg=data, species_id=species_id)   
    x, y, _ = format_data(data, labels)
    x = generate_cog_labels(x=x, cog_map=cog_map)

    # Generate train test splits for positive and negatives
    ptr, pte, ntr, nte = split_on_cogs(x=x, y=y, cog_map=cog_map)    
    a = ptr.cogs.values.tolist()
    b = pte.cogs.values.tolist()
    c = ntr.cogs.values.tolist()
    d = nte.cogs.values.tolist()
    assert bool(set(a) & set(b)) == False
    assert bool(set(c) & set(d)) == False
    
def test_scale_features(data=data, labels=labels):
    """Test data is scaled between zero and one""" 
    x, y, _ = format_data(data, labels)
    x_train, x_test, _, _ = model_splits(x=x, labels=y, test_ratio=0.2)
    sc_tr, sc_te, _ = scale_features(x_train, x_test)
    assert int(np.max(np.max(sc_tr))) == 1
    assert int(np.max(np.max(sc_te))) == 1

def test_isolate_non_zero_features(data=data, labels=labels):
    """Test that all columns except FOI and 'barred names' are zero valued"""
    x, y, _ = format_data(data, labels)
    foi_names = x.columns.tolist()
    barred_names = ['predictions', 'labels']

    # Check all columns are zero after FOI removal
    for name in foi_names:
        if name not in barred_names:
            foi_data =  isolate_non_zero_feature(data=x, labels=y, predictions=y, foi=name)
            foi_data.drop(columns=['predictions', 'labels', name], inplace=True)
            max_val = np.max(np.max(foi_data))
            # Some FOIs may not have non-zero elements, in this case returns nans.
            if not(max_val != max_val):
                assert int(max_val) == 0





