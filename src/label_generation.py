from string_utils import *
import pandas as pd
from time import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

""" This script is resonsible for loading in the data and generating labels.
    - If two proteins are found in the same KEGG pathway, assign label 1
    - If two proteins are found in different pathways, assign label 0
    - If one of the protein pathways can not be determined, assign label 2"""

USE_ARGPASE = True
if USE_ARGPASE:
    parser = argparse.ArgumentParser(description='label_generation')

    parser.add_argument('-sid', '--species_id', type=str, metavar='',
                    required=True, default='511145 9606 4932', help="ids of species to include, sepr=' '")
    
    parser.add_argument('-o', '--output_dir', type=str, metavar='',
                    required=True, default='data/', help='directory to save outputs to')
    args = parser.parse_args()
    
    # Collect arguments from parsers
    output_dir = args.output_dir
    species_id = args.species_id

else:
    output_dir = '/data'
    species_id = '511145'


# Generate a for-loop for specified species.
print("Loading in species data")
full_kegg = pd.read_csv('data/kegg_benchmarking.CONN_maps_in.v11.tsv', header=None, sep='\t')
species_dict = {'511145': 'ecoli', '9606': 'human', '4932': 'yeast'}

# Check whether the specified path exists or not
isExist = os.path.exists(output_dir)
if not isExist:
    # Create it
    os.makedirs(output_dir)
    print("{} directory created.".format(output_dir))


# Loop over all  speices given in provided args
for (species, species_name) in species_dict.items():
    if species in species_id:
        print("Generating labels for {}.".format(species_name))
        fn = os.path.join(output_dir,'{}_labels_debug.csv'.format(species_name))
        
        # Generate the labels
        species_x = full_kegg[full_kegg[0] == int(species)]
        species_kegg = pd.read_csv('data/{}.protein.links.full.v11.5.txt'.format(species), sep=' ', header=0)
        ecoli_labels = generate_labels(X=species_kegg, pathway_names=species_x[1], pathway_members=species_x[3], file_name=fn, verbosity=2)
print("Finished generating labels for all.")
        







# # Subset pathway sets for species entries
# ecoli = full_kegg[full_kegg[0] == 511145]
# human = full_kegg[full_kegg[0] == 9606]
# yeast = full_kegg[full_kegg[0] == 4932]

# # Load in all species KEGG sets
# ecoli_kegg = pd.read_csv('511145.protein.links.full.v11.5.txt', sep=' ', header=0)
# human_kegg = pd.read_csv('9606.protein.links.full.v11.5.txt', sep=' ', header=0)
# yeast_kegg = pd.read_csv('4932.protein.links.full.v11.5.txt', sep=' ', header=0)


# # Create a dict with useful variables
# ecoli_labels = generate_labels(X=ecoli_kegg, pathway_names=ecoli[1], pathway_members=ecoli[3], file_name='data/ecoli_labels.csv', verbosity=2)
# yeast_labels = generate_labels(X=yeast_kegg, pathway_names=yeast[1], pathway_members=yeast[3], file_name='data/yeast_labels.csv', verbosity=2)
# human_labels = generate_labels(X=human_kegg, pathway_names=human[1], pathway_members=human[3], file_name='data/human_labels.csv', verbosity=2)

