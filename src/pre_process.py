import sys
import os
from string_utils import *
from collections import Counter as Counter
import numpy as np 
import pandas as pd
import argparse
import sklearn
from sklearn.preprocessing import StandardScaler


# Extract input variables from Argparse
USE_ARGPASE = True
if USE_ARGPASE:
    parser = argparse.ArgumentParser(description='Pre-proccessor')

    parser.add_argument('-sid', '--species_id', type=str, metavar='',
                    required=True, default='511145 9606 4932', help="ids of species to include, sepr=' '")

    parser.add_argument('-dh', '--drop_homology', type=str, metavar='',
                        required=True, default=True, help='if True, drops homology feature')
   
    parser.add_argument('-nr', '--neg_ratio', type=int, metavar='',
                    required=True, default=4, help='factor increase in neg obs compared to pos obs')

    parser.add_argument('-tr', '--test_ratio', type=float, metavar='',
                    required=True, default=0.8, help='ratio to sample from train set to create test set')
    
    parser.add_argument('-o', '--output_dir', type=str, metavar='',
                    required=True, default='benchmark/cog_predictions', help='directory to save outputs to')

    parser.add_argument('-pp', '--pre_process', type=str, metavar='',
                    required=True, default='False', help='to pre-process train and test splits')



    # Collect command line args
    args = parser.parse_args()
    species_id = args.species_id
    drop_homology = True if args.drop_homology == 'True' else False
    output_dir = os.path.join(args.output_dir)
    neg_ratio = args.neg_ratio
    test_ratio = args.test_ratio
    pre_process = True if args.pre_process == 'True' else False
    print("collected command-line args: {}".format(args))


else:
    # Define vars if not using command-line
    species_id = '511145'
    drop_homology = True
    output_dir = os.path.join('pre_processed_data', 'xgboost')
    neg_ratio = 4
    test_ratio = 0.10
    

# Map species ID to name
species_dict = {'511145': 'ecoli', '9606': 'human', '4932': 'yeast'}
full_kegg_path = 'data/kegg_benchmarking.CONN_maps_in.v11.tsv'
full_kegg = pd.read_csv(full_kegg_path, header=None, sep='\t')

# Check whether the specified path exists or not
isExist = os.path.exists(output_dir)
if not isExist:
    # Create it
    os.makedirs(output_dir)
    print("{} directory created.".format(output_dir))

for (species, species_name) in species_dict.items():
    
    if species in species_id:
        # Execute data correction
        print("Pre-processing data for {}".format(species))
        spec_path = 'data/{}.protein.links.full.v11.5.txt'.format(species)
        label_path = 'data/{}_labels.csv'.format(species_name)
        data = pd.read_csv(spec_path, header=0, sep=' ', low_memory=False)
        labels = pd.read_csv(label_path, index_col=False, header=None)
        x, y, idx  = format_data(data, labels, drop_homology=drop_homology)
        x['labels'] = y.labels


        # Generate and map COGS to protein pairs
        print("Generating COG groups...")
        cog_map = create_cog_map(spec_kegg=data, species_id='{}.'.format(
                            species))
        x_cogs = generate_cog_labels(x, cog_map=cog_map)
        y_cogs = x_cogs.labels
        print("Splitting data based on COG exlusions...")
        x_train, x_val = split_on_cogs_alt(x=x_cogs)

        # All data
        x_all = copy.deepcopy(data)
        x_all.drop(columns=['protein1', 'protein2', 'combined_score', 'homology'], inplace=True)
        ind_all = ["and".join([data.protein1.values[i], data.protein2.values[i]]) 
                for i in range(0, len(data.protein1))]
        x_all['labels'] = labels.values
        x_all.index = ind_all


        if pre_process:

            # Remove string cols
            cols = x_train.columns
            all_cols = x_all.columns

            # All data
            ind_all = x_all.index
            all_labels = x_all['labels'] 
            x_all.drop(columns=['labels'], inplace=True)

            # Train data
            ind_train = x_train.index
            x_labels = x_train['labels'] 
            x_cogs = x_train['cogs']
            x_train.drop(columns=['labels', 'cogs'], inplace=True)

            # Valid data
            ind_val = x_val.index
            v_labels = x_val['labels'] 
            v_cogs = x_val['cogs']
            x_val.drop(columns=['labels', 'cogs'], inplace=True)
            
            # Scale the data
            ss = StandardScaler()
            x_train = pd.DataFrame(ss.fit_transform(x_train))
            x_val = pd.DataFrame(ss.transform(x_val))
            x_all = pd.DataFrame(ss.transform(x_all))

            # Add back cols and indices to train
            x_train['labels'] = x_labels.values
            x_train['cogs'] = x_cogs.values
            x_train.index = ind_train

            # Add back cols and indices to val
            x_val['labels'] = v_labels.values
            x_val['cogs'] = v_cogs.values
            x_val.index = ind_val

            # Add back indices to all
            x_all['labels'] = all_labels.values
            x_all.index = ind_all

            # Add back in the rest of the columns
            x_train.columns = cols
            x_val.columns = cols
            x_all.columns = all_cols
           

        # Save the data to file
        print("Saving data to: {} ...".format(output_dir))
        x_train.to_csv(os.path.join(output_dir, '{}_train.csv'.format(species_name)))
        x_val.to_csv(os.path.join(output_dir, '{}_valid.csv'.format(species_name)))
        x_all.to_csv(os.path.join(output_dir, '{}_all.csv'.format(species_name)))
        print('\n')

print("Finished pre-processing.")

        