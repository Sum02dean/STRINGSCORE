import sys
import os
import pytest
import numpy as np
src_path = os.getcwd().replace('/tests', '')
sys.path.append(src_path)
from string_utils import *

# Begin making test
def test_generate_random_hash():
    hash_list = []
    for _ in range(100):
        r, hash_list = generate_random_hash(hash_list)
        # Assert that all elements are unique
        assert len(np.unique(hash_list)) == len(hash_list)


def test_create_cog_map():
    species_dict = {'511145': 'ecoli', '9606': 'human', '4932': 'yeast'}
    species_id = '4932'
    for (species, species_name) in species_dict.items():
        if species in species_id:

            # Pull in some data
            spec_path = '../data/{}.protein.links.full.v11.5.txt'.format(species)
            data = pd.read_csv(spec_path, header=0, sep=' ', low_memory=False)
            species_id = '{}.'.format(species)
            cog_map = create_cog_map(spec_kegg=data, species_id=species_id)   

            # Create the cog DF
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

