import os
from download_raw_data import download_raw_data_of_collection_1, download_raw_data_of_collection_4, \
    download_constraint_sets_of_collection_1
from prepare_data_sets_of_collection_1 import prepare_collection_1
from prepare_data_sets_of_collection_2 import prepare_collection_2
from prepare_data_sets_of_collection_3 import prepare_collection_3
from prepare_data_sets_of_collection_4 import prepare_collection_4
from prepare_constraint_sets_for_collection_1 import prepare_constraint_sets_for_collection_1, \
    generate_noisy_constraint_sets_for_collection_1, generate_additional_constraint_sets_for_collection_1
from generate_constraint_sets_for_collection_2 import generate_constraint_sets_for_collection_2
from generate_constraint_sets_for_collection_3 import generate_constraint_sets_for_collection_3
from generate_constraint_sets_for_collection_4 import generate_constraint_sets_for_collection_4
from utilities import check_constraint_sets

# %% Create folders and download raw data
data_not_yet_downloaded = False
if data_not_yet_downloaded:
    os.makedirs('raw data/COL1/constraint sets')
    os.makedirs('raw data/COL4')
    os.makedirs('processed data/COL1/constraint sets')
    os.makedirs('processed data/COL1/noisy constraint sets')
    os.makedirs('processed data/COL2/constraint sets')
    os.makedirs('processed data/COL3/constraint sets')
    os.makedirs('processed data/COL4/constraint sets')

    download_raw_data_of_collection_1()
    download_raw_data_of_collection_4()
    download_constraint_sets_of_collection_1()

# %% Prepare data sets
prepare_collection_1()
prepare_collection_2()
prepare_collection_3()
prepare_collection_4()

# %% Prepare constraint sets
prepare_constraint_sets_for_collection_1()
generate_additional_constraint_sets_for_collection_1()

# %% Generate constraint sets
generate_constraint_sets_for_collection_2()
generate_constraint_sets_for_collection_3()
generate_constraint_sets_for_collection_4()

# %% Generate noisy constraint sets
generate_noisy_constraint_sets_for_collection_1(['appendicitis', 'moons', 'zoo'])

# %% Check constraint sets
check_constraint_sets('overview_constraints.xlsx')

