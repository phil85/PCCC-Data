import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_data_from_dat_file(dataset, folder):
    # Read raw data
    f = open('raw data/' + folder + '/' + dataset + '.dat')
    content = f.readlines()

    # Extract number of objects and number of features
    while content[0][0] == '@':
        content.pop(0)
    n = len(content)
    d = len(content[0].split(','))

    # Extract feature values
    ls = []
    for i in range(n):
        if content[i] == '\n':
            continue
        ls.append([])
        for j in range(d):
            entry = content[i].split(',')[j]
            if '\n' in entry:
                entry = entry.replace('\n', '')
            ls[i].append(entry)
    # Create DataFrame
    df = pd.DataFrame(ls)

    return df


def preprocess_and_export(df, dataset, folder):
    # Convert all columns except last one to float
    df.iloc[:, :-1] = df.iloc[:, :-1].astype('float')

    # Adjust column names
    df.columns = ['x' + str(i) for i in range(len(df.columns) - 1)] + ['class']

    # Encode class column as categorical
    categories = df['class'].unique()
    df['class'] = pd.Categorical(df['class'], categories).codes

    # Standardize features
    scaler = StandardScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])

    # Export processed data set
    df.to_csv('processed data/' + folder + '/' + dataset + '_data.csv', index=False)


def process_dataset(dataset, folder):
    # Read raw data
    df = get_data_from_dat_file(dataset, folder)

    # Perform preprocessing
    preprocess_and_export(df, dataset, folder)


def generate_hard_constraints(folder, dataset, param_nf, random_state):
    # Import data set
    df = pd.read_csv('processed data/' + folder + '/' + dataset + '_data.csv')

    # Extract class labels
    y = df['class']

    # Get number of objects
    n = df.shape[0]

    # Determine number of constraints
    n_f = np.ceil(n * param_nf)
    n_constraints = int((n_f * (n_f - 1) / 2))

    if n_constraints == 0:
        constraints = {'ml': [], 'cl': [], 'sml': [], 'scl': [], 'sml_proba': [], 'scl_proba': []}
        file_name = ('processed data/' + folder + '/constraint sets/' + dataset
                     + '_constraints_' + '{:g}'.format(param_nf) + '.json')
        with open(file_name, 'w') as fp:
            json.dump(constraints, fp)
    else:
        # Set random state
        np.random.seed(random_state)

        counter = 0
        while counter < n_constraints:
            i = np.random.randint(0, n, size=n_constraints)
            j = np.random.randint(0, n, size=n_constraints)
            new_matrix = np.tile((i, j), reps=1).T

            # Sort indices
            min_row = new_matrix.min(axis=1)
            max_row = new_matrix.max(axis=1)

            # Remove entries on diagonal of affinity matrix
            idx = min_row != max_row
            min_row = min_row[idx]
            max_row = max_row[idx]
            new_matrix = np.tile((min_row, max_row), reps=1).T

            if counter == 0:
                current_matrix = np.unique(new_matrix, axis=0)
            else:
                current_matrix = np.unique(np.concatenate((current_matrix, new_matrix)), axis=0)

            counter = current_matrix.shape[0]

        i = current_matrix[:n_constraints, 0]
        j = current_matrix[:n_constraints, 1]

        # Extract ml and cl pairs based on ground truth
        idx_ml = y[i].values == y[j].values
        idx_cl = ~idx_ml
        ml = list(zip(i[idx_ml].tolist(), j[idx_ml].tolist()))
        cl = list(zip(i[idx_cl].tolist(), j[idx_cl].tolist()))

        # Export ml and cl constraints into json file
        constraints = {'ml': ml, 'cl': cl, 'sml': [], 'scl': [], 'sml_proba': [], 'scl_proba': []}
        file_name = ('processed data/' + folder + '/constraint sets/' + dataset
                     + '_constraints_' + '{:g}'.format(param_nf) + '.json')
        with open(file_name, 'w') as fp:
            json.dump(constraints, fp)


def generate_noisy_constraints(folder, dataset, param_nf, random_state, lower_bound):
    df = pd.read_csv('processed data/' + folder + '/' + dataset + '_data.csv')
    y = df['class']
    n = df.shape[0]
    n_f = np.ceil(n * param_nf)
    n_constraints = int((n_f * (n_f - 1) / 2))

    # Set random state
    np.random.seed(random_state)

    counter = 0
    while counter < n_constraints:
        i = np.random.randint(0, n, size=n_constraints)
        j = np.random.randint(0, n, size=n_constraints)
        new_matrix = np.tile((i, j), reps=1).T

        # Sort indices
        min_row = new_matrix.min(axis=1)
        max_row = new_matrix.max(axis=1)
        new_matrix = np.tile((min_row, max_row), reps=1).T

        if counter == 0:
            current_matrix = np.unique(new_matrix, axis=0)
        else:
            current_matrix = np.unique(np.concatenate((current_matrix, new_matrix)), axis=0)

        counter = current_matrix.shape[0]

    i = current_matrix[:n_constraints, 0]
    j = current_matrix[:n_constraints, 1]

    probability_values = lower_bound + (1 - lower_bound) * np.random.rand(n_constraints)

    correct_constraint_type = (y[i].values == y[j].values) * 1
    incorrect_constraint_type = 1 - correct_constraint_type
    random_values = np.random.rand(n_constraints)
    ind_correct = probability_values >= random_values
    ind_incorrect = ~ind_correct
    constraint_type = correct_constraint_type.copy()
    constraint_type[ind_incorrect] = incorrect_constraint_type[ind_incorrect].copy()

    # Extract ml and cl pairs based on ground truth
    idx_sml = constraint_type == 1
    idx_scl = constraint_type == 0
    sml = list(zip(i[idx_sml].tolist(), j[idx_sml].tolist()))
    scl = list(zip(i[idx_scl].tolist(), j[idx_scl].tolist()))
    sml_weights = 2 * (probability_values[idx_sml] - 0.5)
    scl_weights = 2 * (probability_values[idx_scl] - 0.5)

    # Export ml and cl constraints into json file
    constraints = {'ml': [], 'cl': [], 'sml': sml, 'scl': scl, 'sml_proba': sml_weights.tolist(),
                   'scl_proba': scl_weights.tolist()}
    file_name = ('processed data/' + folder + '/noisy constraint sets/' + dataset + '_noisy_constraints_'
                 + '{:.2f}'.format(lower_bound) + '_' + '{:g}'.format(param_nf) + '.json')
    with open(file_name, 'w') as fp:
        json.dump(constraints, fp)


def check_constraint_set(folder, dataset, param_nf):

    s = pd.Series(dtype='int')

    # Import data set
    df = pd.read_csv('processed data/' + folder + '/' + dataset + '_data.csv')

    # Extract class labels
    y = df['class']

    constraints = json.load(open('processed data/' + folder + '/constraint sets/' + dataset + '_constraints'
                                 + '_{:g}'.format(param_nf) + '.json'))
    ml = constraints['ml']
    cl = constraints['cl']
    n_ml = len(ml)
    n_cl = len(cl)
    n_constraints = n_ml + n_cl
    n_correct_ml = len([1 for i, j in ml if y[i] == y[j]])
    n_correct_cl = len([1 for i, j in cl if y[i] != y[j]])
    n_incorrect_ml = len([1 for i, j in ml if y[i] != y[j]])
    n_incorrect_cl = len([1 for i, j in cl if y[i] == y[j]])

    s['dataset'] = dataset
    s['param_nf'] = param_nf
    s['n_ml'] = n_ml
    s['n_cl'] = n_cl
    s['n_constraints'] = n_constraints
    s['n_correct_ml'] = n_correct_ml
    s['n_correct_cl'] = n_correct_cl
    s['incorrect_ml'] = n_incorrect_ml
    s['incorrect_cl'] = n_incorrect_cl

    return s


def check_noisy_constraint_set(folder, dataset, param_nf, confidence_lower_bound):

    s = pd.Series(dtype='int')

    # Import data set
    df = pd.read_csv('processed data/' + folder + '/' + dataset + '_data.csv')

    # Extract class labels
    y = df['class']

    constraints = json.load(open('processed data/' + folder + '/noisy constraint sets/' + dataset + '_noisy_constraints'
                                 + '_{:.2f}'.format(confidence_lower_bound) + '_{:g}'.format(param_nf) + '.json'))
    ml = constraints['ml']
    sml = constraints['sml']
    cl = constraints['cl']
    scl = constraints['scl']
    n_ml = len(ml)
    n_cl = len(cl)
    n_sml = len(sml)
    n_scl = len(scl)
    n_constraints = n_ml + n_cl + n_sml + n_scl
    n_correct_ml = len([1 for i, j in ml if y[i] == y[j]])
    n_correct_cl = len([1 for i, j in cl if y[i] != y[j]])
    n_incorrect_ml = len([1 for i, j in ml if y[i] != y[j]])
    n_incorrect_cl = len([1 for i, j in cl if y[i] == y[j]])
    n_correct_sml = len([1 for i, j in sml if y[i] == y[j]])
    n_correct_scl = len([1 for i, j in scl if y[i] != y[j]])
    n_incorrect_sml = len([1 for i, j in sml if y[i] != y[j]])
    n_incorrect_scl = len([1 for i, j in scl if y[i] == y[j]])

    s['dataset'] = dataset
    s['param_nf'] = param_nf
    s['confidence_lower_bound'] = confidence_lower_bound
    s['n_ml'] = n_ml
    s['n_cl'] = n_cl
    s['n_sml'] = n_sml
    s['n_scl'] = n_scl
    s['n_constraints'] = n_constraints
    s['n_correct_ml'] = n_correct_ml
    s['n_correct_cl'] = n_correct_cl
    s['incorrect_ml'] = n_incorrect_ml
    s['incorrect_cl'] = n_incorrect_cl
    s['n_correct_sml'] = n_correct_sml
    s['n_correct_scl'] = n_correct_scl
    s['incorrect_sml'] = n_incorrect_sml
    s['incorrect_scl'] = n_incorrect_scl

    return s


def check_constraint_sets(file_name):
    df = pd.DataFrame()

    # Check noise-free constraint sets of collection COL1
    datasets = ['appendicitis',
                'breast_cancer',
                'bupa',
                'circles',
                'ecoli',
                'glass',
                'haberman',
                'hayes-roth',
                'heart',
                'ionosphere',
                'iris',
                'led7digit',
                'monk-2',
                'moons',
                'movement_libras',
                'newthyroid',
                'saheart',
                'sonar',
                'soybean',
                'spectfheart',
                'spiral',
                'tae',
                'vehicle',
                'wine',
                'zoo']
    parameters = [0.05, 0.10, 0.15, 0.2]

    for dataset in datasets:
        for param_nf in parameters:
            s = check_constraint_set('COL1', dataset, param_nf)
            df = df.append(s, ignore_index=True)

    # Check noise-free constraint sets of collection COL2
    ls_size = [500, 1000, 2000, 5000]
    ls_features = [2]
    ls_clusters = [2, 5, 10, 20, 50, 100]

    for n_samples in ls_size:
        for n_features in ls_features:
            for n_clusters in ls_clusters:

                # Define name of data set
                dataset = 'synthetic_n{}_d{}_k{}'.format(n_samples, n_features, n_clusters)

                for param_nf in [0.05, 0.10, 0.15, 0.20]:
                    s = check_constraint_set('COL2', dataset, param_nf)
                    df = df.append(s, ignore_index=True)

    # Check noise-free constraint sets of collection COL3
    datasets = ['banana', 'cifar10', 'cifar100', 'letter', 'mnist', 'shuttle']
    parameters = [0.005, 0.01, 0.05]

    for dataset in datasets:
        for param_nf in parameters:
            s = check_constraint_set('COL3', dataset, param_nf)
            df = df.append(s, ignore_index=True)


    # Check noisy constraint sets of collection COL1
    datasets = ['appendicitis', 'moons', 'zoo']
    parameters = [0.05, 0.1, 0.15, 0.2]
    confidence_lower_bounds = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    for dataset in datasets:
        for param_nf in parameters:
            for confidence_lower_bound in confidence_lower_bounds:
                s = check_noisy_constraint_set('COL1', dataset, param_nf, confidence_lower_bound)
                df = df.append(s, ignore_index=True)

    df.to_excel(file_name)

