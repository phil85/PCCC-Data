from utilities import generate_hard_constraints


def generate_constraint_sets_for_collection_3():
    datasets = ['banana', 'cifar10', 'cifar100', 'letter', 'mnist', 'shuttle']

    for dataset in datasets:
        for param_nf in [0, 0.005, 0.01, 0.05]:
            generate_hard_constraints('COL3', dataset, param_nf, 24)

    print('Constraint sets for collection COL3 sucessfully generated.')

