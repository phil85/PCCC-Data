import shutil
import requests
import zipfile
import tarfile
import gdown
import gzip
import os

# Store urls of data sets (valid as of April 2022)
links_to_data_sets = {
    'appendicitis': "https://sci2s.ugr.es/keel/dataset/data/classification/appendicitis.zip",
    'banana': "https://sci2s.ugr.es/keel/dataset/data/classification/banana.zip",
    'breast_cancer':
        "https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset",
    'bupa': "https://sci2s.ugr.es/keel/dataset/data/classification/bupa.zip",
    'cifar-10': "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    'cifar-100': "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    'circles': "https://raw.githubusercontent.com/GermangUgr/DILS_CC/master/ToyDatasets/circles_toydataset.txt",
    'ecoli': "https://sci2s.ugr.es/keel/dataset/data/classification/ecoli.zip",
    'glass': "https://sci2s.ugr.es/keel/dataset/data/classification/glass.zip",
    'haberman': "https://sci2s.ugr.es/keel/dataset/data/classification/haberman.zip",
    'hayesroth': "https://sci2s.ugr.es/keel/dataset/data/classification/hayes-roth.zip",
    'heart': "https://sci2s.ugr.es/keel/dataset/data/classification/heart.zip",
    'ionosphere': "https://sci2s.ugr.es/keel/dataset/data/classification/ionosphere.zip",
    'iris': "https://sci2s.ugr.es/keel/dataset/data/classification/iris.zip",
    'led7digit': "https://sci2s.ugr.es/keel/dataset/data/classification/led7digit.zip",
    'letter': "https://sci2s.ugr.es/keel/dataset/data/classification/letter.zip",
    'mnist': ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
              "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
              "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
              "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"],
    'monk2': "https://sci2s.ugr.es/keel/dataset/data/classification/monk-2.zip",
    'moons': "https://raw.githubusercontent.com/GermangUgr/DILS_CC/master/ToyDatasets/moons_toydataset.txt",
    'movement_libras': "https://sci2s.ugr.es/keel/dataset/data/classification/movement_libras.zip",
    'newthyroid': "https://sci2s.ugr.es/keel/dataset/data/classification/newthyroid.zip",
    'saheart': "https://sci2s.ugr.es/keel/dataset/data/classification/saheart.zip",
    'shuttle': "https://sci2s.ugr.es/keel/dataset/data/classification/shuttle.zip",
    'sonar': "https://sci2s.ugr.es/keel/dataset/data/classification/sonar.zip",
    'soybean': "https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data",
    'spectfheart': "https://sci2s.ugr.es/keel/dataset/data/classification/spectfheart.zip",
    'spiral': "https://raw.githubusercontent.com/GermangUgr/DILS_CC/master/ToyDatasets/spiral_toydataset.txt",
    'tae': "https://sci2s.ugr.es/keel/dataset/data/classification/tae.zip",
    'vehicle': "https://sci2s.ugr.es/keel/dataset/data/classification/vehicle.zip",
    'wine': "https://sci2s.ugr.es/keel/dataset/data/classification/wine.zip",
    'zoo': "https://sci2s.ugr.es/keel/dataset/data/classification/zoo.zip"}


def download_raw_data_of_collection_1():
    path = 'raw data/COL1/'

    zip_files = ['appendicitis', 'bupa', 'ecoli', 'glass', 'haberman', 'hayesroth', 'heart', 'ionosphere', 'iris',
                 'led7digit', 'monk2', 'movement_libras', 'newthyroid', 'saheart', 'sonar', 'spectfheart', 'tae',
                 'vehicle', 'wine', 'zoo']

    for name in zip_files:
        # Download file
        response = requests.get(links_to_data_sets[name])
        file_name = path + name + '_data.zip'
        open(file_name, 'wb').write(response.content)

        # Extract zip file
        zipfile.ZipFile(file_name, 'r').extractall(path)

        # Delete zip file
        os.remove(file_name)

        # Print progress
        print('Data set', name, 'successfully downloaded.')

    text_files = ['circles', 'moons', 'spiral']

    for name in text_files:
        # Download file
        response = requests.get(links_to_data_sets[name])
        file_name = path + name + '_toydataset.txt'
        open(file_name, 'wb').write(response.content)

        # Print progress
        print('Data set', name, 'successfully downloaded.')

    data_files = ['soybean']

    for name in data_files:
        # Download file
        response = requests.get(links_to_data_sets[name])
        file_name = path + name + '-small.data'
        open(file_name, 'wb').write(response.content)

        # Print progress
        print('Data set', name, 'successfully downloaded.')


def download_raw_data_of_collection_4():
    path = 'raw data/COL4_tmp/'

    zip_files = ['banana', 'letter', 'shuttle']

    for name in zip_files:
        # Download file
        response = requests.get(links_to_data_sets[name])
        file_name = path + name + '_data.zip'
        open(file_name, 'wb').write(response.content)

        # Extract zip file
        zipfile.ZipFile(file_name, 'r').extractall(path)

        # Delete zip file
        os.remove(file_name)

        # Print progress
        print('Data set', name, 'successfully downloaded.')

    cifar_files = ['cifar-10', 'cifar-100']

    for name in cifar_files:
        # Download file
        response = requests.get(links_to_data_sets[name])
        file_name = path + name + '-python.tar.gz'
        open(file_name, 'wb').write(response.content)

        # Open file
        file = tarfile.open(file_name)

        # Extract file
        file.extractall(path)

        file.close()

        # Delete tar file
        os.remove(file_name)

        # Print progress
        print('Data set', name, 'successfully downloaded.')

    # Download mnist data
    links = links_to_data_sets['mnist']
    for link in links:
        response = requests.get(link)
        file_name = path + link.split('/')[-1]
        open(file_name, 'wb').write(response.content)

        # Extract file
        with gzip.open(file_name, 'rb') as f_in:
            with open(file_name.split('.')[0], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Delete gz file
        os.remove(file_name)


    # Print progress
    print('Data set', 'mnist', 'successfully downloaded.')


def download_constraint_sets_of_collection_1():
    path = 'raw data/COL1/constraint sets/'

    gd_links = {
        'appendicitis(0.05).txt': "https://drive.google.com/file/d/1yMIMUA_1Vlkzsi2LFsPG6qOVv78PgGVL/view?usp=sharing",
        'appendicitis(0.10).txt': "https://drive.google.com/file/d/1You1yga0zG-VQa9woA1d6Kbum0y2vU8A/view?usp=sharing",
        'appendicitis(0.15).txt': "https://drive.google.com/file/d/1yW8XCrX9b4I-gGZ2wEpO9HMyYMZ1oeA1/view?usp=sharing",
        'appendicitis(0.20).txt': "https://drive.google.com/file/d/1FFc5YyMKb5cwIAU-HDM9t8yXDggZUc9w/view?usp=sharing",

        'breast_cancer(0.05).txt': "https://drive.google.com/file/d/18B7Pkl4wzHQcNBaVbzX6S69EmdRpXz7d/view?usp=sharing",
        'breast_cancer(0.10).txt': "https://drive.google.com/file/d/13rLc36aNn-pMHEgvTp1zGWKTS7J0bq_X/view?usp=sharing",
        'breast_cancer(0.15).txt': "https://drive.google.com/file/d/1qRo7Sszjb5X3QwCRX2wMQq5HWVGORhWB/view?usp=sharing",
        'breast_cancer(0.20).txt': "https://drive.google.com/file/d/1AKx74RSx54GUTY-X4r3OqNAgb3JBrJMr/view?usp=sharing",

        'bupa(0.05).txt': "https://drive.google.com/file/d/1hFX0FNm2TAHttFxVqcrCHcCBtyRbtJg7/view?usp=sharing",
        'bupa(0.10).txt': "https://drive.google.com/file/d/1NbzataVVOnnt3FKVaU2t8DLJz8K4XTxL/view?usp=sharing",
        'bupa(0.15).txt': "https://drive.google.com/file/d/1_pRy1cC2UvM9k_qxXYzKJ9BOcSimpYYG/view?usp=sharing",
        'bupa(0.20).txt': "https://drive.google.com/file/d/1edMoMsD1JP9RCFh0QjJjbFH9BeZl-IdY/view?usp=sharing",

        'circles(0.05).txt': "https://drive.google.com/file/d/1DKeYCVzQkqRP260Rj81Aqj2YIPd6gzGO/view?usp=sharing",
        'circles(0.10).txt': "https://drive.google.com/file/d/14BN_GhJS2UrGbVrO2nF2c6Hdw1UFU7X3/view?usp=sharing",
        'circles(0.15).txt': "https://drive.google.com/file/d/1rCKUKrV54hJm-9vknX8sQHIWBXbLhyX8/view?usp=sharing",
        'circles(0.20).txt': "https://drive.google.com/file/d/1ls71FrBZ2l8pB5YU5_ZDNoBubEdXrpSN/view?usp=sharing",

        'ecoli(0.05).txt': "https://drive.google.com/file/d/12XktmcvF6nEudypouFNwCVZ1XF5bx37i/view?usp=sharing",
        'ecoli(0.10).txt': "https://drive.google.com/file/d/1GGRdWiUmAZObXqILgA8chCPM4oxGFZZV/view?usp=sharing",
        'ecoli(0.15).txt': "https://drive.google.com/file/d/1hGVBn64ijvO_4DVmp3y7bLLWApq_zKot/view?usp=sharing",
        'ecoli(0.20).txt': "https://drive.google.com/file/d/1Y5LiY3R8P6INeGsIwUTvHh1X2ADwrju5/view?usp=sharing",

        'glass(0.05).txt': "https://drive.google.com/file/d/1u2yUCrcHPTMOpcMt8UK6TJv4ZVWLIlg4/view?usp=sharing",
        'glass(0.10).txt': "https://drive.google.com/file/d/1JnJRXeM7KCQSHkMSeXzm7zSI3vggWUnE/view?usp=sharing",
        'glass(0.15).txt': "https://drive.google.com/file/d/1xChVDfOoteSU0gliTBQu8YxrNBGIvZbn/view?usp=sharing",
        'glass(0.20).txt': "https://drive.google.com/file/d/18HC2wnsxiF55-zRJ61NGlGt_RBeFSWzh/view?usp=sharing",

        'haberman(0.05).txt': "https://drive.google.com/file/d/1bZP5L5Th3xiy7gVVjUPWXZzS5mp0YrhK/view?usp=sharing",
        'haberman(0.10).txt': "https://drive.google.com/file/d/1LfI4c9CCic0OVsSDQHjh1k2ogw7G-lG3/view?usp=sharing",
        'haberman(0.15).txt': "https://drive.google.com/file/d/1yPziHTYqZP8R3OFR7tqNEwBMXDu-ObLv/view?usp=sharing",
        'haberman(0.20).txt': "https://drive.google.com/file/d/1K_UaT5yLa148JzrGu_AXiAEujMK1PjAk/view?usp=sharing",

        'hayes-roth(0.05).txt': "https://drive.google.com/file/d/1oXiU2a8dgZJpbb48kGWOMavRpK270n6R/view?usp=sharing",
        'hayes-roth(0.10).txt': "https://drive.google.com/file/d/1b8Ep1HCczruzEItlueL0y8wyPhcSso2y/view?usp=sharing",
        'hayes-roth(0.15).txt': "https://drive.google.com/file/d/1S8EsdtbIlN_bKwx-uLCd3-srDqPHpyOU/view?usp=sharing",
        'hayes-roth(0.20).txt': "https://drive.google.com/file/d/17WsJ6BJEv9fRHUh1YWEOb5UhJJLcLHte/view?usp=sharing",

        'heart(0.05).txt': "https://drive.google.com/file/d/1WMZ2jTXClE-cQtvHDTAhQZ9EEVtK5xEh/view?usp=sharing",
        'heart(0.10).txt': "https://drive.google.com/file/d/1aHef6U_IdAlJDx7XgFl6cfaU0PXDOJeY/view?usp=sharing",
        'heart(0.15).txt': "https://drive.google.com/file/d/1hQFncZ0DDXssx0boTBX4SlxSMVwkSpWd/view?usp=sharing",
        'heart(0.20).txt': "https://drive.google.com/file/d/12K7yEp4LaBXygAHDfS_bGf2ctxpuuc3s/view?usp=sharing",

        'ionosphere(0.05).txt': "https://drive.google.com/file/d/1w2IIRpdhtYeVumi4dyw7RRG0Te5HZXiz/view?usp=sharing",
        'ionosphere(0.10).txt': "https://drive.google.com/file/d/16JtnsCWqQug8iUF01dSMHqrdvzBBxzFL/view?usp=sharing",
        'ionosphere(0.15).txt': "https://drive.google.com/file/d/1LSplOwWYZtKccGqmRsPt6qLn1GoFM8_Q/view?usp=sharing",
        'ionosphere(0.20).txt': "https://drive.google.com/file/d/1kDixIbr-7tSMTbHklHSnsSeMhW_t18Fb/view?usp=sharing",

        'iris(0.05).txt': "https://drive.google.com/file/d/1q-tYqh5yRGnX7iFRNO7ZxVxIrRaGOx3S/view?usp=sharing",
        'iris(0.10).txt': "https://drive.google.com/file/d/1TZxKl10dEf-VeCzXqg0g3QLORScSALOi/view?usp=sharing",
        'iris(0.15).txt': "https://drive.google.com/file/d/1saIAjIc7fTflRo0eoB2jMh2W8JSaWR9D/view?usp=sharing",
        'iris(0.20).txt': "https://drive.google.com/file/d/1BZZWf5EIPt_aQd3nT_GlKdjMcOGaMStT/view?usp=sharing",

        'led7digit(0.05).txt': "https://drive.google.com/file/d/1t-pEQAH254IvoqMXRbfoZKgKei-Mlfjg/view?usp=sharing",
        'led7digit(0.10).txt': "https://drive.google.com/file/d/1pPpgRoPs-SB6z_KyPqqFrPje-r4eIrFP/view?usp=sharing",
        'led7digit(0.15).txt': "https://drive.google.com/file/d/1ymC9yzQccEFg1fFvQnmkZ7DENa9jOu6Y/view?usp=sharing",
        'led7digit(0.20).txt': "https://drive.google.com/file/d/1_WemVStB7fwocvYXAXJaoTsk0sZJFPOI/view?usp=sharing",

        'monk-2(0.05).txt': "https://drive.google.com/file/d/1h45tSTEHcqo0CYfPCv_sTeUVWks0YH5y/view?usp=sharing",
        'monk-2(0.10).txt': "https://drive.google.com/file/d/1ToLWfGPkLkaAaLOqMQs0VZFWmeDsqWoB/view?usp=sharing",
        'monk-2(0.15).txt': "https://drive.google.com/file/d/1IJsFaFUeYBM5juza273qp8NnkZTT4-a5/view?usp=sharing",
        'monk-2(0.20).txt': "https://drive.google.com/file/d/1apRaC4R0Kuj8iM0Z11AN34BRBDSU2bV2/view?usp=sharing",

        'moons(0.05).txt': "https://drive.google.com/file/d/1SXuVdHRZyhIcsqXIq9gznj8jFS5Oyw-E/view?usp=sharing",
        'moons(0.10).txt': "https://drive.google.com/file/d/15EN1CWVzoruR-b7bjiYHKxcHCWT_u6r4/view?usp=sharing",
        'moons(0.15).txt': "https://drive.google.com/file/d/143q1Z8PZQpI1xLnri84LH-XD_iAdacOo/view?usp=sharing",
        'moons(0.20).txt': "https://drive.google.com/file/d/1SrcIKahyZfuAR4zxYsV6VLb07dzz-eKK/view?usp=sharing",

        'movement_libras(0.05).txt': "https://drive.google.com/file/d/1PJqdCXKgHygkf64WrFjx7MbkKr6y_oxg/view?usp=sharing",
        'movement_libras(0.10).txt': "https://drive.google.com/file/d/1O5EdY5nclSX3levzmOMvyYxoRbqxpMHq/view?usp=sharing",
        'movement_libras(0.15).txt': "https://drive.google.com/file/d/1zLNX4g10MdgBiyfgR6egielXrwHNdu2f/view?usp=sharing",
        'movement_libras(0.20).txt': "https://drive.google.com/file/d/1wERc4T7UqvNwGrOINWPp9VnlDEoj21OZ/view?usp=sharing",

        'newthyroid(0.05).txt': "https://drive.google.com/file/d/13PkMBxjtTzo9DQojtliqsP-4Lah_cV6F/view?usp=sharing",
        'newthyroid(0.10).txt': "https://drive.google.com/file/d/1FAKIJkbl7nFe5ACSJVv0XWx0ljmXOEs8/view?usp=sharing",
        'newthyroid(0.15).txt': "https://drive.google.com/file/d/1pYTVSOlYiRvX1ihoTkGStXYc1lj-MkMJ/view?usp=sharing",
        'newthyroid(0.20).txt': "https://drive.google.com/file/d/1wRALSQPXjd1Khp72OwRei-qg-oXION49/view?usp=sharing",

        'saheart(0.05).txt': "https://drive.google.com/file/d/1bR0_Df8k6EEajWoF0lGcFrtS7qYfUKbG/view?usp=sharing",
        'saheart(0.10).txt': "https://drive.google.com/file/d/1qHFIs_Izhdgu-8TU2Z1CKw41sJtOfLX4/view?usp=sharing",
        'saheart(0.15).txt': "https://drive.google.com/file/d/1Qq9YBZyaiILWzVAD6BS8PmcBQboIDZXR/view?usp=sharing",
        'saheart(0.20).txt': "https://drive.google.com/file/d/1ucRynAIL2qHuiFsFY_9C8v7jZ4Gts7l0/view?usp=sharing",

        'sonar(0.05).txt': "https://drive.google.com/file/d/1DHD77WhXrfsEX5rxN4V7GQCKD4tzpulz/view?usp=sharing",
        'sonar(0.10).txt': "https://drive.google.com/file/d/1ay7FykAJHh7Uv3mmAULfuR7AxI-8IzjI/view?usp=sharing",
        'sonar(0.15).txt': "https://drive.google.com/file/d/1Ocl2vghKdiIxuLV5yNB1EjtBFjAp6pJA/view?usp=sharing",
        'sonar(0.20).txt': "https://drive.google.com/file/d/1n8hvijKVHd6yAIfd9yD-PXYrvTdbO1m7/view?usp=sharing",

        'soybean(0.05).txt': "https://drive.google.com/file/d/1GqOzwSUnUw0fwgDrFSaP4_Oe_mObngRC/view?usp=sharing",
        'soybean(0.10).txt': "https://drive.google.com/file/d/1vOPyuh7nZJosMfL7JPgjvZ9p0MluGkXh/view?usp=sharing",
        'soybean(0.15).txt': "https://drive.google.com/file/d/1v2P8lC1w0lxLA9WOkPjNNUjRbDz5QzSB/view?usp=sharing",
        'soybean(0.20).txt': "https://drive.google.com/file/d/1D5uCZEuB1QXzS1m60yifpvitKjBcwQxH/view?usp=sharing",

        'spectfheart(0.05).txt': "https://drive.google.com/file/d/1oZIVZWGbS-ju87xDB60XnBaSk_yl5dq2/view?usp=sharing",
        'spectfheart(0.10).txt': "https://drive.google.com/file/d/1g1Jq46TcM9n06UmogbWP9Flp1mLnpwoB/view?usp=sharing",
        'spectfheart(0.15).txt': "https://drive.google.com/file/d/1yvOU7N1VEMbJ1gArwUkejhko5Bn-S7Nd/view?usp=sharing",
        'spectfheart(0.20).txt': "https://drive.google.com/file/d/1s5U7pxoaS6C2MX3vJ8fxj_Vw55EtRGNp/view?usp=sharing",

        'spiral(0.05).txt': "https://drive.google.com/file/d/1ZJ5WBWG46xAK8-UtfFT0EBJfJ4kUvi-I/view?usp=sharing",
        'spiral(0.10).txt': "https://drive.google.com/file/d/19NF_gCmZKVaxWEC7gTxIk8_ZlnopdXVv/view?usp=sharing",
        'spiral(0.15).txt': "https://drive.google.com/file/d/1hReH2fQAWp1J2FeFrNxe3kcHBKY49o3k/view?usp=sharing",
        'spiral(0.20).txt': "https://drive.google.com/file/d/16CxVbW2LeCStMLRFJQKtZO8WoNh4_5vn/view?usp=sharing",

        'tae(0.05).txt': "https://drive.google.com/file/d/1TDw-9IWlseTfVLTkT_RWr8B1C8UGFc5c/view?usp=sharing",
        'tae(0.10).txt': "https://drive.google.com/file/d/1Yl3da3Zu7y35JcrIUspNdGjavumNsTXk/view?usp=sharing",
        'tae(0.15).txt': "https://drive.google.com/file/d/1FHQmIgKrI2qMRSvJ1ZhPUI-P6wCSFrOL/view?usp=sharing",
        'tae(0.20).txt': "https://drive.google.com/file/d/1Mf1f1qTbdbiy9PoURr4GwEVtn877-H-i/view?usp=sharing",

        'vehicle(0.05).txt': "https://drive.google.com/file/d/1_A8fdjSsfMhLnC_47bEp72fYbf3Uv-_i/view?usp=sharing",
        'vehicle(0.10).txt': "https://drive.google.com/file/d/1rxdBspwNDBqHLhf3EBFHxGoLgGXeUa2K/view?usp=sharing",
        'vehicle(0.15).txt': "https://drive.google.com/file/d/1STHtnhpv4OYyX7n0vixdJIX3GUwEiiLI/view?usp=sharing",
        'vehicle(0.20).txt': "https://drive.google.com/file/d/1KiA0SsbZ6Cb3DXhJRT3Q4n89KttoRIoK/view?usp=sharing",

        'wine(0.05).txt': "https://drive.google.com/file/d/1bj_mnFYyp3llJjX7XqtRdC_LLCS0uvLW/view?usp=sharing",
        'wine(0.10).txt': "https://drive.google.com/file/d/10UH99W_8SEzI25EIdftYz4Rn6rDnWWe_/view?usp=sharing",
        'wine(0.15).txt': "https://drive.google.com/file/d/1a_LsnxN1kwPK5jaKPd-G8OaDyeoV10Ba/view?usp=sharing",
        'wine(0.20).txt': "https://drive.google.com/file/d/1UjpkuV3MdPCg3hYvYyi8D_1ODnG4zu_y/view?usp=sharing",

        'zoo(0.05).txt': "https://drive.google.com/file/d/1SGd4e6V1MLMRSwfO28hjIL5ymRkjqsKG/view?usp=sharing",
        'zoo(0.10).txt': "https://drive.google.com/file/d/1AjjgTk84484b_7baM1utS6S8q6ryrtCP/view?usp=sharing",
        'zoo(0.15).txt': "https://drive.google.com/file/d/1y4LQ8URMePFBGg_BUuGbfsuTc8S4iK-m/view?usp=sharing",
        'zoo(0.20).txt': "https://drive.google.com/file/d/1WxHB-QYhGa6G3_54IKZZzxMbq0cmeIzd/view?usp=sharing",
    }

    for file_name in gd_links.keys():
        link = gd_links[file_name]
        file_id = link.split('file/d/')[-1].split('/')[0]
        url = "https://drive.google.com/uc?id=" + file_id
        gdown.download(url, path + file_name, quiet=False)
        print(file_name, 'successfully downloaded.')

