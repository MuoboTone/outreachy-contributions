#this script downloads the target dataset and splits it using PyTDC

from tdc.utils import retrieve_label_name_list 
from tdc.single_pred import Tox
label_list = retrieve_label_name_list('Tox21')


def download_data():
    data = Tox(name='Tox21', label_name=label_list[10])
    split = data.get_split() 

    train_data = split['train']
    valid_data = split['valid']
    test_data = split['test']

    # Save all splits
    train = split['train'].to_csv("SR-MMP_train.csv", index=False)
    valid = split['valid'].to_csv("SR-MMP_valid.csv", index=False)
    test = split['test'].to_csv("SR-MMP_test.csv", index=False)

    # Save full unsplit dataset
    full_data = data.get_data()
    SR_MMP_full = full_data.to_csv("SR-MMP_full.csv", index=False)

    #outputs are csv files for each split and the entire dataset

    return train, valid, test, SR_MMP_full


download_data()