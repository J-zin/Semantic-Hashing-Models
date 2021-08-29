import pickle
import scipy.io
import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset

class Data:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_datasets()

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shuffle_train)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False) \
                                 if get_test else None
        return train_loader, val_loader, test_loader
    
    def load_datasets(self):
        raise NotImplementedError


class LabeledDocuments(Data):
    def __init__(self, file_path):
        super().__init__(file_path=file_path)

    def load_datasets(self):
        dataset = scipy.io.loadmat(self.file_path)
        
        # (num documents) x (vocab size) tensors containing tf-idf values
        X_train = torch.from_numpy(dataset['train'].toarray()).float()
        X_val = torch.from_numpy(dataset['cv'].toarray()).float()
        X_test = torch.from_numpy(dataset['test'].toarray()).float()

        # (num documents) x (num labels) tensors containing {0,1}
        Y_train = torch.from_numpy(dataset['gnd_train']).float()
        Y_val = torch.from_numpy(dataset['gnd_cv']).float()
        Y_test = torch.from_numpy(dataset['gnd_test']).float()

        self.train_dataset = TensorDataset(X_train, Y_train)
        self.val_dataset = TensorDataset(X_val, Y_val)
        self.test_dataset = TensorDataset(X_test, Y_test)

        self.vocab_size = self.train_dataset[0][0].size(0)
        self.num_labels = self.train_dataset[0][1].size(0)