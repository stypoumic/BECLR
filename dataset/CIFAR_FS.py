import os
import pickle

from PIL import Image
from torch.utils.data import Dataset


class CIFAR_FS(Dataset):
    def __init__(self, data_path, partition='train', transform=None):
        super(Dataset, self).__init__()
        self.data_root = data_path
        self.partition = partition
        self.transform = transform

        file_path = os.path.join(
            self.data_root, 'CIFAR_FS_{}.pickle'.format(self.partition))
        self.imgs, self.labels = self._load_data(file_path)

    def _load_data(self, file_path):
        try:
            with open(file_path, 'rb') as fo:
                data = pickle.load(fo)
        except:
            with open(file_path, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
        return data["data"], data["labels"]

    def __getitem__(self, item):
        img = self.imgs[item]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[item]
        return img, target

    def __len__(self):
        return len(self.labels)
