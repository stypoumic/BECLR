from pathlib import Path
import shutil
import argparse
from tqdm import tqdm
import csv
import os


def args_parser():
    parser = argparse.ArgumentParser(
        'tieredImageNet dataset creation arguments', add_help=False)

    parser.add_argument('--data_path', type=str,
                        help="Path to original ImageNet train dataset folder")
    parser.add_argument('--split_path', type=str,
                        help="Path to tieredImageNet split files")
    parser.add_argument('--save_path', type=str,
                        help="Path to new tieredImageNet train dataset folder arranged in this way by default: /\
                            root/dog/xxx.png /\
                            root/dog/xxy.png /\
                            ... /\
                            root/cat/123.png /\
                            root/cat/124.png"
                        )
    return parser


class tieredImageNet_Split(object):
    def __init__(self, data_path, split_path, save_path):
        self.data_root = data_path
        self.save_path = save_path
        self.kept_imgs = []
        self.kept_labels = []

        file_path_train = Path(split_path) / \
            Path('tieredImageNet/train.csv')
        file_path_val = Path(split_path) / \
            Path('tieredImageNet/val.csv')
        file_path_test = Path(split_path) / \
            Path('tieredImageNet/test.csv')

        train_imgs, train_labels, train_labels_name = self._read_csv(
            file_path_train)
        self.kept_imgs.extend(train_imgs)
        self.kept_labels.extend(train_labels_name)

        val_imgs, val_labels, val_labels_name = self._read_csv(file_path_val)
        self.kept_imgs.extend(val_imgs)
        self.kept_labels.extend(val_labels_name)

        test_imgs, test_labels, test_labels_name = self._read_csv(
            file_path_test)
        self.kept_imgs.extend(test_imgs)
        self.kept_labels.extend(test_labels_name)
        print("Total Classes Kept: {}".format(len(self.kept_labels)))

        # delete all ImageNet images that are not used by train/val/test
        # splits of tieredImageNet
        counter = 0
        for subdirectory in tqdm(os.listdir(Path(self.data_root))):
            if subdirectory in self.kept_labels:
                counter += len(os.listdir(Path(self.data_root) /
                               Path(subdirectory)))
            else:
                shutil.rmtree(Path(self.data_root) / Path(subdirectory))
        print("Total Images Kept: {}".format(counter))

        # create tieredImageNet train folder in required format
        self.create_split_folder(train_imgs, train_labels, train_labels_name)
        # remove redundant classes of train split
        for subdirectory in tqdm(os.listdir(Path(self.data_root))):
            if subdirectory in train_labels_name:
                shutil.rmtree(Path(self.data_root) / Path(subdirectory))

    def _read_csv(self, file_path):
        imgs = []
        labels = []
        labels_name = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                img, label = row[0], row[1]
                img = Path(self.data_root) / Path(img)
                imgs.append(img)
                if label not in labels_name:
                    labels_name.append(label)
                labels.append(labels_name.index(label))
        return imgs, labels, labels_name

    def create_split_folder(self, imgs, labels, labels_name):
        save_path = Path(self.save_path)
        print("tieredImageNet save directory: {}".format(save_path))
        print("Number of Classes: {}".format(len(labels_name)))
        print("Number of Images: {}".format(len(imgs)))
        for i in range(len(labels_name)):
            (save_path / Path(str(i))).mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(imgs))):
            shutil.copy(Path(imgs[i]),
                        save_path / Path(str(labels[i])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'tieredImageNet dataset creation arguments', parents=[args_parser()])
    args = parser.parse_args()

    miniImageNetsplit = tieredImageNet_Split(
        args.data_path, args.split_path, args.save_path)
