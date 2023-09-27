import argparse
import csv
import shutil
from pathlib import Path

from tqdm import tqdm


def args_parser():
    parser = argparse.ArgumentParser(
        'miniImageNet dataset creation arguments', add_help=False)

    parser.add_argument('--data_path', type=str,
                        help="Path to original miniImageNet dataset folder")
    parser.add_argument('--split_path', type=str,
                        help="Path to miniImageNet split file")
    parser.add_argument('--save_path', type=str,
                        help="Path to new miniImageNet dataset folder arranged in this way by default: /\
                            root/dog/xxx.png /\
                            root/dog/xxy.png /\
                            ... /\
                            root/cat/123.png /\
                            root/cat/124.png"
                        )
    parser.add_argument('--partition', type=str, default="train", choices=["val", "train", "test"],
                        help="miniImageNet partition")

    return parser


class miniImageNet_Split(object):
    def __init__(self, data_path, split_path, save_path, partition='train'):
        self.data_root = data_path
        self.partition = partition
        self.save_path = save_path

        file_path = Path(split_path) / \
            Path('miniImageNet/{}.csv'.format(self.partition))
        self.imgs, self.labels, self.labels_name = self._read_csv(file_path)

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
                img = Path(self.data_root) / Path('images/{}'.format(img))
                imgs.append(img)
                if label not in labels_name:
                    labels_name.append(label)
                labels.append(labels_name.index(label))
        return imgs, labels, labels_name

    def create_split_folder(self):
        save_path = Path(self.save_path)
        print("miniImageNet save directory: {}".format(save_path))
        print("Number of Classes: {}".format(len(self.labels_name)))
        print("Number of Images: {}".format(len(self.imgs)))
        for i in range(len(self.labels_name)):
            (save_path / Path(str(i))).mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(len(self.imgs))):
            shutil.copy(Path(self.imgs[i]),
                        save_path / Path(str(self.labels[i])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'miniImageNet dataset creation arguments', parents=[args_parser()])
    args = parser.parse_args()

    miniImageNetsplit = miniImageNet_Split(
        args.data_path, args.split_path, args.save_path, args.partition)

    miniImageNetsplit.create_split_folder()
