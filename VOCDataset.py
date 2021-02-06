import torch
from torch.utils.data import Dataset
from torchvision import datasets as datasets

from Keys import Keys


class VOCDataset(Dataset):
    def __init__(self, root, year, image_set, transform=None):
        self.dataset = datasets.VOCDetection(root=root, year=year, image_set=image_set, download=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.dataset[idx]
        image = item[0]
        annotations = item[1][Keys.ANNOTATION][Keys.OBJECT]
        print("before ")
        print(annotations)
        new_annotations = []
        # there can be multiple annotations
        for annotation in annotations:
            bndbox = annotation["bndbox"]
            x_min = float(bndbox[Keys.X_MIN])
            x_max = float(bndbox[Keys.X_MAX])
            y_min = float(bndbox[Keys.Y_MIN])
            y_max = float(bndbox[Keys.Y_MAX])
            annotation["bndbox"] = {Keys.CENTER_X: (x_min + x_max) / 2,
                                    Keys.CENTER_Y: (y_min + y_max) / 2,
                                    Keys.WIDTH: x_max - x_min,
                                    Keys.HEIGHT: y_max - y_min}
            new_annotations.append(annotation)
        print("before ")
        print(new_annotations)

        sample = {Keys.IMAGE: image, Keys.ANNOTATION: new_annotations}

        if self.transform:
            sample = self.transform(sample)
        return sample