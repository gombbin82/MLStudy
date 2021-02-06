import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Set random seeds for reproduction
from Keys import Keys
from Transformers import Rescale, CenterCrop, ToTensor
from VOCDataset import VOCDataset
from YOLODetector import YOLODetector

torch.manual_seed(10012)
np.random.seed(10012)


def init_dataloader():
    print('dataloader')
    # Pascal VOC dataset을 YOLODetector와 앞으로 구현할 Loss function에서 사용가능한 형태로
    # 로드하는 dataloader를 구현해봅시다.
    # https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCDetection
    transform = transforms.Compose([
        Rescale(450),
        CenterCrop(448),
        ToTensor()
    ])

    train_dataset = VOCDataset(
        root='./data/VOC',
        year='2007',
        image_set='train',
        transform=transform
    )
    valid_dataset = VOCDataset(
        root='./data/VOC',
        year='2007',
        image_set='val',
        transform=transform
    )

    print(train_dataset)
    print(valid_dataset)

    # Initialize DataLoaders for training and validation set
    # These DataLoaders help sampling mini batches from the dataset.
    # train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=4, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=1)

    return train_loader, valid_loader


class DetectionLoss(nn.Module):
    def __init__(self):
        super(DetectionLoss, self).__init__()

    def forward(self, predicted_boxes, ground_truth_boxes, predicted_classes, ground_truth_classes):
        # YOLODetector의 출력 양식에 맞게 loss function을 구현해봅시다.
        regression_loss = 0
        classification_loss = 0

        loss = regression_loss + classification_loss
        return loss


def train():
    print('train YOLODetector')
    if torch.cuda.is_available():
        print(f'GPU is available: {torch.cuda.get_device_name(0)}')
        device = 'cuda'
    else:
        print(f'GPU is not available.')
        device = 'cpu'

    train_loader, valid_loader = init_dataloader()

    model = YOLODetector()

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = DetectionLoss()

    # TODO: 저장된 체크포인트가 있다면 로드해서 트레이닝을 이어서 진행하도록 해봅시다.
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

    for epoch in range(1000):
        print(f'\nEpoch: {epoch + 1} (lr: {optimizer.param_groups[0]["lr"]:.5f})')

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            print(f'train batch #{batch_idx}')
            # TODO: Training 파트를 구현해봅시다.
            # 구현할 것: 텐서들을 GPU로 보냄 -> 모델 사용 -> Loss 계산

            # TODO: TensorBoard를 사용해 모니터링을 할 수 있도록 적절한 log를 생성해봅시다.
            # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        for batch_idx, batch in enumerate(valid_loader):
            print(f'valid batch #{batch_idx}')
            # TODO: Validation 파트를 구현해봅시다.
            # 구현할 것: 텐서들을 GPU로 보냄 -> 모델 사용 -> Loss 계산

            # TODO: TensorBoard를 사용해 모니터링을 할 수 있도록 적절한 log를 생성해봅시다.
            # https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

        # TODO: 이번 epoch의 validation loss가 min validation loss보다 작다면 checkpoint를 저장합니다.
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html

    print('\nFinished.')


model = YOLODetector()
# model.to('cuda')
model.eval()
# rst = model(torch.zeros([1, 3, 448, 448], dtype=torch.float32).cuda())
rst = model(torch.zeros([1, 3, 448, 448], dtype=torch.float32))

# init_dataloader()


# Test the data loader

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

def show_plot(image, annotations):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    print("final print(annotation)")
    print(annotations)

    color_map = {}
    rects = []
    for annotation in annotations:

        label = annotation["name"]
        if label in color_map:
            color = color_map[label]
            is_new = False
        else:
            color = [random.random(), random.random(), random.random()]
            color_map[label] = color
            is_new = True

        bndbox = annotation["bndbox"]
        x_min = bndbox[Keys.CENTER_X] - (bndbox[Keys.WIDTH] / 2)
        y_min = bndbox[Keys.CENTER_Y] - (bndbox[Keys.HEIGHT] / 2)
        rect = patches.Rectangle(
            (x_min, y_min), bndbox[Keys.WIDTH], bndbox[Keys.HEIGHT],
            edgecolor=color, facecolor='none', linewidth=2, label=label)
        ax.add_patch(rect)
        if is_new:
            rects.append(rect)

    plt.legend(handles=rects)
    plt.show()

def test_dataloader():
    # random_crop = RandomCrop(488)
    transform = transforms.Compose([
        Rescale(450),
        # ToTensor()
    ])
    test_dataset = VOCDataset(
        root='./data/VOC',
        year='2007',
        image_set='train',
        transform=transform
    )

    for i in range(5):
        random_index = random.randint(0, len(test_dataset) - 1)
        sample = test_dataset[random_index]

        image, annotations = sample[Keys.IMAGE], sample[Keys.ANNOTATION]
        show_plot(image, annotations)


test_dataloader()