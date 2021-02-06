import torch
from torchvision import transforms as transforms

from Keys import Keys


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    def __call__(self, sample):
        image, annotations = sample[Keys.IMAGE], sample[Keys.ANNOTATION]
        print(annotations[0])
        h, w = image.size[:2]
        if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size, self.output_size * w / h
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_h, new_w))

        new_annotations = []
        # there can be multiple annotations
        for annotation in annotations:
            bndbox = annotation["bndbox"]

            new_bndbox = {Keys.CENTER_X: bndbox[Keys.CENTER_X] * (new_w / w),
                              Keys.CENTER_Y: bndbox[Keys.CENTER_Y] * (new_h / h),
                              Keys.WIDTH: bndbox[Keys.WIDTH] * (new_w / w),
                              Keys.HEIGHT: bndbox[Keys.HEIGHT] * (new_h / h)}
            annotation["bndbox"] = new_bndbox
            new_annotations.append(annotation)

        return {Keys.IMAGE: img, Keys.ANNOTATION: new_annotations}


class CenterCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size
    def __call__(self, sample):
        image, annotation = sample[Keys.IMAGE], sample[Keys.ANNOTATION]
        h, w = image.size[:2]
        new_h, new_w = self.output_size, self.output_size
        center_y = (h - new_h) // 2
        center_x = (w - new_w) // 2
        image = image.crop((center_y, center_x, center_y + new_h, center_x + new_w))
        annotation[Keys.CENTER_X] = annotation[Keys.LEFT] - center_y
        annotation[Keys.CENTER_Y] = annotation[Keys.TOP] - center_x
        if annotation[Keys.CENTER_X] < 0:
            annotation[Keys.WIDTH] += annotation[Keys.LEFT]
            annotation[Keys.CENTER_X] = 0
        if annotation[Keys.CENTER_Y] < 0:
            annotation[Keys.HEIGHT] += annotation[Keys.TOP]
            annotation[Keys.CENTER_Y] = 0
        annotation[Keys.WIDTH] = max(0, min(INPUT_SIZE, annotation[Keys.WIDTH]))
        annotation[Keys.HEIGHT] = max(0, min(INPUT_SIZE, annotation[Keys.HEIGHT]))
        return {Keys.IMAGE: image, Keys.ANNOTATION: annotation}


class ColorJitter(object):
    def __call__(self, sample):
        image, annotation = sample[Keys.IMAGE], sample[Keys.ANNOTATION]
        image = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.4,
            hue=0.02
        )(image)
        return {Keys.IMAGE: image, Keys.ANNOTATION: annotation}


class ToTensor(object):
    def __call__(self, sample):
        image, annotation = sample[Keys.IMAGE], sample[Keys.ANNOTATION]
        image = transforms.ToTensor()(image)
        annotation = torch.Tensor([
            annotation[Keys.CLASS_ID],
            (annotation[Keys.CENTER_X] + annotation[Keys.WIDTH] * 0.5),
            (annotation[Keys.CENTER_Y] + annotation[Keys.HEIGHT] * 0.5),
            (annotation[Keys.WIDTH]),
            (annotation[Keys.HEIGHT])
        ])
        return {Keys.IMAGE: image, Keys.ANNOTATION: annotation}