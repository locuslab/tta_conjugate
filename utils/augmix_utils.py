# import augmentations
from utils import augmentations as augmentations 
import numpy as np
import torch
from pathlib import Path
import os 
from robustbench.zenodo_download import DownloadError, zenodo_download
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
from PIL import Image


CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

ZENODO_CORRUPTIONS_LINKS: Dict[BenchmarkDataset, Tuple[str, Set[str]]] = {
    BenchmarkDataset.cifar_10: ("2535967", {"CIFAR-10-C.tar"}),
    BenchmarkDataset.cifar_100: ("3555552", {"CIFAR-100-C.tar"})
}

CORRUPTIONS_DIR_NAMES: Dict[BenchmarkDataset, str] = {
    BenchmarkDataset.cifar_10: "CIFAR-10-C",
    BenchmarkDataset.cifar_100: "CIFAR-100-C",
    BenchmarkDataset.imagenet: "ImageNet-C"
}


def load_corruptions_cifar(
        dataset,
        n_examples,
        severity,
        data_dir,
        corruptions,
        shuffle: bool = False):
    assert 1 <= severity <= 5
    n_total_cifar = 10000

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_dir = Path(data_dir)
    data_root_dir = data_dir / CORRUPTIONS_DIR_NAMES[dataset]

    if not data_root_dir.exists():
        zenodo_download(*ZENODO_CORRUPTIONS_LINKS[dataset], save_dir=data_dir)

    # Download labels
    labels_path = data_root_dir / 'labels.npy'
    if not os.path.isfile(labels_path):
        raise DownloadError("Labels are missing, try to re-download them.")
    labels = np.load(labels_path)

    x_test_list, y_test_list = [], []
    n_pert = len(corruptions)
    for corruption in corruptions:
        corruption_file_path = data_root_dir / (corruption + '.npy')
        if not corruption_file_path.is_file():
            raise DownloadError(
                f"{corruption} file is missing, try to re-download it.")

        images_all = np.load(corruption_file_path)
        images = images_all[(severity - 1) * n_total_cifar:severity *
                            n_total_cifar]
        n_img = int(np.ceil(n_examples / n_pert))
        x_test_list.append(images[:n_img])
        # Duplicate the same labels potentially multiple times
        y_test_list.append(labels[:n_img])

    x_test, y_test = np.concatenate(x_test_list), np.concatenate(y_test_list)
    if shuffle:
        rand_idx = np.random.permutation(np.arange(len(x_test)))
        x_test, y_test = x_test[rand_idx], y_test[rand_idx]

    # # Make it in the PyTorch format
    # x_test = np.transpose(x_test, (0, 3, 1, 2))
    # # Make it compatible with our models
    # x_test = x_test.astype(np.float32) / 255
    # # Make sure that we get exactly n_examples but not a few samples more
    # x_test = torch.tensor(x_test)[:n_examples]
    # y_test = torch.tensor(y_test)[:n_examples]

    return x_test, y_test

def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  mixture_width = 3
  mixture_depth = -1
  aug_severity = 3

  aug_list = augmentations.augmentations_all
#   if args.all_ops:
    # aug_list = augmentations.augmentations_all

  ws = np.float32(np.random.dirichlet([1] * mixture_width))
#   m = np.float32(np.random.beta(1, 1))
  m = 0.4

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, 3)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed

class AugMixDatasetImageNet(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    x, y = self.dataset[i]
    img = x
    # x, y = self.x_test[i], self.y_test[i]
    # print(type(x))
    # print(type(y))
    # print("ssssssssssssssssssss")
    # img = Image.fromarray(x)

    if self.no_jsd:
      return aug(img, self.preprocess), y
    else:
      im_tuple = (self.preprocess(img), aug(img, self.preprocess),
                  aug(img, self.preprocess), aug(img, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)

class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, x_test, y_test, preprocess, no_jsd=False):
    # self.dataset = dataset
    self.x_test = x_test 
    self.y_test = y_test
    self.preprocess = preprocess
    self.no_jsd = no_jsd

  def __getitem__(self, i):
    # x, y = self.dataset[i]
    x, y = self.x_test[i], self.y_test[i]
    img = Image.fromarray(x)

    if self.no_jsd:
      return aug(img, self.preprocess), y
    else:
      im_tuple = (self.preprocess(img), aug(img, self.preprocess),
                  aug(img, self.preprocess), aug(img, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return self.x_test.shape[0]