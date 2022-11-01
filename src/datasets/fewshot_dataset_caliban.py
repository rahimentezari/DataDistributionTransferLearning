import os
from PIL import Image

import torch
import numpy as np
import pandas as pd
from torchvision import transforms, datasets
from google.cloud import storage
import pickle

def download_pkl(bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    pickle_in = blob.download_as_string()
    return pickle.loads(pickle_in)
    # return pickle_in

def download_blob(bucket_name, source_blob_name, destination_file_name,
                  blob_path_prefix=""):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path_prefix + source_blob_name)
    blob.download_to_filename(destination_file_name)

def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None, shots=None):
        self.images = images
        self.labels = self._lbl2idx(labels)
        self.transform = transform
        self.num_classes = len(set(self.labels))

        if shots is not None:
            self._few_shot(shots)


    def _lbl2idx(self, labels):
        unique = sorted(list(set(labels)))
        self.lbl2id = {lbl: i for i, lbl in enumerate(unique)}
        self.id2lbl = {v: k for k, v in self.lbl2id.items()}

        return [self.lbl2id[lbl] for lbl in labels]

    def _few_shot(self, shots):
        images = np.array(self.images)
        labels = np.array(self.labels)
        # print("Pickling dataset!")
        # with open('images.pkl', 'wb') as handle:
        #     pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('labels.pkl', 'wb') as handle:
        #     pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open('images.pkl', 'rb') as handle:
        #     images = pickle.load(handle)
        # with open('labels.pkl', 'rb') as handle:
        #     labels = pickle.load(handle)

        # print(images[0])
        # print(labels[0])

        selected_images = []
        selected_labels = []
        for i in range(self.num_classes):
            mask = labels == i
            masked_images = images[mask]
            masked_labels = labels[mask]

            # np.random.seed(0)
            # print("shots", i, shots, len(images), len(masked_images))
            if shots <= len(masked_images):
                rand_idxs = np.random.choice(range(len(masked_images)), shots, replace=False)
            else:
                print("shots", i, shots, len(images), len(masked_images))
                rand_idxs = np.random.choice(range(len(masked_images)), shots, replace=True)  #### e.g. domainnet in unbalanced
            # rand_idxs = np.random.choice(range(len(masked_images)), len(masked_images), replace=False)
            selected_images.extend(masked_images[rand_idxs])
            selected_labels.extend(masked_labels[rand_idxs])

        self.images = selected_images
        self.labels = selected_labels

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        # print(image)
        # print(label)
        image = Image.open(image).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        return len(self.images)


class BucketDataset(Dataset):
    # def __init__(self, is_train, transform=None, shots=None):
    def __init__(self, args, is_train, transform=None, shots=None):
        print(f"===== Downloading the DS {args.train_dataset} dataset from Google Bucket. It may take a while! Please wait :) ====")
        bucket_name = 'clip_uw_cp'
        if is_train:
            if args.train_dataset == 'CIFAR100':
                source_file_name = 'train.pkl'
                destination_blob_name = f'datasets/few_shot/cifar-100-python/{source_file_name}'
                train = download_pkl(bucket_name, destination_blob_name)
                self.images = np.array(train[b'data']).reshape((-1, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
                self.images_idx = np.arange(len(self.images))
                self.labels = [int(x) for x in np.array(train[b'fine_labels'])]
            elif args.train_dataset == 'CIFAR10':
                destination_blob_name = 'train_batch'
                source_file_name = f'datasets/few_shot/CIFAR10/{destination_blob_name}'
                download_blob(bucket_name, source_file_name, destination_blob_name)
                with open('train_batch', 'rb') as fo:
                    train = pickle.load(fo, encoding='bytes')
                self.images = np.array(train[b'data']).reshape((-1, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
                self.images_idx = np.arange(len(self.images))
                self.labels = [int(x) for x in np.array(train[b'labels'])]
            else:

                source_file_name = f'images_train.pkl'
                destination_blob_name = f'datasets/few_shot/{args.train_dataset}/{source_file_name}'
                print(destination_blob_name)
                train = download_pkl(bucket_name, destination_blob_name)
                # self.images = np.array(train).reshape((-1, 3, 32, 32))
                # self.images = self.images.transpose((0, 2, 3, 1))
                self.images = np.array(train)
                self.images_idx = np.arange(len(self.images))

                source_file_name = f'labels_train.pkl'
                destination_blob_name = f'datasets/few_shot/{args.train_dataset}/{source_file_name}'
                labels = download_pkl(bucket_name, destination_blob_name)
                self.labels = self._lbl2idx(labels)
        else:
            if args.train_dataset == 'CIFAR100':
                source_file_name = 'test.pkl'
                destination_blob_name = f'datasets/few_shot/cifar-100-python/{source_file_name}'
                test = download_pkl(bucket_name, destination_blob_name)
                self.images = np.array(test[b'data']).reshape((-1, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
                self.images_idx = np.arange(len(self.images))
                self.labels = [int(x) for x in np.array(test[b'fine_labels'])]
            elif args.train_dataset == 'CIFAR10':
                destination_blob_name = 'test_batch'
                source_file_name = f'datasets/few_shot/CIFAR10/{destination_blob_name}'
                download_blob(bucket_name, source_file_name, destination_blob_name)
                with open('test_batch', 'rb') as fo:
                    test = pickle.load(fo, encoding='bytes')
                self.images = np.array(test[b'data']).reshape((-1, 3, 32, 32))
                self.images = self.images.transpose((0, 2, 3, 1))
                self.images_idx = np.arange(len(self.images))
                self.labels = [int(x) for x in np.array(test[b'labels'])]
            else:
                source_file_name = f'images_test.pkl'
                destination_blob_name = f'datasets/few_shot/{args.train_dataset}/{source_file_name}'
                test = download_pkl(bucket_name, destination_blob_name)
                # self.images = np.array(test).reshape((-1, 3, 32, 32))
                # self.images = self.images.transpose((0, 2, 3, 1))
                self.images = np.array(test)
                self.images_idx = np.arange(len(self.images))

                source_file_name = f'labels_test.pkl'
                destination_blob_name = f'datasets/few_shot/{args.train_dataset}/{source_file_name}'
                labels = download_pkl(bucket_name, destination_blob_name)
                self.labels = self._lbl2idx(labels)

        self.transform = transform
        self.num_classes = len(set(self.labels))
        if shots is not None:
            self._few_shot(shots)

    def _few_shot(self, shots):
        images = np.array(self.images_idx)
        labels = np.array(self.labels)

        selected_images = []
        selected_labels = []
        for i in range(self.num_classes):
            mask = labels == i
            masked_images = images[mask]
            masked_labels = labels[mask]

            # np.random.seed(0)
            rand_idxs = np.random.choice(range(len(masked_images)), shots, replace=False)
            selected_images.extend(masked_images[rand_idxs])
            selected_labels.extend(masked_labels[rand_idxs])

        self.images = self.images[selected_images]
        self.labels = selected_labels

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        # print("get_item, few_line200", image.shape)
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)
            # print("get_item, few_line273", image.shape)
            # print()

        return image, label

def get_transforms(config):

    if config.transfer_mode == 'ZeroShot':
        train_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        valid_transforms = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    elif config.transfer_mode == 'Linear':
        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        valid_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    elif config.transfer_mode == 'Finetune':
        pass

    return train_transforms, valid_transforms

