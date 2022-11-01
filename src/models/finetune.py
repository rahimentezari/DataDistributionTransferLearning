import os
import copy
import time
import tqdm
import numpy as np
import torch
import sys
sys.path.insert(1, 'src')
# import clip.clip as clip
import pandas as pd
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from models.eval import evaluate
from models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from models.utils import cosine_lr, torch_load, LabelSmoothing
import  pickle
import datasets as datasets
from collections import defaultdict, deque
import torch.distributed as dist
from timm.utils import accuracy, ModelEma
from google.cloud import storage
from datasets.datasets import build_transform
from datasets.fewshot_dataset_caliban import Dataset, BucketDataset
import datetime


def download_blob(bucket_name, source_blob_name, destination_file_name,
                  blob_path_prefix=""):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path_prefix + source_blob_name)
    blob.download_to_filename(destination_file_name)
def download_pkl(bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    pickle_in = blob.download_as_string()
    return pickle.loads(pickle_in)


def upload_pkl(bucket_name, pickle_out, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(pickle_out)

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

# def get_dataset(is_train, args):
#     transform = build_transform(is_train, args)
#     shots = args.shots if is_train else None
#     if args.train_dataset == 'imagenet':
#         # df = pd.read_csv("/mnt/external/few_shot/path/imagenet_0.5.csv")
#         df = pd.read_csv("imagenet_0.5.csv")
#         df = df[df['split'] == 'test'] if not is_train else df[df['split'] != 'test']
#
#         dataset = Dataset(
#             images=df['path'].values,
#             labels=df['label'].values,
#             transform=transform,
#             shots=args.shots
#         )
#     else:
#         dataset = BucketDataset(args, is_train, transform=transform, shots=shots)
#     print(f"is_train: {is_train}, num_samples: {len(dataset)}, num_classes: {dataset.num_classes}")
#     return dataset, dataset.num_classes


def get_dataset(is_train, args):

    transform = build_transform(is_train, args)
    shots = args.shots if is_train else None

    if args.train_dataset.lower() == 'imagenet':
        if not os.path.exists('small_imagenet.tar.gz'):
            bucket_name = 'clip_uw_cp'
            destination_blob_name = 'small_imagenet.tar.gz'
            # destination_blob_name = '2KSubset_train_val.tar.gz'
            source_file_name = f'datasets/few_shot/ImageNet/{destination_blob_name}'
            print("Downloading Small ImageNet of 20GB. That takes ~2 minutes!")
            download_blob(bucket_name, source_file_name, destination_blob_name)

            # destination_blob_name = 'small_imagenet_samples_relativepath.npy'
            destination_blob_name = 'imagenet_0.5_relative.csv'
            # destination_blob_name = '2KSubset_imagenet_0.5_relative.csv'
            source_file_name = f'datasets/few_shot/ImageNet/{destination_blob_name}'
            download_blob(bucket_name, source_file_name, destination_blob_name)
            # samples = np.load('small_imagenet_samples_relativepath.npy')
            print("unzipping Small ImageNet. That takes ~7 minutes!")
            os.system("tar -xzf small_imagenet.tar.gz")
            # os.system("tar -xzf 2KSubset_train_val.tar.gz")
            os.system("ls")

        df = pd.read_csv("imagenet_0.5_relative.csv")
        # df = pd.read_csv("2KSubset_imagenet_0.5_relative.csv")
        df = df[df['split'] == 'test'] if not is_train else df[df['split'] != 'test']

        dataset = Dataset(
            images=df['path'].values,
            labels=df['label'].values,
            transform=transform,
            shots=args.shots
        )
    elif args.train_dataset.lower() == 'real':
        if not os.path.exists('real.tar.gz'):
            bucket_name = 'clip_uw_cp'
            destination_blob_name = 'real.tar.gz'
            # destination_blob_name = '2KSubset_train_val.tar.gz'
            source_file_name = f'datasets/few_shot/REAL/{destination_blob_name}'
            print("Downloading REAL of 5GB!")
            download_blob(bucket_name, source_file_name, destination_blob_name)

            destination_blob_name = 'REAL.csv'
            source_file_name = f'datasets/few_shot/REAL/{destination_blob_name}'
            download_blob(bucket_name, source_file_name, destination_blob_name)
            # samples = np.load('small_imagenet_samples_relativepath.npy')
            print("unzipping takes ~3 minutes!")
            os.system("tar -xzf real.tar.gz")
            # os.system("tar -xzf 2KSubset_train_val.tar.gz")
            os.system("ls")

        df = pd.read_csv("REAL.csv")
        # df = pd.read_csv("2KSubset_imagenet_0.5_relative.csv")
        df = df[df['split'] == 'test'] if not is_train else df[df['split'] != 'test']

        dataset = Dataset(
            images=df['path'].values,
            labels=df['label'].values,
            transform=transform,
            shots=args.shots
        )
    elif args.train_dataset.lower() == 'quickdraw':
        if not os.path.exists('quickdraw.tar.gz'):
            bucket_name = 'clip_uw_cp'
            destination_blob_name = 'quickdraw.tar.gz'
            # destination_blob_name = '2KSubset_train_val.tar.gz'
            source_file_name = f'datasets/few_shot/QUICKDRAW/{destination_blob_name}'
            print("Downloading QUICKDRAW of 0.5GB!")
            download_blob(bucket_name, source_file_name, destination_blob_name)

            destination_blob_name = 'QUICKDRAW.csv'
            source_file_name = f'datasets/few_shot/QUICKDRAW/{destination_blob_name}'
            download_blob(bucket_name, source_file_name, destination_blob_name)
            # samples = np.load('small_imagenet_samples_relativepath.npy')
            print("unzipping takes ~1 minutes!")
            os.system("tar -xzf quickdraw.tar.gz")
            # os.system("tar -xzf 2KSubset_train_val.tar.gz")
            os.system("ls")

        df = pd.read_csv("QUICKDRAW.csv")
        # df = pd.read_csv("2KSubset_imagenet_0.5_relative.csv")
        df = df[df['split'] == 'test'] if not is_train else df[df['split'] != 'test']

        dataset = Dataset(
            images=df['path'].values,
            labels=df['label'].values,
            transform=transform,
            shots=args.shots
        )
    elif args.train_dataset.lower() == 'clipart':
        if not os.path.exists('clipart.tar.gz'):
            bucket_name = 'clip_uw_cp'
            destination_blob_name = 'clipart.tar.gz'
            # destination_blob_name = '2KSubset_train_val.tar.gz'
            source_file_name = f'datasets/few_shot/CLIPART/{destination_blob_name}'
            print("Downloading CLIPART of 0.5GB!")
            download_blob(bucket_name, source_file_name, destination_blob_name)

            destination_blob_name = 'CLIPART.csv'
            source_file_name = f'datasets/few_shot/CLIPART/{destination_blob_name}'
            download_blob(bucket_name, source_file_name, destination_blob_name)
            # samples = np.load('small_imagenet_samples_relativepath.npy')
            print("unzipping takes ~1 minutes!")
            os.system("tar -xzf clipart.tar.gz")
            # os.system("tar -xzf 2KSubset_train_val.tar.gz")
            os.system("ls")

        df = pd.read_csv("CLIPART.csv")
        # df = pd.read_csv("2KSubset_imagenet_0.5_relative.csv")
        df = df[df['split'] == 'test'] if not is_train else df[df['split'] != 'test']

        dataset = Dataset(
            images=df['path'].values,
            labels=df['label'].values,
            transform=transform,
            shots=args.shots
        )

    elif args.train_dataset.lower() == 'cassavaleafdisease':
        if not os.path.exists('train_images.tar.gz'):
            bucket_name = 'clip_uw_cp'
            destination_blob_name = 'train_images.tar.gz'
            source_file_name = f'datasets/few_shot/CassavaLeafDisease/{destination_blob_name}'
            print("Downloading CassavaLeafDisease of 2.4GB!")
            download_blob(bucket_name, source_file_name, destination_blob_name)

            destination_blob_name = 'cassavaleafdisease.csv'
            source_file_name = f'datasets/few_shot/CassavaLeafDisease/{destination_blob_name}'
            download_blob(bucket_name, source_file_name, destination_blob_name)
            # samples = np.load('small_imagenet_samples_relativepath.npy')
            print("unzipping takes ~1 minutes!")
            os.system("tar -xzf train_images.tar.gz")
            os.system("ls")

        df = pd.read_csv("cassavaleafdisease.csv")
        # df = pd.read_csv("2KSubset_imagenet_0.5_relative.csv")
        df = df[df['split'] == 'test'] if not is_train else df[df['split'] != 'test']

        dataset = Dataset(
            images=df['path'].values,
            labels=df['label'].values,
            transform=transform,
            shots=args.shots
        )
    elif args.train_dataset.lower() == 'eurosat':
        if not os.path.exists('images.tar.gz'):
            bucket_name = 'clip_uw_cp'
            destination_blob_name = 'images.tar.gz'
            source_file_name = f'datasets/few_shot/EuroSAT/{destination_blob_name}'
            print("Downloading eurosat of 0.1GB!")
            download_blob(bucket_name, source_file_name, destination_blob_name)

            destination_blob_name = 'eurosat.csv'
            source_file_name = f'datasets/few_shot/EuroSAT/{destination_blob_name}'
            download_blob(bucket_name, source_file_name, destination_blob_name)
            # samples = np.load('small_imagenet_samples_relativepath.npy')
            print("unzipping takes ~1 minutes!")
            os.system("tar -xzf images.tar.gz")
            os.system("ls")

        df = pd.read_csv("eurosat.csv")
        # df = pd.read_csv("2KSubset_imagenet_0.5_relative.csv")
        df = df[df['split'] == 'test'] if not is_train else df[df['split'] != 'test']

        dataset = Dataset(
            images=df['path'].values,
            labels=df['label'].values,
            transform=transform,
            shots=args.shots
        )
    elif args.train_dataset.lower() == 'cameratraps':
        if not os.path.exists('eccv_18_all_images_sm.tar.gz'):
            bucket_name = 'clip_uw_cp'
            destination_blob_name = 'eccv_18_all_images_sm.tar.gz'
            source_file_name = f'datasets/few_shot/CameraTraps/{destination_blob_name}'
            print("Downloading cameratraps of 6GB!")
            download_blob(bucket_name, source_file_name, destination_blob_name)

            destination_blob_name = 'cameratraps.csv'
            source_file_name = f'datasets/few_shot/CameraTraps/{destination_blob_name}'
            download_blob(bucket_name, source_file_name, destination_blob_name)
            # samples = np.load('small_imagenet_samples_relativepath.npy')
            print("unzipping takes ~1 minutes!")
            os.system("tar -xzf eccv_18_all_images_sm.tar.gz")
            os.system("ls")

        df = pd.read_csv("cameratraps.csv")
        # df = pd.read_csv("2KSubset_imagenet_0.5_relative.csv")
        df = df[df['split'] == 'test'] if not is_train else df[df['split'] != 'test']

        dataset = Dataset(
            images=df['path'].values,
            labels=df['label'].values,
            transform=transform,
            shots=args.shots
        )



    else:
        dataset = BucketDataset(args, is_train, transform=transform, shots=shots)
    print(f"is_train: {is_train}, num_samples: {len(dataset)}, num_classes: {dataset.num_classes}")
    return dataset, dataset.num_classes


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def evaluate_(args, data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    model.to(device)
    if args.freeze_encoder:
        input_key = 'features'
    else:
        input_key = 'images'
    # if 'SimCLR' in args.upstream_loss or args.finetune_strategy == 'lp':
    if args.finetune_strategy == 'lp':
        # for i, batch in enumerate(data_loader):
        for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
            batch = maybe_dictionarize(batch)
            images = batch[input_key].cuda()
            target = batch['labels'].cuda()

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
            # print(images.shape, target.shape, output.shape)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            images = samples.to(device, non_blocking=True)
            target = targets.to(device, non_blocking=True)


            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
            # print(images.shape, target.shape, output.shape)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def finetune(args):
    assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    assert args.train_dataset is not None, "Please provide a training dataset."

    if 'SimCLR' in args.upstream_loss or args.finetune_strategy == 'lp_ft':
        image_classifier = args.classifier
    else:
        image_classifier = ImageClassifier.load(args.load)

    if args.freeze_encoder:
        print('Fine-tuning a linear classifier')
        model = image_classifier.classification_head
        input_key = 'features'
        preprocess_fn = image_classifier.val_preprocess
        image_enc = image_classifier.image_encoder
        print_every = 1000
    else:
        print('Fine-tuning end-to-end')
        model = image_classifier
        input_key = 'images'
        preprocess_fn = image_classifier.train_preprocess
        image_enc = None
        image_classifier.process_images = True
        print_every = 100

    # if args.train_dataset.lower() == 'imagenet':
    #     dataset_class = getattr(datasets, args.train_dataset)
    #     dataset = dataset_class(
    #         preprocess_fn,
    #         location=args.data_location,
    #         batch_size=args.batch_size
    #     )
    #     data_loader_train = dataset.train_loader
    #     data_loader_val = dataset.test_loader
    #     nb_classes = 1000
    #
    # else:
    dataset_train, nb_classes = get_dataset(is_train=True, args=args)
    dataset_val, nb_classes = get_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.RandomSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )


    num_batches = len(data_loader_train)
    print("nb_classes, num_batches: ", nb_classes, num_batches)


    model = model.cuda()
    devices = list(range(torch.cuda.device_count()))
    print('Using devices', devices)
    model = torch.nn.DataParallel(model, device_ids=devices)
    model.train()

    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)

    if 'SimCLR' in args.upstream_loss:
        pretrain_epochs, pretrain_lr, pretrain_wd, pretrain_bs = 16, 0.001, 0.1, 1024
    elif 'captions' in args.upstream_dataset or 'YFCC_0.5m' in args.upstream_dataset:
        pretrain_epochs, pretrain_lr, pretrain_wd, pretrain_bs = 32, 0.001, 0.1, 1024
    elif 'LAION_400m' in args.upstream_dataset:
        pretrain_epochs, pretrain_lr, pretrain_wd, pretrain_bs = 32, 0.001, 0.1, 1024
    elif 'LAION_2B' in args.upstream_dataset:
        pretrain_epochs, pretrain_lr, pretrain_wd, pretrain_bs = 16, 0.001, 0.1, 1024
    elif 'ImageNet1K_Captions' in args.upstream_dataset:
        pretrain_epochs, pretrain_lr, pretrain_wd, pretrain_bs = 90, 0.001, 0.1, 1024
    else:
        ################### zero shot eval
        ##### first read the pretainig hyperparam from Bucket
        bucket_name = 'clip_uw_cp'
        destination_blob_name = 'params_Thao.csv'
        source_file_name = f'codes/{destination_blob_name}'
        params_Thao = download_blob(bucket_name, source_file_name, destination_blob_name)
        pretrain_parampandas = pd.read_csv('params_Thao.csv', sep='\t')
        # print(pretrain_parampandas)
        param_index = pretrain_parampandas.index[pretrain_parampandas['model.1'] == f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'].tolist()
        pretrain_epochs = pretrain_parampandas.iloc[param_index]['epochs'].tolist()[0]
        pretrain_lr = pretrain_parampandas.iloc[param_index]['lr'].tolist()[0]
        pretrain_bs = 8 * int(pretrain_parampandas.iloc[param_index]['batch_size'].tolist()[0])
        pretrain_wd = pretrain_parampandas.iloc[param_index]['wd'].tolist()[0]
        print("pretrain_lr, pretrain_wd, pretrain_bs", pretrain_lr, pretrain_wd, pretrain_bs)

        ############################ 16 vs 40 epochs
        # pretrain_epochs = 40

    if 'SimCLR' in args.upstream_loss or args.finetune_strategy == 'lp':
        zero_shot_stats = {}
        zero_shot_stats['acc1'] = np.nan
        zero_shot_stats['acc5'] = np.nan
        zero_shot_stats['loss'] = np.nan
        ##################################
    elif args.finetune_strategy != 'lp':
        # if args.train_dataset.lower() == 'imagenet':
        #     evaluate(image_classifier, args)
        # else:
        zero_shot_stats = evaluate_(args, data_loader_val, model, device='cuda')


        print("zero_shot_stats", zero_shot_stats)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {zero_shot_stats['acc1']:.1f}%")
        bucket_name = 'clip_uw_cp'
        source_file_name = f'zeroshot.pkl'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        # test
        # destination_blob_name = f'pretraining/{save_dir}/DS/{args.train_dataset.upper()}/US_/{args.shots}/Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}/{source_file_name}'

        destination_blob_name = f'pretraining/{save_dir}/DS/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        pickle_out = pickle.dumps(zero_shot_stats)
        upload_pkl(bucket_name, pickle_out, destination_blob_name)

    best_acc = 0
    eval_results = {'us_epochs': 0, 'us_lr': 0, 'us_bs': 0, 'us_wd': 0, 'ds_epochs': 0, 'ds_lr': 0, 'ds_bs': 0, 'ds_wd': 0, 'loss': [], 'acc1': [], 'acc5': [], 'best_acc': 0, 'zeroshot_stats': []}
    eval_results['zeroshot_stats'] = zero_shot_stats
    eval_results['us_epochs'] = pretrain_epochs
    eval_results['us_lr'] = pretrain_lr
    eval_results['us_bs'] = pretrain_bs
    eval_results['us_wd'] = pretrain_wd

    eval_results['ds_epochs'] = args.epochs
    eval_results['ds_lr'] = args.lr
    eval_results['ds_bs'] = args.batch_size
    eval_results['ds_wd'] = args.wd


    eval_results['zeroshot_stats'] = zero_shot_stats

    ###### upload init to Bucket
    os.makedirs(args.save, exist_ok=True)
    model_path = f'checkpoint_init.pt'
    print('Saving model to', model_path)
    image_classifier.save(model_path)
    optim_path = f'optim_init.pt'
    torch.save(optimizer.state_dict(), optim_path)

    bucket_name = 'clip_uw_cp'
    source_file_name = f'checkpoint_init.pt'
    if args.finetune_strategy == 'lp':
        ds_path = 'DS_LinearProbe'
    elif args.finetune_strategy == 'lp_ft':
        ds_path = 'DS_lp_ft'
    elif args.finetune_strategy == 'ft':
        ds_path = 'DS'

    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
    destination_blob_name = f'pretraining/{save_dir}/{ds_path}/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
    upload_blob(bucket_name, source_file_name, destination_blob_name)

    source_file_name = f'optim_init.pt'
    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
    destination_blob_name = f'pretraining/{save_dir}/{ds_path}/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
    upload_blob(bucket_name, source_file_name, destination_blob_name)

    if 'SimCLR' in args.upstream_loss and args.finetune_strategy == 'lp':
        # data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=image_enc)
        data_loader = get_dataloader(data_loader_train, data_loader_val, is_train=True, args=args, image_encoder=image_enc)
        # data_loader = get_dataloader(is_train=True, args=args, image_encoder=image_enc)

        for epoch in range(args.epochs):
            model.train()
            model = model.cuda()

            for i, batch in enumerate(data_loader):
                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch[input_key].cuda()
                labels = batch['labels'].cuda()
                data_time = time.time() - start_time

                logits = model(inputs)

                loss = loss_fn(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if i % print_every == 0:
                    percent_complete = 100 * i / len(data_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(data_loader_train)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )

            if args.freeze_encoder:
                image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
            else:
                image_classifier = model.module

            # # Saving model
            # if args.save is not None and epoch % args.save_ckpt_freq == 0:
            #     os.makedirs(args.save, exist_ok=True)
            #     model_path = f'checkpoint_{epoch+1}.pt'
            #     print('Saving model to', model_path)
            #     image_classifier.save(model_path)
            #     optim_path = f'optim_{epoch+1}.pt'
            #     torch.save(optimizer.state_dict(), optim_path)

            # Evaluate
            args.current_epoch = epoch
            if epoch % args.save_ckpt_freq == 0:
                # eval_result = evaluate(image_classifier, args)
                data_loader = get_dataloader(data_loader_train, data_loader_val, is_train=False, args=args, image_encoder=image_enc)
                eval_result = evaluate_(args, data_loader, model, device='cuda')
                print(eval_result)
                eval_results['loss'].append(eval_result['loss'])
                eval_results['acc1'].append(eval_result['acc1'])
                eval_results['acc5'].append(eval_result['acc5'])
                print(eval_results)
                if eval_result['acc1'] > best_acc:
                    best_acc = eval_result['acc1']
                    eval_results['best_acc'] = best_acc
                    print(eval_results)

                    ###### upload to Bucket
                    os.makedirs(args.save, exist_ok=True)
                    model_path = f'checkpoint_best.pt'
                    print('Saving model to', model_path)
                    image_classifier.save(model_path)
                    optim_path = f'optim_best.pt'
                    torch.save(optimizer.state_dict(), optim_path)

                    bucket_name = 'clip_uw_cp'
                    source_file_name = f'checkpoint_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                    source_file_name = f'optim_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                bucket_name = 'clip_uw_cp'
                source_file_name = f'eval_results.pkl'
                save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                pickle_out = pickle.dumps(eval_results)
                upload_pkl(bucket_name, pickle_out, destination_blob_name)



            bucket_name = 'clip_uw_cp'
            source_file_name = f'eval_results.pkl'
            save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
            destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
            pickle_out = pickle.dumps(eval_results)
            upload_pkl(bucket_name, pickle_out, destination_blob_name)

        ###### last epoch upload to Bucket
        os.makedirs(args.save, exist_ok=True)
        model_path = f'checkpoint_last.pt'
        print('Saving model to', model_path)
        image_classifier.save(model_path)
        optim_path = f'optim_last.pt'
        torch.save(optimizer.state_dict(), optim_path)

        bucket_name = 'clip_uw_cp'
        source_file_name = f'checkpoint_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)

        source_file_name = f'optim_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)
    elif 'SimCLR' in args.upstream_loss and args.finetune_strategy == 'lp_ft':
        for epoch in range(args.epochs):

            model.train()
            model = model.cuda()
            # data_loader = get_dataloader(
            #     dataset, is_train=True, args=args, image_encoder=image_enc)
            # for i, batch in enumerate(data_loader):
            for i, batch in enumerate(data_loader_train):
                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch[input_key].cuda()
                labels = batch['labels'].cuda()
                data_time = time.time() - start_time
                # print(model)

                # from torchsummary import summary
                # summary(model, (3, 224, 224))

                logits = model(inputs)
                loss = loss_fn(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if i % print_every == 0:
                    percent_complete = 100 * i / len(data_loader_train)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(data_loader_train)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )

            if args.freeze_encoder:
                image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
            else:
                image_classifier = model.module

            # # Saving model
            # if args.save is not None and epoch % args.save_ckpt_freq == 0:
            #     os.makedirs(args.save, exist_ok=True)
            #     model_path = f'checkpoint_{epoch+1}.pt'
            #     print('Saving model to', model_path)
            #     image_classifier.save(model_path)
            #     optim_path = f'optim_{epoch+1}.pt'
            #     torch.save(optimizer.state_dict(), optim_path)




            # Evaluate
            args.current_epoch = epoch
            if epoch % args.save_ckpt_freq == 0:
                # eval_results = evaluate(image_classifier, args, device='cuda')
                eval_result = evaluate_(args, data_loader_val, model, device='cuda')
                eval_results['loss'].append(eval_result['loss'])
                eval_results['acc1'].append(eval_result['acc1'])
                eval_results['acc5'].append(eval_result['acc5'])
                print(eval_results)
                if eval_result['acc1'] > best_acc:
                    best_acc = eval_result['acc1']
                    eval_results['best_acc'] = best_acc
                    print(eval_results)

                    ###### upload to Bucket
                    os.makedirs(args.save, exist_ok=True)
                    model_path = f'checkpoint_best.pt'
                    print('Saving model to', model_path)
                    image_classifier.save(model_path)
                    optim_path = f'optim_best.pt'
                    torch.save(optimizer.state_dict(), optim_path)


                    bucket_name = 'clip_uw_cp'
                    source_file_name = f'checkpoint_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                    source_file_name = f'optim_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                bucket_name = 'clip_uw_cp'
                source_file_name = f'eval_results.pkl'
                save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                pickle_out = pickle.dumps(eval_results)
                upload_pkl(bucket_name, pickle_out, destination_blob_name)

        ###### last epoch upload to Bucket
        os.makedirs(args.save, exist_ok=True)
        model_path = f'checkpoint_last.pt'
        print('Saving model to', model_path)
        image_classifier.save(model_path)
        optim_path = f'optim_last.pt'
        torch.save(optimizer.state_dict(), optim_path)

        bucket_name = 'clip_uw_cp'
        source_file_name = f'checkpoint_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)

        source_file_name = f'optim_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)
    elif 'CLIP' in args.upstream_loss and args.finetune_strategy == 'lp':
        # data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=image_enc)
        data_loader = get_dataloader(data_loader_train, data_loader_val, is_train=True, args=args, image_encoder=image_enc)
        # data_loader = get_dataloader(is_train=True, args=args, image_encoder=image_enc)

        for epoch in range(args.epochs):
            model.train()
            model = model.cuda()

            for i, batch in enumerate(data_loader):
                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch[input_key].cuda()
                labels = batch['labels'].cuda()
                data_time = time.time() - start_time

                logits = model(inputs)

                loss = loss_fn(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if i % print_every == 0:
                    percent_complete = 100 * i / len(data_loader)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(data_loader_train)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )

            if args.freeze_encoder:
                image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
            else:
                image_classifier = model.module

            # # Saving model
            # if args.save is not None and epoch % args.save_ckpt_freq == 0:
            #     os.makedirs(args.save, exist_ok=True)
            #     model_path = f'checkpoint_{epoch+1}.pt'
            #     print('Saving model to', model_path)
            #     image_classifier.save(model_path)
            #     optim_path = f'optim_{epoch+1}.pt'
            #     torch.save(optimizer.state_dict(), optim_path)

            # Evaluate
            args.current_epoch = epoch
            if epoch % args.save_ckpt_freq == 0:
                # eval_result = evaluate(image_classifier, args)
                data_loader = get_dataloader(data_loader_train, data_loader_val, is_train=False, args=args, image_encoder=image_enc)
                eval_result = evaluate_(args, data_loader, model, device='cuda')
                print(eval_result)
                eval_results['loss'].append(eval_result['loss'])
                eval_results['acc1'].append(eval_result['acc1'])
                eval_results['acc5'].append(eval_result['acc5'])
                print(eval_results)
                if eval_result['acc1'] > best_acc:
                    best_acc = eval_result['acc1']
                    eval_results['best_acc'] = best_acc
                    print(eval_results)

                    ###### upload to Bucket
                    os.makedirs(args.save, exist_ok=True)
                    model_path = f'checkpoint_best.pt'
                    print('Saving model to', model_path)
                    image_classifier.save(model_path)
                    optim_path = f'optim_best.pt'
                    torch.save(optimizer.state_dict(), optim_path)

                    bucket_name = 'clip_uw_cp'
                    source_file_name = f'checkpoint_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                    source_file_name = f'optim_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                bucket_name = 'clip_uw_cp'
                source_file_name = f'eval_results.pkl'
                save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                pickle_out = pickle.dumps(eval_results)
                upload_pkl(bucket_name, pickle_out, destination_blob_name)



            bucket_name = 'clip_uw_cp'
            source_file_name = f'eval_results.pkl'
            save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
            destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
            pickle_out = pickle.dumps(eval_results)
            upload_pkl(bucket_name, pickle_out, destination_blob_name)

        ###### last epoch upload to Bucket
        os.makedirs(args.save, exist_ok=True)
        model_path = f'checkpoint_last.pt'
        print('Saving model to', model_path)
        image_classifier.save(model_path)
        optim_path = f'optim_last.pt'
        torch.save(optimizer.state_dict(), optim_path)

        bucket_name = 'clip_uw_cp'
        source_file_name = f'checkpoint_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)

        source_file_name = f'optim_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)
    elif 'CLIP' in args.upstream_loss and args.finetune_strategy == 'lp_ft':
        for epoch in range(args.epochs):

            model.train()
            model = model.cuda()
            # data_loader = get_dataloader(
            #     dataset, is_train=True, args=args, image_encoder=image_enc)
            # for i, batch in enumerate(data_loader):
            for i, batch in enumerate(data_loader_train):
                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch[input_key].cuda()
                labels = batch['labels'].cuda()
                data_time = time.time() - start_time
                # print(model)

                # from torchsummary import summary
                # summary(model, (3, 224, 224))

                logits = model(inputs)
                loss = loss_fn(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if i % print_every == 0:
                    percent_complete = 100 * i / len(data_loader_train)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(data_loader_train)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )

            if args.freeze_encoder:
                image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
            else:
                image_classifier = model.module

            # # Saving model
            # if args.save is not None and epoch % args.save_ckpt_freq == 0:
            #     os.makedirs(args.save, exist_ok=True)
            #     model_path = f'checkpoint_{epoch+1}.pt'
            #     print('Saving model to', model_path)
            #     image_classifier.save(model_path)
            #     optim_path = f'optim_{epoch+1}.pt'
            #     torch.save(optimizer.state_dict(), optim_path)




            # Evaluate
            args.current_epoch = epoch
            if epoch % args.save_ckpt_freq == 0:
                # eval_results = evaluate(image_classifier, args, device='cuda')
                eval_result = evaluate_(args, data_loader_val, model, device='cuda')
                eval_results['loss'].append(eval_result['loss'])
                eval_results['acc1'].append(eval_result['acc1'])
                eval_results['acc5'].append(eval_result['acc5'])
                print(eval_results)
                if eval_result['acc1'] > best_acc:
                    best_acc = eval_result['acc1']
                    eval_results['best_acc'] = best_acc
                    print(eval_results)

                    ###### upload to Bucket
                    os.makedirs(args.save, exist_ok=True)
                    model_path = f'checkpoint_best.pt'
                    print('Saving model to', model_path)
                    image_classifier.save(model_path)
                    optim_path = f'optim_best.pt'
                    torch.save(optimizer.state_dict(), optim_path)


                    bucket_name = 'clip_uw_cp'
                    source_file_name = f'checkpoint_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                    source_file_name = f'optim_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                bucket_name = 'clip_uw_cp'
                source_file_name = f'eval_results.pkl'
                save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                pickle_out = pickle.dumps(eval_results)
                upload_pkl(bucket_name, pickle_out, destination_blob_name)

        ###### last epoch upload to Bucket
        os.makedirs(args.save, exist_ok=True)
        model_path = f'checkpoint_last.pt'
        print('Saving model to', model_path)
        image_classifier.save(model_path)
        optim_path = f'optim_last.pt'
        torch.save(optimizer.state_dict(), optim_path)

        bucket_name = 'clip_uw_cp'
        source_file_name = f'checkpoint_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)

        source_file_name = f'optim_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS_lp_ft/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)
    else:
        for epoch in range(args.epochs):

            model.train()
            model = model.cuda()
            # data_loader = get_dataloader(
            #     dataset, is_train=True, args=args, image_encoder=image_enc)
            # for i, batch in enumerate(data_loader):
            for i, batch in enumerate(data_loader_train):
                start_time = time.time()

                step = i + epoch * num_batches
                scheduler(step)
                optimizer.zero_grad()

                batch = maybe_dictionarize(batch)
                inputs = batch[input_key].cuda()
                labels = batch['labels'].cuda()
                data_time = time.time() - start_time
                # print(model)

                # from torchsummary import summary
                # summary(model, (3, 224, 224))

                logits = model(inputs)
                loss = loss_fn(logits, labels)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)

                optimizer.step()
                batch_time = time.time() - start_time

                if i % print_every == 0:
                    percent_complete = 100 * i / len(data_loader_train)
                    print(
                        f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(data_loader_train)}]\t"
                        f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                    )

            if args.freeze_encoder:
                image_classifier = ImageClassifier(image_classifier.image_encoder, model.module)
            else:
                image_classifier = model.module

            # # Saving model
            # if args.save is not None and epoch % args.save_ckpt_freq == 0:
            #     os.makedirs(args.save, exist_ok=True)
            #     model_path = f'checkpoint_{epoch+1}.pt'
            #     print('Saving model to', model_path)
            #     image_classifier.save(model_path)
            #     optim_path = f'optim_{epoch+1}.pt'
            #     torch.save(optimizer.state_dict(), optim_path)




            # Evaluate
            args.current_epoch = epoch
            if epoch % args.save_ckpt_freq == 0:
                # eval_results = evaluate(image_classifier, args, device='cuda')
                eval_result = evaluate_(args, data_loader_val, model, device='cuda')
                eval_results['loss'].append(eval_result['loss'])
                eval_results['acc1'].append(eval_result['acc1'])
                eval_results['acc5'].append(eval_result['acc5'])
                print(eval_results)
                if eval_result['acc1'] > best_acc:
                    best_acc = eval_result['acc1']
                    eval_results['best_acc'] = best_acc
                    print(eval_results)

                    ###### upload to Bucket
                    os.makedirs(args.save, exist_ok=True)
                    model_path = f'checkpoint_best.pt'
                    print('Saving model to', model_path)
                    image_classifier.save(model_path)
                    optim_path = f'optim_best.pt'
                    torch.save(optimizer.state_dict(), optim_path)


                    bucket_name = 'clip_uw_cp'
                    source_file_name = f'checkpoint_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                    source_file_name = f'optim_best.pt'
                    save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                    destination_blob_name = f'pretraining/{save_dir}/DS/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                bucket_name = 'clip_uw_cp'
                source_file_name = f'eval_results.pkl'
                save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                destination_blob_name = f'pretraining/{save_dir}/DS/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                pickle_out = pickle.dumps(eval_results)
                upload_pkl(bucket_name, pickle_out, destination_blob_name)

        ###### last epoch upload to Bucket
        os.makedirs(args.save, exist_ok=True)
        model_path = f'checkpoint_last.pt'
        print('Saving model to', model_path)
        image_classifier.save(model_path)
        optim_path = f'optim_last.pt'
        torch.save(optimizer.state_dict(), optim_path)


        bucket_name = 'clip_uw_cp'
        source_file_name = f'checkpoint_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)

        source_file_name = f'optim_last.pt'
        save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
        destination_blob_name = f'pretraining/{save_dir}/DS/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)

    if args.save is not None:
        return model_path


if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)
