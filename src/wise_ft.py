import os

import numpy as np
# some_file.py
import sys
sys.path.insert(1, 'src')
import torch

from models.finetune import finetune
from models.modeling import ImageEncoder, ImageClassifier
from models.utils import fisher_load
from models.zeroshot import get_zeroshot_classifier
from args import parse_arguments
from google.cloud import storage
import pickle
import pandas as pd
import random
import models.simclr.ResNet as ResNet
from collections import OrderedDict
from torch import nn
from models.simclr import models
# from models.simclr.models import models
from models.modeling import ClassificationHead


def download_pkl(bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    pickle_in = blob.download_as_string()
    return pickle.loads(pickle_in)

def download_blob(bucket_name, source_blob_name, destination_file_name,
                  blob_path_prefix=""):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path_prefix + source_blob_name)
    blob.download_to_filename(destination_file_name)

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


def _merge(alpha, theta_0, theta_1, fishers, fisher_floor):
    if fishers is None:
        # interpolate between all weights in the checkpoints
        return {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

    fisher_0, fisher_1 = fishers

    theta = {}
    for key in theta_0.keys():
        # Make sure that either we have a Fisher for this variable for
        # both checkpoints or none of the checkpoints. Default to regular
        # interpolation if no Fisher is found.
        assert (key in fisher_0) == (key in fisher_1)
        ones = torch.ones_like(theta_0[key])
        f_0 = torch.maximum(fisher_0.get(key, ones), fisher_floor * ones)
        f_1 = torch.maximum(fisher_1.get(key, ones), fisher_floor * ones)

        c_0 = (1 - alpha) * f_0
        c_1 = alpha * f_1

        theta[key] = (c_0 * theta_0[key] + c_1 * theta_1[key]) / (c_0 + c_1)

    return theta


def wise_ft(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    assert args.save is not None, 'Please provide a path to store models'
    if args.load is None:
        ##### download from bucket
        ##### first read the pretainig hyperparam from Bucket
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
            bucket_name = 'clip_uw_cp'
            destination_blob_name = 'params_Thao.csv'
            source_file_name = f'codes/{destination_blob_name}'
            params = download_blob(bucket_name, source_file_name, destination_blob_name)
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

        if args.model == 'bucket':
            print("Loading pretrained checkpoint from Google Bucket. Please wait :)")
            bucket_name = 'clip_uw_cp'
            if 'captions' in args.upstream_dataset or 'YFCC_0.5m' in args.upstream_dataset or 'LAION_400m' in args.upstream_dataset:
                destination_blob_name = 'epoch_32.pt'
                load_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                source_file_name = f'pretraining/{load_dir}/US/LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/{destination_blob_name}'
                download_blob(bucket_name, source_file_name, destination_blob_name)
                args.model = 'epoch_32.pt'
            elif 'ImageNet1K_Captions' in args.upstream_dataset:
                destination_blob_name = 'epoch_90.pt'
                load_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                source_file_name = f'pretraining/{load_dir}/US/LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/{destination_blob_name}'
                download_blob(bucket_name, source_file_name, destination_blob_name)
                args.model = 'epoch_90.pt'
            elif args.finetune_strategy == 'lp_ft':
                destination_blob_name = 'epoch_16.pt'
                load_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                source_file_name = f'pretraining/{load_dir}/US/LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/{destination_blob_name}'
                download_blob(bucket_name, source_file_name, destination_blob_name)

                destination_blob_name = 'checkpoint_best.pt'
                load_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                source_file_name = f'pretraining/{load_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_128_LR_0.003_BS_128_WD_0.1/{destination_blob_name}'
                download_blob(bucket_name, source_file_name, destination_blob_name)

                args.model = 'epoch_16.pt'
                args.model_lp = 'checkpoint_best.pt'

            else:
                destination_blob_name = 'epoch_16.pt'
                load_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                source_file_name = f'pretraining/{load_dir}/US/LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/{destination_blob_name}'
                download_blob(bucket_name, source_file_name, destination_blob_name)
                args.model = 'epoch_16.pt'


        # Build and save zero-shot model
        image_encoder = ImageEncoder(args, keep_lang=True)
        if 'SimCLR' in args.upstream_loss:
            bucket_name = 'clip_uw_cp'
            source_file_name = 'classnames.pkl'
            destination_blob_name = f'datasets/few_shot/{source_file_name}'
            classnames = download_pkl(bucket_name, destination_blob_name)
            dataset_classnames = classnames[f'{args.train_dataset.lower()}']
            print(len(dataset_classnames))

            weights = torch.zeros(len(dataset_classnames), 1024)
            classification_head = ClassificationHead(normalize=True, weights=weights)
        else:
            classification_head = get_zeroshot_classifier(args, image_encoder.model)

        if 'SimCLR' in args.upstream_loss:
            if args.finetune_strategy == 'lp_ft':
                classifier = ImageClassifier.load(args.model_lp)
            else:
                classifier = ImageClassifier(image_encoder, classification_head, process_images=False)

            args.classifier = classifier
            args.load = 'empty'
        elif 'CLIP' in args.upstream_loss:
            if args.finetune_strategy == 'lp_ft':
                classifier = ImageClassifier.load(args.model_lp)
                args.classifier = classifier
                args.load = 'empty'  ### already loaded to the classifier

            elif args.finetune_strategy == 'ft':
                delattr(image_encoder.model, 'transformer')
                classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
                zeroshot_checkpoint = 'zeroshot.pt'
                classifier.save(zeroshot_checkpoint)
                ###### upload to Bucket
                bucket_name = 'clip_uw_cp'
                source_file_name = 'zeroshot.pt'
                save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                destination_blob_name = f'pretraining/{save_dir}/DS/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                upload_blob(bucket_name, source_file_name, destination_blob_name)
                # Standard fine-tuning
                args.load = zeroshot_checkpoint

            elif args.finetune_strategy == 'lp':
                delattr(image_encoder.model, 'transformer')
                classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
                zeroshot_checkpoint = 'zeroshot.pt'
                classifier.save(zeroshot_checkpoint)
                ###### upload to Bucket
                bucket_name = 'clip_uw_cp'
                source_file_name = 'zeroshot.pt'
                save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
                destination_blob_name = f'pretraining/{save_dir}/DS_LinearProbe/{args.train_dataset}/shots_{args.shots}/US_Epochs_{pretrain_epochs}_LR_{pretrain_lr}_BS_{pretrain_bs}_WD_{pretrain_wd}/DS_Epochs_{args.epochs}_LR_{args.lr}_BS_{args.batch_size}_WD_{args.wd}/{source_file_name}'
                upload_blob(bucket_name, source_file_name, destination_blob_name)
                # Standard fine-tuning
                args.load = zeroshot_checkpoint
        args.save = os.path.join(args.save, 'finetuned')
        finetuned_checkpoint = finetune(args)
    else:
        # No need to compute things from stratch
        assert len(args.load) == 2
        zeroshot_checkpoint, finetuned_checkpoint = args.load

    # # Load models
    # zeroshot = ImageClassifier.load(zeroshot_checkpoint)
    # finetuned = ImageClassifier.load(finetuned_checkpoint)
    # theta_0 = {k: v.clone() for k, v in zeroshot.state_dict().items()}
    # theta_1 = {k: v.clone() for k, v in finetuned.state_dict().items()}
    # del zeroshot
    #
    # if args.fisher is None:
    #     fishers = None
    # else:
    #     fisher_0_file, fisher_1_file = args.fisher
    #     fisher_0 = fisher_load(os.path.expanduser(fisher_0_file))
    #     fisher_1 = fisher_load(os.path.expanduser(fisher_1_file))
    #     fishers = fisher_0, fisher_1
    #
    # # make sure checkpoints are compatible
    # assert set(theta_0.keys()) == set(theta_1.keys())


    # alphas = args.alpha
    # for alpha in alphas:
    #     args.alpha = alpha
    #
    #     theta = _merge(alpha, theta_0, theta_1, fishers, args.fisher_floor)
    #
    #     # update the model (in-place) acccording to the new weights
    #     finetuned.load_state_dict(theta)
    #
    #     # save model
    #     finetuned.save(os.path.join(args.save, f'wise_ft_alpha={alpha:.3f}.pt'))
    #     ###### upload to Bucket
    #     bucket_name = 'clip_uw_cp'
    #     source_file_name = 'zeroshot.pt'
    #     save_dir = f'{args.upstream_loss}_{args.upstream_arch}_{args.upstream_dataset}'
    #     destination_blob_name = f'pretraining/{save_dir}/DS/{args.train_dataset}/US_/LR_{args.lr}_BS_{args.batch_size}/{source_file_name}'
    #     upload_blob(bucket_name, source_file_name, destination_blob_name)
    #
    #     # evaluate
    #     evaluate(finetuned, args)


if __name__ == '__main__':
    args = parse_arguments()
    if args.freezeencoder == '1':
        args.freeze_encoder = True
    else:
        args.freeze_encoder = False
    print(args.freeze_encoder, args.freezeencoder)

    # save_ckpt = np.random.randint(100, size=1)
    # args.save_ckpt_freq = save_ckpt

    wise_ft(args)
