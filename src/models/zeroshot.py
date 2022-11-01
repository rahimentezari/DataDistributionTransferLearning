import os

import torch
from tqdm import tqdm

import numpy as np

# import clip.clip as clip
import sys
sys.path.insert(1, 'src')
import clip as clip
import templates as templates
import datasets as datasets
import pickle
from args import parse_arguments
from models.modeling import ClassificationHead, ImageEncoder, ImageClassifier
from models.eval import evaluate
from google.cloud import storage


def download_pkl(bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    pickle_in = blob.download_as_string()
    return  pickle.loads(pickle_in)

def get_zeroshot_classifier(args, clip_model):
    assert args.template is not None
    assert args.train_dataset is not None
    template = getattr(templates, args.template)
    logit_scale = clip_model.logit_scale

    device = args.device
    clip_model.eval()
    clip_model.to(device)

    bucket_name = 'clip_uw_cp'
    source_file_name = 'classnames.pkl'
    destination_blob_name = f'datasets/few_shot/{source_file_name}'
    classnames = download_pkl(bucket_name, destination_blob_name)
    # [print(k) for k,v in classnames.items()]
    print(args.train_dataset)
    dataset_classnames = classnames[f'{args.train_dataset.lower()}']
    print(dataset_classnames)
    print(len(dataset_classnames))

    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        # for classname in tqdm(dataset.classnames):
        for classname in tqdm(dataset_classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = clip.tokenize(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def eval(args):
    args.freeze_encoder = True
    if args.load is not None:
        classifier = ImageClassifier.load(args.load)
    else:
        image_encoder = ImageEncoder(args, keep_lang=True)
        classification_head = get_zeroshot_classifier(args, image_encoder.model)
        delattr(image_encoder.model, 'transformer')
        classifier = ImageClassifier(image_encoder, classification_head, process_images=False)
    
    evaluate(classifier, args)

    if args.save is not None:
        classifier.save(args.save)


if __name__ == '__main__':
    args = parse_arguments()
    eval(args)