# DataDisributionTransferLearning
This is the code used for "The Role of Pre-training Data in Transfer Learning".
Our CLIP models are trained from scratch on each of the pre-training datasets unless otherwise mentioned and follow the training code from the [OpenCLIP GitHub repository](https://github.com/mlfoundations/open_clip). CLIP models are trained using AdamW optimizer with default PyTorch parameters $\beta_1= 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, batch size 1024, and weight decay of 0.1. For learning rate, we start with a learning rate of $10^{-3}$ and apply a cosine-annealing learning rate schedule~\citep{loshchilov2016sgdr} with 5,000 steps warm-up. We use the same data augmentations as in [SimCLR paper](https://arxiv.org/pdf/2002.05709.pdf). 



### SimCLR training
Our SimCLR implementation closely follows the training code from the [SLIP](https://github.com/facebookresearch/SLIP)
SimCLR models are also trained for 16 epochs from scratch using AdamW optimizer~\citep{loshchilov2017decoupled} with $\beta_1= 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-8}$, batch size 1024, and weight decay of 0.1. we start with a learning rate of $10^{-3}$ and apply a cosine-annealing learning rate schedule with 2 epochs of warm-up. The hidden dimension of SimCLR MLP projection head is set to 4,094 and the output embedding dimension of MLP projection head is set to 256.

### Finetuning detail
Each pretrained model is finetuned on the specific downstream task for 128 epochs while the learning rate is from {0.0001, 0.0003, 0.001, 0.003} as starting and applying a cosine-annealing learning rate schedule with 500 steps warm-up and batch size of 128. For each fine-tuning, we choose the best performing result on the test set among the performed grid search. We use the implementation from the [WiSE-FT GitHub repository](https://github.com/mlfoundations/wise-ft) for fine-tuning, where we have only one model and $\alpha=1$.


### Install dependencies

```bash
conda env create
conda activate DataDisributionTransferLearning
```

### Add directory to PYTHONPATH:

```bash
cd DataDisributionTransferLearning
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Working with Caliban
Most experiments in this repositoty were done using [Caliban](https://github.com/google/caliban). Caliban is a tool for developing research workflow and notebooks in an isolated Docker environment and submitting those isolated environments to Google Compute Cloud.
Below is a short step-by-step how to run Caliban on GCP:

