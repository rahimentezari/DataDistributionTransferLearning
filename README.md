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
Basically you can use the commands in run.sh for different experiments. Each run will load the hyperparameters from [config.json](https://github.com/rahimentezari/DataDisributionTransferLearning/blob/main/config.json) and save results in the Google Bucket.
Below is a short step-by-step how to run Caliban on GCP:
1. sudo apt-get install python3 python3-venv python3-pip
2. sudo usermod -a -G docker ${USER}
3. Install Docker:
Note: check if docker is already installed:
sudo apt-get install -y nvidia-docker2
If not continue:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
4. sudo pkill -SIGHUP dockerd
5. python3 -m pip install --user pipx
6. python3 -m pipx ensurepath
7. source ~/.bashrc (or re-login for the PATH changes to take effect)
8. pipx install caliban
> To check if all is well, run
caliban --help

### Setting up Google Cloud for Caliban
9. Give the account owner the name of the account:
Go to vm details> API and identity management
> Service account 
Add the Service account($$$@developer.gserviceaccount.com) as an owner to the IAMadmin in google console.

10. Also add this to the bucket as storage object admin if you are using Google Bucket

11. gcloud init
- Select the account
- Set default zone to some zone e.g. europe-west4-a (number 14)
12. Add the following lines to the end of “~/.bashrc”
export REGION="your zone e.g. europe-west4 "
export PROJECT_ID="your project ID"

source ~/.bashrc

> Test your Environment: gcloud auth list
13. [Follow these steps to get a JSON file for credentials](https://cloud.google.com/iam/docs/creating-managing-service-account-keys#iam-service-account-keys-create-console)
14. Move the json file to a path
15. Add the following to the end of “~/.bashrc”:
export GOOGLE_APPLICATION_CREDENTIALS=path to the JSON file
16. source ~/.bashrc

Then you can either run caliban [locally](https://caliban.readthedocs.io/en/stable/cli/caliban_run.html) or on the [cloud using GCP Training jobs](https://caliban.readthedocs.io/en/stable/cli/caliban_cloud.html)

