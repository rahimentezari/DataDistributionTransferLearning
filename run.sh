################## Open_clip RN50
#CUDA_VISIBLE_DEVICES=0 python src/wise_ft.py --train-dataset "ImageNet" --upstream_dataset "YFCC_15m" --epochs 128 --freezeencoder 0 --lr 0.0001 --batch-size 128 --wd 0.1 --upstream_loss CLIP --upstream_arch "RN50" --model "bucket" --template "openai_imagenet_template" --save_ckpt_freq 1 --results-db results.json --save models/wiseft/RN50 --data-location /mnt/external/ImageNet1K --alpha 1.0 --seed 0
#caliban run --experiment_config config.json src/wise_ft.py
#caliban cloud --name rn50_cifar100 --xgroup rn50_cifar100 --experiment_config config.json --gpu_spec 1xV100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name rn50_cifar100 --xgroup rn50_cifar100 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py



#####CC_12m_2.7m

#caliban cloud --name rn50_dtd --xgroup rn50_dtd --experiment_config config.json --gpu_spec 1xK80  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name rn50_dtd_128 --xgroup rn50_dtd_128 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name rn50_cifar100 --xgroup rn50_cifar100 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name cc12m_dtd_cifar_full --xgroup cc12m_dtd_cifar_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py


#####Redcaps_2.7m
###
#caliban cloud --name dtd_few_not128 --xgroup dtd_few_not128 --experiment_config config.json --gpu_spec 1xK80  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name dtd_few_128 --xgroup dtd_few_128 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name redcaps_cifar100_few --xgroup redcaps_cifar100_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name redcaps_dtd_cifar100_full --xgroup redcaps_dtd_cifar100_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py



#####YFCC_2.7m
#caliban cloud --name yfcc_dtd_cifar100_full --xgroup yfcc_dtd_cifar100_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name yfcc_dtd_few_not128 --xgroup yfcc_dtd_few_not128 --experiment_config config.json --gpu_spec 1xK80  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name yfcc_dtd_few_not128 --xgroup yfcc_dtd_few_not128 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name yfcc_dtd_few_128 --xgroup yfcc_dtd_few_128 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name yfcc_cifar100_few --xgroup yfcc_cifar100_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
### IN
#caliban cloud --name yfcc_IN_few --xgroup yfcc_IN_few --experiment_config config.json --gpu_spec 4xP100  --machine_type 'n1-standard-32' src/wise_ft.py
#caliban cloud --name yfcc_IN_full --xgroup yfcc_cifar100_few --experiment_config config.json --gpu_spec 4xV100  --machine_type 'n1-standard-32' src/wise_ft.py



#####LAION_2.7m
#caliban cloud --name LAION2_dtd_few_not128 --xgroup LAION2_dtd_few_not128 --experiment_config config.json --gpu_spec 1xK80  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name LAION2_dtd_few_128 --xgroup LAION2_dtd_few_128 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name LAION2_cifar100_few --xgroup LAION2_cifar100_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name LAION2_dtd_cifar100_full --xgroup LAION2_dtd_cifar100_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
### scratch
#caliban cloud --name LAION2_cifar100_20_scratch --xgroup LAION2_cifar100_20_scratch --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name LAION_IN_full --xgroup LAION_IN_full --experiment_config config.json --gpu_spec 4xV100  --machine_type 'n1-standard-32' src/wise_ft.py

#####WIT_2.7m
#caliban cloud --name WIT_dtd_cifar_few_not128 --xgroup WIT_dtd_cifar_few_not128 --experiment_config config.json --gpu_spec 1xK80  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name WIT_dtd_cifar_few_128 --xgroup WIT_dtd_cifar_few_128 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name WIT_dtd_cifar100_full --xgroup WIT_dtd_cifar100_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py

#####shutterstock_2.7m
#caliban cloud --name Shutter_dtd_cifar_few_not128 --xgroup Shutter_dtd_cifar_few_not128 --experiment_config config.json --gpu_spec 1xK80  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name Shutter_dtd_cifar_few_128 --xgroup Shutter_dtd_cifar_few_128 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name Shutter_dtd_cifar100_full --xgroup Shutter_dtd_cifar100_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py



########### dataset size : scratch_!!!
#caliban cloud --name datasetsize_dtd_cifar100 --xgroup datasetsize_dtd_cifar100 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name datasetsize_caltech --xgroup datasetsize_caltech --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py

## cifar100_100 shots
#caliban cloud --name datasetsize_cifar100_few100 --xgroup datasetsize_cifar100_few100 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py





####### more datasets: CALTECH, REAL, CLIPART, PETS
#caliban cloud --name caltech --xgroup caltech --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name domainnet_7us --xgroup domainnet_7us --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name real_failed --xgroup real_failed --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clipart_failed_shot5 --xgroup clipart_failed_shot5 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clipart_real_10_20shots --xgroup clipart_real_10_20shots --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name pets_7us_full --xgroup pets_7us_full --experiment_config config.json --gpu_spec 1xV100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name pets2_7us_full --xgroup pets2_7us_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name pets_7us_few --xgroup pets_7us_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py


###### copare 16 epochs with 40 epochs
#caliban cloud --name caltech_cc3m_16vs40 --xgroup caltech_cc3m_16vs40 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name caltech_cc3m_16vs40_full --xgroup caltech_cc3m_16vs40_full --experiment_config config.json --gpu_spec 1xV100  --machine_type 'n1-standard-8' src/wise_ft.py




###### upper and lowerbound
#caliban cloud --name LAION2_dtd_full_scratch --xgroup LAION2_dtd_full_scratch --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name LAION2_CIFAR100_full_scratch --xgroup LAION2_CIFAR100_full_scratch --experiment_config config.json --gpu_spec 1xV100  --machine_type 'n1-standard-8' src/wise_ft.py

#caliban cloud --name LAION2_dtd_full_openai --xgroup LAION2_dtd_full_scratch --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name LAION2_CIFAR100_full_openai --xgroup LAION2_CIFAR100_full_scratch --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name scratch_LAION2 --xgroup scratch_LAION2 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py







############## IN captions
#caliban cloud --name INcaptions_dtd_full --xgroup INcaptions_dtd_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name INcaptions_dtd_pets --xgroup INcaptions_dtd_pets --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name INcaptions_caltech --xgroup INcaptions_caltech --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name INcaptions_cifar100 --xgroup INcaptions_cifar100 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name INcaptions_clipart --xgroup INcaptions_clipart --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name INcaptions_real_few --xgroup INcaptions_real_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name INcaptions_real_full --xgroup INcaptions_real_full --experiment_config config.json --gpu_spec 2xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name INcaptions_test --xgroup INcaptions_test --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name INcaptions_all --xgroup INcaptions_all --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py





################### CLIP_RN50_YFCC_0.5 on 6 DS:
#caliban cloud --name YFCC_halfM_dtd --xgroup YFCC_halfM_dtd --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name YFCC_halfM_pets --xgroup YFCC_halfM_pets --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name YFCC_halfM_caltech --xgroup YFCC_halfM_caltech --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name YFCC_halfM_cifar100_clipart --xgroup YFCC_halfM_cifar100_clipart --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name YFCC_halfM_real_few --xgroup YFCC_halfM_real_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name YFCC_halfM_real_full --xgroup YFCC_halfM_real_full --experiment_config config.json --gpu_spec 2xP100  --machine_type 'n1-standard-8' src/wise_ft.py




########## SimCLR
#caliban cloud --name dtd_sim --xgroup dtd_sim --experiment_config config.json --gpu_spec 2xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_cifar100_full --xgroup sim_cifar100_full --experiment_config config.json --gpu_spec 2xV100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_caltech_cifar100 --xgroup sim_caltech_cifar100 --experiment_config config.json --gpu_spec 2xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_caltech --xgroup sim_caltech --experiment_config config.json --gpu_spec 2xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_cifar100_few --xgroup sim_cifar100_few --experiment_config config.json --gpu_spec 2xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_cifar100_few_lp --xgroup sim_cifar100_few_lp --experiment_config config.json --gpu_spec 2xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_lp --xgroup sim_lp --experiment_config config.json --gpu_spec 2xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_lp_ft_cifar100_few --xgroup sim_lp_ft_cifar100_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_lp_ft_cifar100_full --xgroup sim_lp_ft_cifar100_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_lp_ft_dtd --xgroup sim_lp_ft_dtd --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py

#caliban cloud --name clip_lp_cifar100 --xgroup clip_lp_cifar100 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clip_lp_ft_cifar100 --xgroup clip_lp_ft_cifar100 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py

#####sim and clip lp_ft larger grid
#caliban cloud --name clip_sim_lp_ft_cifar100 --xgroup clip_sim_lp_ft_cifar100 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_lp_ft_cifar100 --xgroup sim_lp_ft_cifar100 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py


######### CLIP: LP, lp_ft
#caliban cloud --name sim_lp_dtd_caltech --xgroup sim_lp_dtd_caltech --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clip_lp_dtd_caltech --xgroup clip_lp_dtd_caltech --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_clip_lp_dtd_caltech --xgroup sim_clip_lp_dtd_caltech --experiment_config config.json --gpu_spec 1xK80  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clip_sim_lp_ft_cifar_full --xgroup clip_sim_lp_ft_cifar_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clip_lp_ft_cifar_full --xgroup clip_lp_ft_cifar_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py




#caliban cloud --name clip_lp_dtd_caltech_full --xgroup clip_lp_dtd_caltech_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clip_lp_dtd_caltech_few --xgroup clip_lp_dtd_caltech_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clip_lp_dtd_caltech_few2 --xgroup clip_lp_dtd_caltech_few2 --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py

#caliban cloud --name clip_lp_ft_dtd_caltech_full --xgroup clip_lp_dtd_caltech_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clip_lp_cifar_full --xgroup clip_lp_cifar_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py

#caliban cloud --name clip_lp_ft_dtd_caltech_few --xgroup clip_lp_ft_dtd_caltech_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name sim_lp_ft_dtd_few_strange --xgroup sim_lp_ft_dtd_few_strange --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py





############### LAION2B
#caliban cloud --name clip_LAION2B_cifar100_few --xgroup clip_LAION2B_cifar100_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name clip_LAION2B_dtd_caltech_few --xgroup clip_LAION2B_dtd_caltech_few --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name LAION2B_cifar_dtd_caltech_full --xgroup LAION2B_cifar_dtd_caltech_full --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py


######## YFCC 1m
#caliban cloud --name yfcc_1m --xgroup yfcc_1m --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name laion_15m --xgroup laion_15m --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py



########################################################### new dataset
#### EuroSAT
#caliban run --experiment_config config.json src/wise_ft.py
#caliban cloud --name cassaleaf --xgroup cassaleaf --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name cameratraps --xgroup cameratraps --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
#caliban cloud --name eurosat --xgroup eurosat --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
caliban cloud --name finegrained_hyper --xgroup finegrained_hyper --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py




##### openclip
#python  src/training/main.py     --save-frequency 2     --zeroshot-frequency 5     --report-to wandb,tensorboard   --train-data="/data/yfcc-tmp/yfcc/chunks_merged_1e3/shard_{00000..00500}.tar"  --val-data="/data/yfcc-tmp/yfcc/chunks_merged_1e3/shard_{14720..14729}.tar"    --imagenet-val=/data/yfcc-tmp/data/imagenet/val    --warmup 5000     --batch-size=1024 --lr=1e-3     --wd=0.1     --epochs=32     --workers=8 --dist-url tcp://localhost:10034


#caliban cloud --name rn50_dtd --xgroup rn50_dtd --experiment_config config.json --gpu_spec 1xP100  --machine_type 'n1-standard-8' src/wise_ft.py
### IN
#caliban run --experiment_config config.json src/wise_ft.py


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python src/wise_ft.py   \
#    --train-dataset=cifar100  \
#    --epochs=100  \
#    --lr=0.00003  \
#    --batch-size=100  \
#    --cache-dir=cache  \
#    --model=/home/rahiment/SLIP/CLIP_RN50_cc_3m/epoch_26.pt \
#    --eval-datasets=cifar100  \
#    --template=openai_imagenet_template  \
#    --results-db=results.jsonl  \
#    --save=models/wiseft/RN50  \
#    --data-location=~/data \
#    --alpha 1 \
#    --shots 10


