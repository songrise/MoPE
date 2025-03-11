# MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts｜ [Arxiv](https://arxiv.org/abs/2403.10568)
Official Implementation of MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

![fig_overview-v2](fig_overview-v2.png)

TL;DR: MoPE is a prompt-based method to fuse unimodal pretrained models (e.g., ViT, Bert, Wav2Vec) for downstream multimodal tasks. MoPE is parameter-efficient and scalable, and achieves state-of-the-art performance on various multimodal tasks.

 The key innovation of MoPE is that we decompose long prompts into short and specialized prompt experts, which are routed instance-wisely with a multimodal router.

:fire: :fire: Update (2025/3/11): We released preliminary code for MoPE for vision-language classification.

## MoPE Result

![image](https://github.com/user-attachments/assets/af532d87-9692-4600-967d-f987d6e87ee9)


## MoPE Visualization

![fig_route](route_example.png)

## Quick Start
### Environment and Datasets
First install dependencies (tested with Python=3.8) 

`pip install -r requirements.txt`

Download the pretrained Swin-base model from [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) (alternative link [here](https://github.com/microsoft/Swin-Transformer?tab=readme-ov-file)). Put the `swin_base_patch4_window7_224_22k.pth` into the `pretrained` folder.

Download datasets put them into the `data` folder. The classification datasets include: [UPMC Food-101](https://github.com/artelab/Image-and-Text-fusion-for-UPMC-Food-101-using-BERT-and-CNNs/issues/3), [SNLI-VE](https://github.com/necla-ml/SNLI-VE), and [MM-IMDB](https://github.com/johnarevalo/gmu-mmimdb?tab=readme-ov-file). For MUStARD dataset, refer to [here](https://github.com/liangsheng02/Modular-and-Parameter-Efficient-Multimodal-Fusionwith-Prompting). 

After downloading, you may use `utils/process_food_101.py`, `utils/process_mm_imdb.py` to preprocess the datasets into JsonL format.

The data folder should look like this:
```
data
├── food-101
├── mmimdb
├── snli_ve
```

### Training
Train the MoPE model use the following command, for example on MM-IMDB:
```
python main_classify.py --exp_name full_imdb --use_vpt --use_pbert --fuse_method mope --train_instructor --dataset imdb --prompt_length 6 --moe_n_experts 4 --t_prompt_length 4 --lr_vis 4e-4 --lr_text 5e-4  --w_imp 0.01 --use_instruct 
```

Monitor the training process with tensorboard:
```
tensorboard --logdir /logs --port 6006
```


## Conclusion
If you have any questions, please feel free to contact me via email or Github issue. If you find this project useful, please consider citing our paper. 

```
@article{jiang2024mope,
  title={MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts},
  author={Jiang, Ruixiang and Liu, Lingbo and Chen, Changwen},
  journal={arXiv preprint arXiv:2403.10568},
  year={2024}
}
```
