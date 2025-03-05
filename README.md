# MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts｜ [Arxiv](https://arxiv.org/abs/2403.10568)
Official Implementation of MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

![fig_overview-v2](fig_overview-v2.png)

What is MoPE? MoPE is a prompt-based method to fuse unimodal pretrained models (e.g., ViT, Bert, Wav2Vec) for downstream multimodal tasks. MoPE is parameter-efficient and scalable, and achieves state-of-the-art performance on various multimodal tasks. The key innovation of MoPE is that we decompose long prompts into short and specialized prompt experts, which are routed instance-wisely with a multimodal router.

## MoPE Result

![image](https://github.com/user-attachments/assets/af532d87-9692-4600-967d-f987d6e87ee9)


## MoPE Visualization

![fig_route](route_example.png)
