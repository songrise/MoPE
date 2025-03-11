import torch
import torch.nn as nn
import einops
import clip

class CLIPTextTransformer(nn.Module):
    """
    Transfromer encoder (text) for CLIP
    """
    def __init__(self, use_coop:bool = True, n_ctx:int = 6) -> None:
        super().__init__()
        self.clip_model = clip.load('ViT-B/32', jit=False)[0].cuda()
        self.learnable_context = None
        self.use_coop = use_coop #global context for all classes
        if use_coop:
            self.n_ctx = n_ctx
            context_vectors = torch.empty(self.n_ctx, self.clip_model.ln_final.weight.shape[0])
            torch.nn.init.normal_(context_vectors, std=.02)
            self.learnable_context = nn.Parameter(context_vectors) # [n_ctx, 512]
        self.head = nn.Linear(512,768)

    def forward(self, text, **kwargs):
        """
        Input:
            text: tokenized text, shape = [batch_size, n_ctx]
        """
        text = [' '.join(t[:75]) for t in text]
        text = clip.tokenize(text).type(torch.long).cuda()
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
        if self.use_coop:
            sos_token = x[:, 0, :].unsqueeze(1)  # [batch_size, 1, d_model]
            suffix_tokens = x[:, 1:-self.n_ctx, :] # class tokens + [EOS] token
            ctx = einops.repeat(self.learnable_context, 'n d -> b n d', b=x.shape[0])
            x = torch.cat([sos_token, ctx, suffix_tokens], dim=1)
        

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.visual.conv1.weight.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        x = self.head(x)
        # x = x.  # [batch_size, 1, transformer.width]
        return x, x