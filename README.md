normal transformers we use to have input of size (seq_len). and in the model we used to have a embedding to map this into (seq_len, n_embd) 

This same embedding was also used in the later lm_head.


Now based off Vision transformers. We have a image of size say: HxWxC. 
We take N patches where  = H*W/(P*P) where P = patch size. 
So each patch corresponds to a token of the sequence. And for n_embd we flatten out the patch (P*P*C).

Here we still use a linear projection to map out the flattened values to n_embd

Also, we would want to use non-masked attention as we want all the tokens to look at all other tokens in the sequence
Window attention (Swin Transformer) -> for big sequences/ high res images

Also, we won't be using the same lm_head for wte as in normal transformer -> no weight tying

Using a <CLS> token for prediction. The target token will be a appended CLS token of the sequence

Can't use Normal RoPE -> need to use 2D RoPE oder Axial RoPE as in models like: EVA-02, oder Hiera oder VideoMAE v2

Hierarchical structure (Swin, Hiera)