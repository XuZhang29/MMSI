import torch.nn as nn

class CustomAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(CustomAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
    
    def forward(self, query, key, value, attn_mask=None):
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        if attn_mask is not None:
            if attn_mask.dim() == 4:
                batch_size, num_heads, seq_len, _ = attn_mask.size()
                attn_mask = attn_mask.reshape(batch_size * num_heads, seq_len, seq_len)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.permute(1, 0, 2)

        attn_output, attn_output_weights = self.attention(query, key, value, attn_mask=attn_mask)
        attn_output = attn_output.permute(1, 0, 2)  
        
        return attn_output, attn_output_weights
