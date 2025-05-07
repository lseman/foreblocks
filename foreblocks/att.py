import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import xformers.ops as xops
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

class AttentionLayer(nn.Module):
    def __init__(
        self,
        decoder_hidden_size,
        encoder_hidden_size=None,
        attention_size=None,
        method='dot',
        nhead=4,
        dropout=0.1,
        use_xformers=True,
        attention_backend='xformers',  # Options: 'xformers', 'torch', 'prob', 'temporal', 'multiscale'
        time_embed_dim=8,  # for temporal attention
        num_scales=10  # number of attention scales for multiscale attention
    ):
        super(AttentionLayer, self).__init__()
        # print the attention method and backend
        print(f"Attention method: {method}, Attention backend: {attention_backend}")
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size or decoder_hidden_size
        self.attention_size = attention_size or decoder_hidden_size
        self.method = method
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.use_xformers = use_xformers and HAS_XFORMERS
        self.attention_backend = attention_backend
        self.time_embed_dim = time_embed_dim
        self.num_scales = num_scales

        self.needs_projection = self.decoder_hidden_size != self.encoder_hidden_size
        if self.needs_projection:
            self.encoder_projection = nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)

        if method == 'mha':
            assert decoder_hidden_size % nhead == 0, "Hidden size must be divisible by number of heads"
            self.head_dim = decoder_hidden_size // nhead
            self.q_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)
            self.k_proj = nn.Linear(self.encoder_hidden_size, decoder_hidden_size)
            self.v_proj = nn.Linear(self.encoder_hidden_size, decoder_hidden_size)
            self.out_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)
            self.scale = math.sqrt(self.head_dim)
            
            # Multi-scale attention components
            if attention_backend == 'multiscale':
                # Create multiple key/value projections for different scales
                self.k_projs = nn.ModuleList([
                    nn.Linear(self.encoder_hidden_size, decoder_hidden_size) 
                    for _ in range(num_scales)
                ])
                self.v_projs = nn.ModuleList([
                    nn.Linear(self.encoder_hidden_size, decoder_hidden_size) 
                    for _ in range(num_scales)
                ])
                # Scale mixing weights (learnable)
                self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
                # Dilations for different scales (increasing receptive field)
                self.dilations = [1, 2, 4][:num_scales]  # Adjust as needed
                # Final projection to combine multi-scale outputs
                self.scale_out_proj = nn.Linear(decoder_hidden_size, decoder_hidden_size)

        if attention_backend == 'temporal':
            self.time_bias = nn.Linear(time_embed_dim, self.encoder_hidden_size)

        self.combined_layer = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.decoder_hidden_size)

    def _prepare_decoder_hidden(self, decoder_hidden):
        if isinstance(decoder_hidden, tuple):
            decoder_hidden = decoder_hidden[0]
        if decoder_hidden.dim() == 3:
            decoder_hidden = decoder_hidden[-1]
        return decoder_hidden  # [B, D]

    def _split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x, batch_size):
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.head_dim)
    
    def _dilate_sequence(self, x, dilation):
        """Apply dilation to a sequence by selecting every dilation-th element"""
        # x: [B, T, D]
        batch_size, seq_len, dim = x.size()
        if dilation == 1:
            return x
        
        # Calculate the new sequence length after dilation
        new_seq_len = seq_len // dilation + (1 if seq_len % dilation else 0)
        
        # Create indices for dilated sampling
        indices = torch.arange(0, seq_len, dilation, device=x.device)
        indices = indices[indices < seq_len]  # Ensure indices are valid
        
        # Apply dilation by selecting entries at the specified indices
        dilated_x = x[:, indices, :]
        
        return dilated_x

    def _mha_attention(self, query, key, value):
        batch_size = query.size(0)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._split_heads(q, batch_size)
        k = self._split_heads(k, batch_size)
        v = self._split_heads(v, batch_size)

        if self.use_xformers and self.attention_backend == 'xformers':
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = xops.memory_efficient_attention(q, k, v)
            attn_output = attn_output.transpose(1, 2)
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)

        attn_output = self._combine_heads(attn_output, batch_size)
        attn_output = self.out_proj(attn_output)
        return attn_output, None
    
    def _multiscale_attention(self, query, key, value):
        """Multi-scale attention that captures dependencies at different time scales"""
        batch_size = query.size(0)
        q = self.q_proj(query)  # [B, 1, D]
        q = self._split_heads(q, batch_size)  # [B, H, 1, D/H]
        
        # Multi-scale processing
        outputs = []
        attentions = []
        
        # Process each scale
        for i, dilation in enumerate(self.dilations):
            # Apply dilation to key and value
            dilated_key = self._dilate_sequence(key, dilation)
            dilated_value = self._dilate_sequence(value, dilation)
            
            # Project dilated sequences
            k_i = self.k_projs[i](dilated_key)  # [B, T/dilation, D]
            v_i = self.v_projs[i](dilated_value)  # [B, T/dilation, D]
            
            # Split into heads
            k_i = self._split_heads(k_i, batch_size)  # [B, H, T/dilation, D/H]
            v_i = self._split_heads(v_i, batch_size)  # [B, H, T/dilation, D/H]
            
            # Apply attention
            if self.use_xformers and self.attention_backend == 'xformers':
                q_i = q.transpose(1, 2)  # [B, 1, H, D/H]
                k_i = k_i.transpose(1, 2)  # [B, T/dilation, H, D/H]
                v_i = v_i.transpose(1, 2)  # [B, T/dilation, H, D/H]
                out_i = xops.memory_efficient_attention(q_i, k_i, v_i)
                out_i = out_i.transpose(1, 2)  # [B, H, 1, D/H]
            else:
                out_i = F.scaled_dot_product_attention(q, k_i, v_i, dropout_p=self.dropout.p)
            
            # Combine heads and save output
            out_i = self._combine_heads(out_i, batch_size)  # [B, 1, D]
            outputs.append(out_i)
        
        # Weighted combination of multi-scale outputs
        scale_weights = F.softmax(self.scale_weights, dim=0)
        combined_output = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            combined_output = combined_output + scale_weights[i] * output
        
        # Final projection
        combined_output = self.scale_out_proj(combined_output)
        
        return combined_output, None

    def forward(self, decoder_hidden, encoder_outputs, encoder_timestamps=None):
        batch_size, src_len, _ = encoder_outputs.size()
        decoder_hidden = self._prepare_decoder_hidden(decoder_hidden)

        if self.needs_projection:
            encoder_outputs = self.encoder_projection(encoder_outputs)

        # Add time-aware bias if using 'temporal' attention
        if self.attention_backend == 'temporal' and encoder_timestamps is not None:
            encoder_outputs = encoder_outputs + self.time_bias(encoder_timestamps)

        # Multi-scale attention
        if self.method == 'mha' and self.attention_backend == 'multiscale':
            query = decoder_hidden.unsqueeze(1)  # [B, 1, D]
            attn_output, _ = self._multiscale_attention(query, encoder_outputs, encoder_outputs)
            context = attn_output.squeeze(1)
            combined = self.combined_layer(torch.cat((context, decoder_hidden), dim=1))
            combined = torch.tanh(combined)
            return context, None

        # Multi-Head Attention
        elif self.method == 'mha' and self.attention_backend in ['xformers', 'torch']:
            query = decoder_hidden.unsqueeze(1)  # [B, 1, D]
            attn_output, _ = self._mha_attention(query, encoder_outputs, encoder_outputs)
            context = attn_output.squeeze(1)
            combined = self.combined_layer(torch.cat((context, decoder_hidden), dim=1))
            combined = torch.tanh(combined)
            return context, None

        # ProbSparse Attention (top-k approximation)
        elif self.attention_backend == 'prob':
            query = decoder_hidden.unsqueeze(1)  # [B, 1, D]
            scores = torch.bmm(query, encoder_outputs.transpose(1, 2))  # [B, 1, T]
            topk = max(1, int(encoder_outputs.size(1) * 0.3))
            topk_values, topk_indices = torch.topk(scores, topk, dim=-1)
            mask = torch.zeros_like(scores).scatter_(-1, topk_indices, 1.0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)  # [B, 1, T]
            context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)  # [B, D]
            combined = self.combined_layer(torch.cat((context, decoder_hidden), dim=1))
            combined = torch.tanh(combined)
            return context, attn_weights.squeeze(1)

        # Dot-product attention (default)
        else:
            attn_weights = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
            attn_weights = F.softmax(attn_weights, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            combined = self.combined_layer(torch.cat((context, decoder_hidden), dim=1))
            combined = torch.tanh(combined)
            return context, attn_weights