import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils import get_activation_function


# Perform MLP on last dimension
# activation: elu elu gelu hardshrink hardtanh leaky_relu prelu relu rrelu tanh
class MLP(nn.Module):
    def __init__(self, activate, d_in, d_hidden, d_out, bias):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=bias)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=bias)
        self.activation = get_activation_function(activate)
    
    # x: [bs, l, k, d] k=modalityKinds mask: [bs, l]
    def forward(self, x, mask=None):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# d_ins=[l, k, d]'s inputlength same for d_hiddens,dropouts
class MLPsBlock(nn.Module):    
    def __init__(self, activate, d_ins, d_hiddens, d_outs, dropouts, bias, ln_first=False, res_project=False):
        super(MLPsBlock, self).__init__()
        self.mlp_l = MLP(activate, d_ins[0], d_hiddens[0], d_outs[0], bias)
        self.mlp_k = MLP(activate, d_ins[1], d_hiddens[1], d_outs[1], bias)
        self.mlp_d = MLP(activate, d_ins[2], d_hiddens[2], d_outs[2], bias)
        self.dropout_l = nn.Dropout(p=dropouts[0])
        self.dropout_k = nn.Dropout(p=dropouts[1])
        self.dropout_d = nn.Dropout(p=dropouts[2])
        if ln_first:
            self.ln_l = nn.LayerNorm(d_ins[0], eps=1e-6)
            self.ln_k = nn.LayerNorm(d_ins[1], eps=1e-6)
            self.ln_d = nn.LayerNorm(d_ins[2], eps=1e-6)
        else:
            self.ln_l = nn.LayerNorm(d_outs[0], eps=1e-6)
            self.ln_k = nn.LayerNorm(d_outs[1], eps=1e-6)
            self.ln_d = nn.LayerNorm(d_outs[2], eps=1e-6)

        self.ln_fist = ln_first
        self.res_project = res_project
        if not res_project:
            assert d_ins[0]==d_outs[0], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
            assert d_ins[1]==d_outs[1], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
            assert d_ins[2]==d_outs[2], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
        else:
            self.res_projection_l = nn.Linear(d_ins[0], d_outs[0], bias=False)
            self.res_projection_k = nn.Linear(d_ins[1], d_outs[1], bias=False)
            self.res_projection_d = nn.Linear(d_ins[2], d_outs[2], bias=False)
    
    # x: [bs, l, k, d] k=modalityKinds mask: [bs, l]
    def forward(self, x, mask=None):
        if mask is not None:
            print("Warning from MLPsBlock: If using mask, d_in should be equal to d_out.")
        if self.ln_fist:
            x = self.forward_ln_first(x, mask)
        else:
            x = self.forward_ln_last(x, mask)
        return x

    def forward_ln_first(self, x, mask):
        if self.res_project:
            residual_l = self.res_projection_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            residual_l = x
        x = self.ln_l(x.permute(0, 2, 3, 1))
        x = self.mlp_l(x, None).permute(0, 3, 1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).bool(), 0.0) # Fill mask=True to 0.0
        x = self.dropout_l(x)
        x = x + residual_l
        
        if self.res_project:
            residual_k = self.res_projection_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        else:
            residual_k = x
        x = self.ln_k(x.permute(0, 1, 3, 2))
        x = self.dropout_k(self.mlp_k(x, None).permute(0, 1, 3, 2))
        x = x + residual_k
        
        if self.res_project:
            residual_d = self.res_projection_d(x)
        else:
            residual_d = x
        x = self.ln_d(x)
        x = self.dropout_d(self.mlp_d(x, None))
        x = x + residual_d

        return x

    def forward_ln_last(self, x, mask):
        if self.res_project:
            residual_l = self.res_projection_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            residual_l = x
        x = self.mlp_l(x.permute(0, 2, 3, 1), None).permute(0, 3, 1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).bool(), 0.0) # Fill mask=True to 0.0
        x = self.dropout_l(x)
        x = x + residual_l
        x = self.ln_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if self.res_project:
            residual_k = self.res_projection_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        else:
            residual_k = x
        x = self.dropout_k(self.mlp_k(x.permute(0, 1, 3, 2), None).permute(0, 1, 3, 2))
        x = x + residual_k
        x = self.ln_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        
        if self.res_project:
            residual_d = self.res_projection_d(x)
        else:
            residual_d = x
        x = self.dropout_d(self.mlp_d(x, None))
        x = x + residual_d
        x = self.ln_d(x)

        return x


# d_in=[l,k,d], hiddens, outs = [[l,k,d], [l,k,d], ..., [l,k,d]] for n layers
class MLPEncoder(nn.Module):
    def __init__(self, activate, d_in, d_hiddens, d_outs, dropouts, bias, ln_first=False, res_project=[False, False, True]):
        super(MLPEncoder, self).__init__()
        assert len(d_hiddens)==len(d_outs)==len(res_project)
        self.layers_stack = nn.ModuleList([
            MLPsBlock(activate=activate, d_ins=d_in if i==0 else d_outs[i-1], d_hiddens=d_hiddens[i], d_outs=d_outs[i], dropouts=dropouts, bias=bias, ln_first=ln_first, res_project=res_project[i])
        for i in range(len(d_hiddens))])

    def forward(self, x, mask=None):
        for enc_layer in self.layers_stack:
            x = enc_layer(x, mask)
        return x


if __name__ == '__main__':
    from Utils import to_gpu, get_mask_from_sequence

    print('='*40, 'Testing Mask', '='*40)
    x = torch.randn(2, 3, 4)
    mask = get_mask_from_sequence(x, -1) # valid=False/0
    print(mask)
    print(mask.shape)

    x = to_gpu(torch.randn(2, 3, 5, 6))
    mask = to_gpu(torch.Tensor([
        [False, False, False],
        [False, False, True],
    ]))

    # print('='*40, 'Testing MLP', '='*40)
    # mlp = to_gpu(MLP('gelu', 6, 16, 26, bias=False))
    # y = mlp(x, mask)
    # print(y.shape)

    # print('='*40, 'Testing MLPsBlock', '='*40)
    # mlpsBlock = to_gpu(MLPsBlock(activate='gelu', d_ins=[3, 5, 6], d_hiddens=[13, 15, 16], d_outs=[23, 25, 26], bias=False, res_project=True, dropouts=[0.1, 0.2, 0.3], ln_first=False))
    # y = mlpsBlock(x, mask=None)
    # print(y.shape)

    print('='*40, 'Testing MLPEncoder', '='*40)
    x = to_gpu(torch.randn(2, 100, 3, 128))
    encoder = to_gpu(MLPEncoder(activate='gelu', d_in=[100, 3, 128], d_hiddens=[[100, 3, 128], [100, 3, 128], [50, 2, 64], [50, 2, 64]], d_outs=[[100, 3, 128], [50, 2, 64], [50, 2, 64], [10, 1, 32]], dropouts=[0.3,0.5,0.6], bias=False, ln_first=True, res_project=[True, True, True, True]))
    y = encoder(x, mask=None)
    print(y.shape)


