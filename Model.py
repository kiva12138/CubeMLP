import torch
import torch.nn as nn
import torch.nn.functional as F
from MLPProcess import MLPEncoder
from Utils import mean_temporal, get_mask_from_sequence, to_cpu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig, BertTokenizer

def get_output_dim(features_compose_t, features_compose_k, d_out, t_out, k_out):
    if features_compose_t in ['mean', 'sum']:
        classify_dim = d_out
    elif features_compose_t == 'cat':
        classify_dim = d_out * t_out
    else:
        raise NotImplementedError

    if features_compose_k in ['mean', 'sum']:
        classify_dim = classify_dim
    elif features_compose_k == 'cat':
        classify_dim = classify_dim * k_out
    else:
        raise NotImplementedError
    return classify_dim


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        d_t, d_a, d_v, d_common, encoders = opt.d_t, opt.d_a, opt.d_v, opt.d_common, opt.encoders
        features_compose_t, features_compose_k, num_class = opt.features_compose_t, opt.features_compose_k, opt.num_class
        self.time_len = opt.time_len

        self.d_t, self.d_a, self.d_v, self.d_common = d_t, d_a, d_v, d_common
        self.encoders = encoders
        assert self.encoders in ['lstm', 'gru', 'conv']
        self.features_compose_t, self.features_compose_k = features_compose_t, features_compose_k
        assert self.features_compose_t in ['mean', 'cat', 'sum']
        assert self.features_compose_k in ['mean', 'cat', 'sum']

        # Bert Extractor
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)

        # Extractors
        if self.encoders == 'conv':
            self.conv_a = nn.Conv1d(in_channels=d_a, out_channels=d_common, kernel_size=3, stride=1, padding=1)
            self.conv_v = nn.Conv1d(in_channels=d_v, out_channels=d_common, kernel_size=3, stride=1, padding=1)
        elif self.encoders == 'lstm':
            self.rnn_v = nn.LSTM(d_v, d_common, 1, bidirectional=True, batch_first=True)
            self.rnn_a = nn.LSTM(d_a, d_common, 1, bidirectional=True, batch_first=True)
        elif self.encoders == 'gru':
            self.rnn_v = nn.GRU(d_v, d_common, 2, bidirectional=True, batch_first=True)
            self.rnn_a = nn.GRU(d_a, d_common, 2, bidirectional=True, batch_first=True)
        else:
            raise NotImplementedError

        # LayerNormalize & Dropout
        self.ln_a, self.ln_v = nn.LayerNorm(d_common, eps=1e-6), nn.LayerNorm(d_common, eps=1e-6)
        self.dropout_t, self.dropout_a, self.dropout_v = nn.Dropout(opt.dropout[0]), nn.Dropout(opt.dropout[1]), nn.Dropout(opt.dropout[2])

        # Projector
        self.W_t = nn.Linear(d_t, d_common, bias=False)

        # MLPsEncoder
        self.mlp_encoder = MLPEncoder(activate=opt.activate, d_in=[opt.time_len, 3, d_common], d_hiddens=opt.d_hiddens, d_outs=opt.d_outs, dropouts=opt.dropout_mlp, bias=opt.bias, ln_first=opt.ln_first, res_project=opt.res_project)

        # Define the Classifier
        classify_dim = get_output_dim(self.features_compose_t, self.features_compose_k, opt.d_outs[-1][2], opt.d_outs[-1][0], opt.d_outs[-1][1])
        if classify_dim <= 128:
            self.classifier = nn.Sequential(
                nn.Linear(classify_dim, num_class)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(classify_dim, 128),
                nn.ReLU(),
                nn.Dropout(opt.dropout[3]),
                nn.Linear(128, num_class),
            )

    # tav:[bs, len, d]; mask:[bs, len]
    def forward(self, bert_sentences, bert_sentence_types, bert_sentence_att_mask, a, v, return_features=False, debug=False):
        l_av = a.shape[1]

        # Extract Bert features
        t = self.bertmodel(input_ids=bert_sentences, attention_mask=bert_sentence_att_mask, token_type_ids=bert_sentence_types)[0]
        if debug:
            print('Origin:', t.shape, a.shape, v.shape)
        mask_t = bert_sentence_att_mask # Valid = 1
        t = self.W_t(t)

        # Pad audio & video
        length_padded = t.shape[1]
        pad_before = int((length_padded - l_av)/2)
        pad_after = length_padded - l_av - pad_before
        a = F.pad(a, (0, 0, pad_before, pad_after, 0, 0), "constant", 0)
        v = F.pad(v, (0, 0, pad_before, pad_after, 0, 0), "constant", 0)
        a_fill_pos = (get_mask_from_sequence(a, dim=-1).int() * mask_t).bool()
        v_fill_pos = (get_mask_from_sequence(v, dim=-1).int() * mask_t).bool()
        a, v = a.masked_fill(a_fill_pos.unsqueeze(-1), 1e-6), v.masked_fill(v_fill_pos.unsqueeze(-1), 1e-6)
        if debug:
            print('Padded:', t.shape, a.shape, v.shape)
        mask_a = get_mask_from_sequence(a, dim=-1) # Valid = False
        mask_v = get_mask_from_sequence(v, dim=-1) # Valid = False
        if debug:
            print('Padded mask:', mask_t, mask_a, mask_v, sep='\n')
        lengths = to_cpu(bert_sentence_att_mask).sum(dim=1)
        l_av_padded = a.shape[1]

        # Extract features
        if self.encoders == 'conv':
            a, v = self.conv_a(a.transpose(1, 2)).transpose(1, 2), self.conv_v(v.transpose(1, 2)).transpose(1, 2)
            a, v = F.relu(self.ln_a(a)), F.relu(self.ln_v(v))
        elif self.encoders in ['lstm', 'gru']:
            a = pack_padded_sequence(a, lengths, batch_first=True, enforce_sorted=False)
            v = pack_padded_sequence(v, lengths, batch_first=True, enforce_sorted=False)
            self.rnn_a.flatten_parameters()
            self.rnn_v.flatten_parameters()
            (packed_a, a_out), (packed_v, v_out) = self.rnn_a(a), self.rnn_v(v)
            a, _ = pad_packed_sequence(packed_a, batch_first=True, total_length=l_av_padded)
            v, _ = pad_packed_sequence(packed_v, batch_first=True, total_length=l_av_padded)
            if debug:
                print('After RNN', a.shape, v.shape)
            if self.encoders == 'lstm':
                a_out, v_out =a_out[0], v_out[0]
            a = torch.stack(torch.split(a, self.d_common, dim=-1), -1).sum(-1)
            v = torch.stack(torch.split(v, self.d_common, dim=-1), -1).sum(-1)
            if debug:
                print('After Union', a.shape, v.shape)
            # a, v = F.relu(a), F.relu(v)
            a, v = F.relu(self.ln_a(a)), F.relu(self.ln_v(v))
            # t = F.relu(self.ln_t(t))
        else:
            raise NotImplementedError
        t, a, v = self.dropout_t(t), self.dropout_a(a), self.dropout_v(v)

        if debug:
            print('After Extracted', t.shape, a.shape, v.shape)

        # Padding temporal axis
        t = F.pad(t, (0, 0, 0, self.time_len-l_av_padded, 0, 0), "constant", 0)
        a = F.pad(a, (0, 0, 0, self.time_len-l_av_padded, 0, 0), "constant", 0)
        v = F.pad(v, (0, 0, 0, self.time_len-l_av_padded, 0, 0), "constant", 0)

        # Union 
        # x = torch.stack([t, a, a], dim=2)
        x = torch.stack([t, a, v], dim=2)

        if debug:
            print('After Padded and Unioned on Temporal', t.shape, a.shape, v.shape, x.shape)

        # Encoding
        x = self.mlp_encoder(x, mask=None)
        features = x
        if debug:
            print('After Encoder', x.shape)

        # Compose [bs, t, k, d]
        if self.features_compose_t == 'mean':
            fused_features = x.mean(dim=1)
        elif self.features_compose_t == 'sum':
            fused_features = x.mean(dim=1)
        elif self.features_compose_t == 'cat':
            fused_features = torch.cat(torch.split(x, 1, dim=1), dim=-1).squeeze(1)
        else:
            raise NotImplementedError

        if self.features_compose_k == 'mean':
            fused_features = fused_features.mean(dim=1)
        elif self.features_compose_k == 'sum':
            fused_features = fused_features.mean(dim=1)
        elif self.features_compose_k == 'cat':
            fused_features = torch.cat(torch.split(fused_features, 1, dim=1), dim=-1).squeeze(1)
        else:
            raise NotImplementedError

        if debug:
            print('Fused', fused_features.shape)
            
        # Predictions
        output = self.classifier(fused_features)
        if return_features:
            return [output, features]
        else:
            return [output]


if __name__ == '__main__':
    from Utils import to_gpu

    print('='*40, 'Testing Model', '='*40)
    from types import SimpleNamespace    
    opts = SimpleNamespace(d_t=768, d_a=74, d_v=35, d_common=128, encoders='gru', features_compose_t='cat', features_compose_k='mean', num_class=7, 
            activate='gelu', time_len=50, d_hiddens=[[20, 3, 128],[10, 2, 64],[5, 2, 32]], d_outs=[[20, 3, 128],[10, 2, 64],[5, 1, 32]],
            dropout_mlp=[0.3,0.4,0.5], dropout=[0.3,0.4,0.5,0.6], bias=False, ln_first=False, res_project=[True,True,True]
            )
    print(opts)

    t = [
        ["And", "the", "very", "very", "last", "one", "one"],
        ["And", "the", "very", "very", "last", "one"],
    ]
    a = torch.randn(2, 7, 74).cuda()
    v = torch.randn(2, 7, 35).cuda()
    mask_a = torch.BoolTensor([[False, False, False, False, False, False, False],
        [False, False, False, False, False, False, True]]).cuda()
    mask_v = torch.BoolTensor([[False, False, False, False, False, False, False],
        [False, False, False, False, False, False, True]]).cuda()
    a = a.masked_fill(mask_a.unsqueeze(-1), 0)
    v = v.masked_fill(mask_v.unsqueeze(-1), 0)
    # print(get_mask_from_sequence(a, dim=-1))

    model = Model(opts).cuda()
    
    from transformers import BertTokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sentences = [" ".join(sample) for sample in t]
    bert_details = bert_tokenizer.batch_encode_plus(sentences, add_special_tokens=True, padding=True)
    bert_sentences = to_gpu(torch.LongTensor(bert_details["input_ids"]))
    bert_sentence_types = to_gpu(torch.LongTensor(bert_details["token_type_ids"]))
    bert_sentence_att_mask = to_gpu(torch.LongTensor(bert_details["attention_mask"]))

    result = model(bert_sentences, bert_sentence_types, bert_sentence_att_mask, a, v, return_features=True, debug=True)
    print([r.shape for r in result])