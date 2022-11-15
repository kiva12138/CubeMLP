# CubeMLP
The implementation of CubeMLP


To run the codes:
```
python Train.py --dataset mosi_SDK --batch_size 128 --features_compose_t mean --features_compose_k cat --d_hiddens 50-3-128=10-3-32  --d_outs 50-3-128=10-3-32 --res_project 1-1 --bias --ln_first  -dropout_mlp 0.1-0.1-0.1 --dropout 0.1-0.1-0.1-0.1 --bert_freeze part --bert_lr_rate 0.01  --learning_rate 4e-3
```
