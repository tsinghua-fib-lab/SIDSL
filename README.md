**Official implementation of our CIKM 2025 work "Structure-prior Informed Diffusion Model for Graph Source Localization with Limited Data"**

## run code

run command:
> python main_newdiff.py --dataset [dataset]

Available datasets:(as in datasets\synthesis)
- android_10_30, christianity_10_30, jazz_IC50, jazz_LT50, jazz_SIS50, netscience_IC50, netscience_LT50, netscience_SIS50, power_IC50, power_LT50, power_SIS50,

- Some large datasets files are not included in the repository. Find them in the following link:
https://drive.google.com/drive/folders/1Ga7qTNscLug49o69E18AAyiXvRKDjjY5?usp=sharing

Tunable params can be checked in the following description.

## load saved model to run code

run command:
> python main_newdiff.py --dataset [dataset] --save_dict [path to saved model]

e.g.
> python main_newdiff.py --dataset jazz_IC50 --state_dict ./saved_diffusers/jazz_IC50_best.pt


## Requirements

Please check requirements.txt for the required packages.

### Main Parameters

- `dataset`: Name of the dataset to use
- `gnn_type`: Type of GNN architecture (default: gcn)
- `noise_emb_dim`: Dimension of noise embedding (default: 128)
- `hidden_dim`: Hidden layer dimension (default: 128)
- `num_layers`: Number of GNN layers (default: 5)
- `activation`: Activation function (default: prelu)
- `mlp_layers`: Number of MLP layers (default: 4)
- `num_advisors`: Number of advisor models (default: 1)

### Optional Parameters

- `feat_drop`: Feature dropout rate (default: 0.0)
- `attn_drop`: Attention dropout rate (default: 0.0)
- `negative_slope`: Negative slope for LeakyReLU (default: 0.2)
- `residual`: Whether to use residual connections (default: True)
- `scheduler`: Whether to use learning rate scheduler (default: True)
- `train_cond`: Whether to train the condition module (default: False)

