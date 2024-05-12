# AntiFraud
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/fraud-detection-on-amazon-fraud)](https://paperswithcode.com/sota/fraud-detection-on-amazon-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/node-classification-on-amazon-fraud)](https://paperswithcode.com/sota/node-classification-on-amazon-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/fraud-detection-on-yelp-fraud)](https://paperswithcode.com/sota/fraud-detection-on-yelp-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/node-classification-on-yelpchi)](https://paperswithcode.com/sota/node-classification-on-yelpchi?p=semi-supervised-credit-card-fraud-detection)

A Financial Fraud Detection Framework.

Source codes implementation of papers:
- `MCNN`: Credit card fraud detection using convolutional neural networks, in ICONIP 2016. 
- `STAN`: Spatio-temporal attention-based neural network for credit card fraud detection, in AAAI2020
- `STAGN`: Graph Neural Network for Fraud Detection via Spatial-temporal Attention, in TKDE2020
- `GTAN`: Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation, in AAAI2023
- `RGTAN`: Enhancing Attribute-driven Fraud Detection with Risk-aware Graph Representation, 



## Usage

### Data processing
1. Run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets; 
2. Run `python feature_engineering/data_process.py
`
to pre-process all datasets needed in this repo.

### Training & Evalutaion
<!-- 
To use fraud detection baselines including GBDT, LSTM, etc., simply run

```
python main.py --method LSTM
python main.py  --method GBDT
```
You may change relevant configurations in `config/base_cfg.yaml`. -->

To test implementations of `MCNN`, `STAN` and `STAGN`, run
```
python main.py --method mcnn
python main.py --method stan
python main.py --method stagn
```
Configuration files can be found in `config/mcnn_cfg.yaml`, `config/stan_cfg.yaml` and `config/stagn_cfg.yaml`, respectively.

Models in `GTAN` and `RGTAN` can be run via:
```
python main.py --method gtan
python main.py --method rgtan
```
For specification of hyperparameters, please refer to `config/gtan_cfg.yaml` and `config/rgtan_cfg.yaml`.



### Data Description

There are three datasets, YelpChi, Amazon and S-FFSD, utilized for model experiments in this repository.

<!-- YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data) or [dgl.data.FraudDataset](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset).

Put them in `/data` directory and run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets. -->

YelpChi and Amazon datasets are from [CARE-GNN](https://dl.acm.org/doi/abs/10.1145/3340531.3411903), whose original source data can be found in [this repository](https://github.com/YingtongDou/CARE-GNN/tree/master/data).

S-FFSD is a simulated & small version of finacial fraud semi-supervised dataset. Description of S-FFSD are listed as follows:
|Name|Type|Range|Note|
|--|--|--|--|
|Time|np.int32|from $\mathbf{0}$ to $\mathbf{N}$|$\mathbf{N}$ denotes the number of trasactions.  |
|Source|string|from $\mathbf{S_0}$ to $\mathbf{S}_{ns}$|$ns$ denotes the number of transaction senders.|
|Target|string|from $\mathbf{T_0}$  to $\mathbf{T}_{nt}$ | $nt$ denotes the number of transaction reveicers.|
|Amount|np.float32|from **0.00** to **np.inf**|The amount of each transaction. |
|Location|string|from $\mathbf{L_0}$  to $\mathbf{L}_{nl}$ |$nl$ denotes the number of transacation locations.|
|Type|string|from $\mathbf{TP_0}$ to $\mathbf{TP}_{np}$|$np$ denotes the number of different transaction types. |
|Labels|np.int32|from **0** to **2**|**2** denotes **unlabeled**||


> We are looking for interesting public datasets! If you have any suggestions, please let us know!

## Test Result
The performance of five models tested on three datasets are listed as follows:
| |YelpChi| | |Amazon| | |S-FFSD| | |
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
| |AUC|F1|AP|AUC|F1|AP|AUC|F1|AP|
|MCNN||- | -| -| -| -|0.7129|0.6861|0.3309|
|STAN|- |- | -| -| -| -|0.7446|0.6791|0.3395|
|STAGN|- |- | -| -| -| -|0.7659|0.6852|0.3599|
|GTAN|0.9241|0.7988|0.7513|0.9630|0.9213|0.8838|0.8286|0.7336|0.6585|
|RGTAN|0.9498|0.8492|0.8241|0.9750|0.9200|0.8926|0.8461|0.7513|0.6939|

> `MCNN`, `STAN` and `STAGN` are presently not applicable to YelpChi and Amazon datasets.

## Repo Structure
The repository is organized as follows:
- `models/`: the pre-trained models for each method. The readers could either train the models by themselves or directly use our pre-trained models;
- `data/`: dataset files;
- `config/`: configuration files for different models;
- `feature_engineering/`: data processing;
- `methods/`: implementations of models;
- `main.py`: organize all models;
- `requirements.txt`: package dependencies;

    
## Requirements
```
python           3.7
scikit-learn     1.0.2
pandas           1.3.5
numpy            1.21.6
networkx         2.6.3
scipy            1.7.3
torch            1.12.1+cu113
dgl-cu113        0.8.1
```

### Contributors :
<a href="https://github.com/AI4Risk/antifraud/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Risk/antifraud" />
</a>

### Citing

If you find *Antifraud* is useful for your research, please consider citing the following papers:

    @inproceedings{Xiang2023SemiSupervisedCC,
        title={Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation},
        author={Sheng Xiang and Mingzhi Zhu and Dawei Cheng and Enxia Li and Ruihui Zhao and Yi Ouyang and Ling Chen and Yefeng Zheng},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2023}
    }
    @article{cheng2020graph,
        title={Graph Neural Network for Fraud Detection via Spatial-temporal Attention},
        author={Cheng, Dawei and Wang, Xiaoyang and Zhang, Ying and Zhang, Liqing},
        journal={IEEE Transactions on Knowledge and Data Engineering},
        year={2020},
        publisher={IEEE}
    }
    @inproceedings{cheng2020spatio,
        title={Spatio-temporal attention-based neural network for credit card fraud detection},
        author={Cheng, Dawei and Xiang, Sheng and Shang, Chencheng and Zhang, Yiyi and Yang, Fangzhou and Zhang, Liqing},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={34},
        number={01},
        pages={362--369},
        year={2020}
    }
    @inproceedings{fu2016credit,
        title={Credit card fraud detection using convolutional neural networks},
        author={Fu, Kang and Cheng, Dawei and Tu, Yi and Zhang, Liqing},
        booktitle={International Conference on Neural Information Processing},
        pages={483--490},
        year={2016},
        organization={Springer}
    }

##
conda create -n antifraud python=3.7
conda config --add channels conda-forge
conda config --add channels dglteam
conda install --file requirements.txt
conda install pytorch==1.12.1 cpuonly -c pytorch
conda install anaconda::pyyaml
conda install catboost
conda install xgboost
conda install conda-forge::py-xgboost-cpu=1.6.0



```
python main.py --method mcnn

-- one experiment takes minutes
-- mcnn results: auc: 0.7214, F1: 0.6865, AP: 0.3344
-- mcnn results: auc: 0.7544, F1: 0.6681, AP: 0.3381
-- mcnn results: auc: 0.7254, F1: 0.6999, AP: 0.3484
-- mcnn results: auc: 0.7438, F1: 0.6430, AP: 0.3198
-- mcnn results: auc: 0.7396, F1: 0.6324, AP: 0.3130
-- mcnn results: auc: 0.7497, F1: 0.6797, AP: 0.3426
-- mcnn results: auc: 0.7285, F1: 0.6931, AP: 0.3429
-- mcnn results: auc: 0.7012, F1: 0.6857, AP: 0.3271
```



```
python main.py --method stan_2d

-- one experiment takes minutes
-- stan_2d results: auc: 0.7270, F1: 0.7251, AP: 0.3800
-- stan_2d results: auc: 0.7602, F1: 0.7164, AP: 0.3774
-- stan_2d results: auc: 0.7413, F1: 0.6312, AP: 0.3136
-- stan_2d results: auc: 0.6959, F1: 0.7315, AP: 0.4135
-- stan_2d results: auc: 0.6928, F1: 0.7261, AP: 0.4018
```


```
python main.py --method stan

-- one experiment takes about one hour
-- stan results: auc: 0.7121, F1: 0.7186, AP: 0.3710
-- stan results: auc: 0.7548, F1: 0.7106, AP: 0.3696
-- stan results: auc: 0.7403, F1: 0.6788, AP: 0.3372
-- stan results: auc: 0.7074, F1: 0.6502, AP: 0.3014
```



```
python main.py --method stagn

-- one experiment takes about minutes
-- stagn results: auc: 0.5794, F1: 0.5550, AP: 0.2177
-- stagn results: auc: 0.6862, F1: 0.5302, AP: 0.2622
-- stagn results: auc: 0.7477, F1: 0.6608, AP: 0.3338
-- stagn results: auc: 0.6484, F1: 0.4681, AP: 0.2323
-- stagn results: auc: 0.6852, F1: 0.6365, AP: 0.2901
-- stagn results: auc: 0.6968, F1: 0.6033, AP: 0.2688
-- stagn results: auc: 0.5929, F1: 0.5763, AP: 0.2234
-- stagn results: auc: 0.6541, F1: 0.4735, AP: 0.2336
```



```
python main.py --method gtan

-- one experiment takes about one hour
-- gtan results: AUC: 0.8564, f1: 0.7277, AP: 0.6446
-- gtan results: AUC: 0.8435, f1: 0.7447, AP: 0.6788
-- gtan results: AUC: 0.8407, f1: 0.7248, AP: 0.6362
```



```
python main.py --method rgtan
```


```
python main.py --method cb

-- one experiment takes seconds
-- cb results: auc: 0.7063, F1: 0.7513, AP: 0.4630 (10 times identical results)
```


```
python main.py --method xgb

-- one experiment takes seconds
-- xgb results: auc: 0.7184, F1: 0.7586, AP: 0.4631  (10 times identical results)
```
