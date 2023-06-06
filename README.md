# Ternary_Binary_Transformer


This repository contains the training code of TBT introduced in our work: "[Binary and Ternary Natural Language Generation](https://arxiv.org/abs/2306.01841)", published in ACL 2023.

We approach the problem with a mix of statistics-based quantization for the weights and elastic quantization of the activations and demonstrate the first ternary and binary transformer models on the downstream tasks of summarization and machine translation. 

<div align=center>
<img width=60% src="https://github.com/facebookresearch/Ternary_Binary_Transformer/blob/main/overview.png"/>
</div>


## Citation

If you find our code useful for your research, please consider citing:

    @article{liu2023binary,
    title={Binary and Ternary Natural Language Generation},
    author={Liu, Zechun and Oguz, Barlas and Pappu, Aasish and Shi, Yangyang and Krishnamoorthi, Raghuraman},
    booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics}
    year={2023}
    }

Our previous papers related to binarizing BERT model:
* BiT: Robustly Binarized Multi-distilled Transformer (NeurIPS 2022) \[[code](https://github.com/facebookresearch/bit)\] \[[paper](https://arxiv.org/pdf/2205.13016.pdf)\]

## Run

### 1. Requirements:
* python 3.9.12, pytorch 1.12.1

### 2. Pretrained models:
* Download pretrained models from hugging face model zoo.
  | Dataset | Finetuned full-precision model |
  | --- | --- |
  | XSUM | [bart-base-xsum](https://huggingface.co/Aktsvigun/bart-base_xsum_42) |
  | CNN/DailyMail | [bart-base-cnn](https://huggingface.co/ainize/bart-base-cnn) |
  
### 3. Steps to run:
* For XSUM benchmark, `bash scrips/run_xsum.sh $w_bit $a_bit $lr` .
* For CNN/DailyMail benchmark, `bash scrips/run_cnn.sh $w_bit $a_bit $lr` .
* Learning rate for each model:

|  |  XSUM | CNN/DailyMail |  
| --- | --- | --- |
| W2A8 | 3e-4 | 1e-4 |
| W2A2 | 3.5e-4 | 7e-4 |
| W1A8 | 2.5e-4 | 1.5e-4 |
| W1A1 | 5e-4 | 5e-4 |

## Models
 
|  |  |  |  |  |  XSUM |  |  | CNN|  |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
|  | **#Bits** | **Size (M)** | **FLOPs (G)** | **R1** | **R2** | **RL** | **R1** | **R2** | **RL** | 
|BART | 32-32-32 | 532.0 | 1x | 43.84 | 20.79 | 35.71 | 44.90 | 22.25 | 42.09 |
|QuantBart| 8 - 8 - 8 | 138.1 | -- | 40.25 | 17.78 | 32.70 | -- | -- | -- |
|DQ-BART| 8 - 8 - 8 | 138.1 | -- | 42.51 | 19.61 | 34.61 | 44.66 | 21.92 | 41.86 |
|*Ternary* | | | | | | | | | |
|Baseline (TWN) | 2 - 2 - 8 | 39.6 | 0.25x | 39.99 | 17.13 | 31.99 | 42.99 | 20.05 | 40.18|
|QuantBart| 2 - 2 - 8 | 39.6 | 0.25x | 39.15 | 16.72 | 31.72 | -- | -- | -- |
|DQ-BART| 2 - 2 - 8 | 39.6 | 0.25x | 40.06 | 17.34 | 32.46 | 42.94 | 20.07 | 40.13 |
|**TBT** | 2 - 2 - 8 | 39.6 | 0.25x | [**42.40**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Ej9CwkrXVVBDmecbj94IoloBA768OTzQSnQyc7U_2iabzA?e=OLtJt9) | [**19.54**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Ej9CwkrXVVBDmecbj94IoloBA768OTzQSnQyc7U_2iabzA?e=OLtJt9) | [**34.51**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Ej9CwkrXVVBDmecbj94IoloBA768OTzQSnQyc7U_2iabzA?e=OLtJt9) | [**43.46**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Eq0HeMLo0RNIntzXsfHY9gIBZHmVCNL1L-AWvT54IxLn7A?e=Ccr8U5) | [**20.52**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Eq0HeMLo0RNIntzXsfHY9gIBZHmVCNL1L-AWvT54IxLn7A?e=Ccr8U5) | [**40.58**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Eq0HeMLo0RNIntzXsfHY9gIBZHmVCNL1L-AWvT54IxLn7A?e=Ccr8U5) |
|Baseline (TWN) | 2 - 2 - 2 | 39.6 | 0.0625x | 12.80 | 1.21 | 11.4 | 12.92 | 0.32 | 12.42|
|**TBT** | 2 - 2 - 2 | 39.6 | 0.0625x | [**36.21**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Ery8OufgDpRIjFL2P9NBxukBHCJ34Tkth7DfhLu5BiHkXA?e=5KrmKE) | [**14.38**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Ery8OufgDpRIjFL2P9NBxukBHCJ34Tkth7DfhLu5BiHkXA?e=5KrmKE) | [**29.07**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Ery8OufgDpRIjFL2P9NBxukBHCJ34Tkth7DfhLu5BiHkXA?e=5KrmKE) | [**41.03**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/EryiIhiloWFAjdDkiqRBeYwBm7l-DQxlXViu8Sm_FAzCSg?e=UEHUvB) | [**18.18**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/EryiIhiloWFAjdDkiqRBeYwBm7l-DQxlXViu8Sm_FAzCSg?e=UEHUvB) | [**38.30**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/EryiIhiloWFAjdDkiqRBeYwBm7l-DQxlXViu8Sm_FAzCSg?e=UEHUvB) |
|*Binary* | | | | | | | | | |
|Baseline (BWN) | 1 - 1 - 8 | 23.2 | 0.125x | 1.90 | 0.01 | 1.78 | 2.78 | 0.08 | 2.48|
|**TBT** | 1 - 1 - 8 | 23.2 | 0.125x | [**40.96**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/ErpKKcUo_RlDpIHqqQXMTU8BamD85JA0Ebtg4J5oFhTYJA?e=wtNL2b) | [**18.37**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/ErpKKcUo_RlDpIHqqQXMTU8BamD85JA0Ebtg4J5oFhTYJA?e=wtNL2b) | [**33.30**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/ErpKKcUo_RlDpIHqqQXMTU8BamD85JA0Ebtg4J5oFhTYJA?e=wtNL2b) | [**42.66**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/EriYUYGE2cZAqgSM0YKY7vcBs0PmVvIyqtdnZKWXlANztQ?e=Q3aOWx) | [**19.72**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/EriYUYGE2cZAqgSM0YKY7vcBs0PmVvIyqtdnZKWXlANztQ?e=Q3aOWx) | [**39.80**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/EriYUYGE2cZAqgSM0YKY7vcBs0PmVvIyqtdnZKWXlANztQ?e=Q3aOWx) |
|Baseline (BWN)| 1 - 1 - 1 | 23.2 | 0.0156x | 1.90 | 0.01 | 1.78 | 2.78 | 0.08 | 2.48|
|**TBT** | 1 - 1 - 1 | 23.2 | 0.0156x | [**31.68**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/ElDbaSZyLx1ItZSp2O6rzocBHsjHf_IRMT9kvWk3QIZOkQ?e=WOUFza) | [**11.19**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/ElDbaSZyLx1ItZSp2O6rzocBHsjHf_IRMT9kvWk3QIZOkQ?e=WOUFza) | [**25.29**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/ElDbaSZyLx1ItZSp2O6rzocBHsjHf_IRMT9kvWk3QIZOkQ?e=WOUFza) | [**35.56**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Epvrka2zQfJNvqXfevDA3KkBALQr_0571d-iFD8d6ezyug?e=PsgzBM) | [**11.71**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Epvrka2zQfJNvqXfevDA3KkBALQr_0571d-iFD8d6ezyug?e=PsgzBM) | [**33.23**](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliubq_connect_ust_hk/Epvrka2zQfJNvqXfevDA3KkBALQr_0571d-iFD8d6ezyug?e=PsgzBM) |


## Acknowledgement

The original code is borrowed from [DQ-BART](https://github.com/amazon-science/dq-bart).

## Contact

Zechun Liu, Reality Labs, Meta Inc (liuzechun0216 at gmail.com)

## License
BiT is CC-BY-NC 4.0 licensed as of now.

