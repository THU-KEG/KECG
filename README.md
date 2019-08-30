# KECG
Source code and datasets for EMNLP 2019 paper "[Semi-supervised Entity Alignment via Joint Knowledge Embedding Model and Cross-graph Model](https://xlore.org)".


## Reqirements
- Python3 (tested on 3.6.6)
- Pytorch (tested on 0.4.1)


## Code
This implementation includes KECG, KECG(w\o K), and KECG(w\o NNS).
Example of running KECG on DBP15K(ZH-EN)
```
CUDA_VISIBLE_DEVICES=0 python3.6 run.py --file_dir data/DBP15K/zh_en --rate 0.3 --lr 0.005 --epochs 1000
```

Example of running KECG(w\o K)
```
CUDA_VISIBLE_DEVICES=0 python3.6 run.py --file_dir data/DBP15K/zh_en --rate 0.3 --lr 0.001 --epochs 500 --wo_K
```

Example of running KECG(w\o NNS)
```
CUDA_VISIBLE_DEVICES=0 python3.6 run.py --file_dir data/DBP15K/zh_en --rate 0.3 --lr 0.05 --epochs 1000 --wo_NNS
```


## Dataset
The used datasets DBP15K and DWY100K are from subfolder named "mapping" of [BootEA](https://github.com/nju-websoft/BootEA) and [JAPE](https://github.com/nju-websoft/JAPE). (But need to combine "ref_ent_ids" and "sup_ent_ids" into a single file named "ill_ent_ids" before running KECG.) Here, you can directly unpack the data file
```
unzip data.zip
```


## Acknowledgement
We refer to some codes of these repos: [pyGAT](https://github.com/Diego999/pyGAT), [BootEA](https://github.com/nju-websoft/BootEA), [GCN-Align](https://github.com/1049451037/GCN-Align). Appreciate for their great contributions!


## Cite
If you use the code, please cite this paper:
- Chengjiang Li, Yixin Cao, Lei Hou, Jiaxin Shi, Juanzi Li and Tat-Seng Chua. Semi-supervised Entity Alignment via Joint Knowledge Embedding Model and Cross-graph Model. In EMNLP 2019.
