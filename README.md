# Data and Code for "DualContextGCN: Modeling differential Word Associations for Fake News Detection 


## Data
Our work is based on the `PolitiFact` and `GossipCop` datasets from the [FakeNewsNet benchmark](https://github.com/KaiDMML/FakeNewsNet), and the `LUN` dataset from [(Rashkin et al., 2017)](https://aclanthology.org/D17-1317.pdf). 

We provide the data files utilized for training and evaluating DcgcnR under `data/`. 

**Original Unaltered Training / Test Articles**

The `.pkl` files under `data/news_articles/` contain the unaltered news article texts. 



## Run DcgcnR
 

Start training with the following command:

python src/DcgcnR.py --dataset_name politifact
[--dataset_name]: politifact / gossipcop / lun


