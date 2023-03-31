# LightGCL
This is the PyTorch implementation for LightGCL proposed in the paper [**LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation**](https://openreview.net/forum?id=FKXVK9dyMM), *International Conference on Learning Representation*, 2023.

<br>
<p align='center'>
<img src="https://user-images.githubusercontent.com/60952950/219573564-64d5e9cc-6dbc-4cc9-b115-95fb6d46f1a7.png"  width="600" height="300"><br>
<i> Fig: Model Structure of LightGCL </i>
</p>


### 1. Note on datasets and directories
Due to the large size of datasets *ML-10M*, *Amazon* and *Tmall*, we have compressed them into zip files. Please unzip them before running the model on these datasets. For *Yelp* and *Gowalla*, keeping the current directory structure is fine.

Before running the codes, please ensure that two directories `log/` and `saved_model/` are created under the root directory. They are used to store the training results and the saved model and optimizer states.

### 2. Running environment

We develope our codes in the following environment:

```
Python version 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0
```

### 3. How to run the codes

* Yelp
```
python main.py --data yelp
```

* Gowalla

```
python main.py --data gowalla --lambda2 0
```

* ML-10M
```
python main.py --data ml10m --temp 0.5
```

* Tmall

```
python main.py --data tmall --gnn_layer 1
```

* Amazon

```
python main.py --data amazon --gnn_layer 1 --lambda2 0 --temp 0.1
```

### 4. Some configurable arguments

* `--cuda` specifies which GPU to run on if there are more than one.
* `--data` selects the dataset to use.
* `--lambda1` specifies $\lambda_1$, the regularization weight for CL loss.
* `--lambda2` is $\lambda_2$, the L2 regularization weight.
* `--temp` specifies $\tau$, the temperature in CL loss.
* `--dropout` is the edge dropout rate.
* `--q` decides the rank q for SVD.

### 5. On the complexity of LightGCL

We notice that many readers are confused about the complexity of performing graph convolution on the SVD-reconstructed view, arguing that the complexity should be O(2IJLd) since the SVD-reconstructed view is fully-connected. In fact, this issue has been clearly explained in the **Appendix D.3** in our paper. We also answered a Github issue about it (<a href='https://github.com/HKUDS/LightGCL/issues/3'>issue #3</a>). We hereby clarify again:

It is correct that the reconstructed graph is fully connected. However, please note that the reconstructed graph is actually the **product of three low dimension matrices U,S,V'**, whose dimensions are I×q, q×q, q×J, respectively (where q is as small as 5). So we don't really need to compute the reconstructed graph, but just store the three low-dimension matrices. And by doing the matrix multiplication in the following order: US [pre-calculated, complexity not counted into training], V'E [complexity is O(qJd)], and then (US)(V'E) [complexity is O(qId)], we never need to construct that large matrix, and the complexity is proportional to **(I+J)** instead of **(IJ)**.

### 6. Citing our paper
Please kindly cite our paper if you find this paper and the codes helpful.
```
@inproceedings{caisimple,
  title={LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation},
  author={Cai, Xuheng and Huang, Chao and Xia, Lianghao and Ren, Xubin},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```
