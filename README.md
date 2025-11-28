# Federated Learning Baselines

This repository provides implementations of commonly used **Federated Learning baselines**, forked from the repository by *CityChan* due to its well-structured **Client–Server** modular framework. Note that this repository is modified for personal use.

## Implemented Algorithms

* **FedAvg**
* **FedProx**
* **FedMD**
* **FedHKD**
* **FedProto**

---

## Supported Dataset

* **CIFAR-10**

  * Supports **Dirichlet-based non-IID partitioning** using the `--beta` parameter.


## ⚙ Parameters

| Argument          | Description                                                  | Default   |
| ----------------- | ------------------------------------------------------------ | --------- |
| `--dataset`       | Dataset name (`CIFAR10`, `SVHN`)                             | `CIFAR10` |
| `--batch_size`    | Local batch size                                             | `64`      |
| `--num_epochs`    | Number of **global rounds**                                  | `50`      |
| `--lr`            | Learning rate                                                | `0.001`   |
| `--lr_sh_rate`    | Epochs before LR decay                                       | `10`      |
| `--dropout_rate`  | Dropout per layer                                            | `0.2`     |
| `--clip_grad`     | Gradient clipping norm                                       | `1.0`     |
| `--num_users`     | Number of federated clients                                  | `10`      |
| `--sampling_rate` | Client sampling per round                                    | `1.1`     |
| `--local_ep`      | Local training epochs per client                             | `5`       |
| `--beta`          | Dirichlet partition concentration                            | `0.5`     |
| `--seed`          | Random seed                                                  | `0`       |
| `--std`           | Differential privacy noise std dev                           | `2.0`     |
| `--code_len`      | Latent vector dimension                                      | `32`      |
| `--alg`           | Algorithm (`FedAvg, FedProx, Moon, FedMD, Fedproto, FedHKD`) | —         |
| `--eval_only`     | Only evaluate test accuracy                                  | `False`   |
| `--part`          | Fraction of local data per client                            | —         |
| `--temp`          | Temperature for soft predictions                             | —         |
| `--lam`           | Weight of loss term 2                                        | —         |
| `--gamma`         | Weight of loss term 3                                        | —         |
| `--model`         | Model architecture (`CNN`, `resnet18`, `shufflenet`)         | —         |
| `--save_model`    | Save model checkpoints                                       | `False`   |

---

## Example Run

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset 'SVHN' \
  --batch_size 64 \
  --num_epochs 50 \
  --clip_grad 1.1 \
  --lr 0.001 \
  --num_clients 10 \
  --num_classes 10 \
  --sampling_rate 1 \
  --local_ep 3 \
  --beta 0.5 \
  --seed 0 \
  --code_len 50 \
  --alg 'FedAvg' \
  --part 0.1 \
  --model 'resnet18' \
  --temp 0.5
```

---

## Citation

If you use **FedHKD** or refer to concepts implemented in this repository, please cite:

```bibtex
@inproceedings{chen2023best,
  title={The Best of Both Worlds: Accurate Global and Personalized Models through Federated Learning with Data-Free Hyper-Knowledge Distillation},
  author={Chen, Huancheng and Vikalo, Haris and others},
  journal={arXiv preprint arXiv:2301.08968},
  year={2023}
}
```

---

## 📣 Acknowledgment

This repository is **forked from CityChan’s open-source implementation**, which provides a clean split between **Client** and **Server** logic for easier experimentation and extension in federated learning research.


