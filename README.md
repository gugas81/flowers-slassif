# Learning Flowers images datasets without Labels


This repo contains the Pytorch implementation based on papers:
> [**SCAN: Learning to Classify Images without Labels**](https://arxiv.org/pdf/2005.12320.pdf)
>
> [Wouter Van Gansbeke](https://twitter.com/WGansbeke), [Simon Vandenhende](https://twitter.com/svandenh1), [Stamatios Georgoulis](https://twitter.com/stam_g), Marc Proesmans and Luc Van Gool.
link to its repo [code](https://github.com/wvangansbeke/Unsupervised-Classification)
 
and attention mechanism from paper
> @inproceedings{kong2022efficient,
  title={Efficient Classification of Very Large Images with Tiny Objects},
  author={Kong, Fanjie and Henao, Ricardo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2384--2394},
  year={2022}
}
> 
> [Code repo](https://github.com/timqqt/pytorch-zoom-in-network)


#My changes
 1. Usage public available [Kaggle flowers dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) for training without labels(!!!!)
 2. Bug fixing/small refactoring of paper repo
 3. Small changes augementation in tretext and Semantic clustering tasks
 4. Usage more rubsut consistency function in Semantic clustering task(SCAN)
The First term of Formula 2 from the SCAN papers  based k-nn classification, 
```math
L(X) = \sum_{X_K \in N_X} D_{KL}(\Phi_{\eta}(X), w_k * \Phi_{\eta}(X_K) )
```
Where $w_k$ in weight of k neighbors based on simularity to $X$ , where soft max of 
$\Phi_{\mu}(X_k)$
 6. Usage MLP net with 'leakly_rely' activation and l2 normalization in clustering head
 7. I implemented only implementation representation learning for 
semantic clustering(pretext) and Semantic clustering tasks without fine-tuning(self-labeling)

#Run code
1. Training pretex model
```
python trainer/simclr_trainer.py --config_file_path configs/env.yml --config_exp_path configs/scan/scan_flowers.yml --exp_name exeriment_name
```

2. Train scan model
```
python trainer/scan_trainer.py --config_file_path configs/env.yml --config_exp_path configs/scan/scan_flowers.yml --exp_name exeriment_name
```
3. Inference pretrained SCAN model
```
python trainer/model_evaluate.py scan_eval --config_file_path config/scan_flowers.yml --pretrained_ckpt /path/to/pretrained-models
```



#Features improvement 
1. Improve performance with stronger GPU machines
2. Hierarchical patches portion and learning Attention its map