# ST-GAN: Spatial Transformer Generative Adversarial Networks for Image Compositing
Chen-Hsuan Lin, Ersin Yumer, Oliver Wang, Eli Shechtman, and Simon Lucey  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018  

Website: https://chenhsuanlin.bitbucket.io/spatial-transformer-GAN  
Paper: https://chenhsuanlin.bitbucket.io/spatial-transformer-GAN/paper.pdf  
arXiv preprint: https://arxiv.org/abs/1612.03897

We provide TensorFlow code for the following experiments:
- glasses compositing

--------------------------------------

## Glasses

### Prerequisites

This code is developed with Python3 (`python3`). TensorFlow r1.0+ is required. The dependencies can install by running  
```
pip3 install --upgrade numpy scipy termcolor tensorflow-gpu
```
If you don't have sudo access, add the `--user` flag.

### Dataset  
The following from the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset is required:
- Aligned & cropped images
- Attribute annotations
- Train/val/test partitions

They should be placed in the `dataset` folder under `glasses`.

--------------------------------------

If you find our code useful for your research, please cite
```
@inproceedings{lin2018stgan,
  title={ST-GAN: Spatial Transformer Generative Adversarial Networks for Image Compositing},
  author={Lin, Chen-Hsuan and Yumer, Ersin and Wang, Oliver and Shechtman, Eli and Lucey, Simon},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2018}
}
```

Please contact me (chlin@cmu.edu) if you have any questions!
