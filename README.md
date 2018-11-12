
# CORnet: Modeling the Neural Mechanisms of Core Object Recognition

## What is this?

A family of simple yet powerful deep neural networks for visual neuroscience. What makes CORnets useful:

- Simple
- Lightweight
- Recurrent
- State-of-the-art on predicting primate neural and behavioral responses [Brain-Score](http://brain-score.org)

[read more...](#longer-motivation)

*Brought to you with <img src="resources/brain.png" width="25px"/> from [DiCarlo Lab](http://dicarlolab.mit.edu) @ [MIT](https://mit.edu).*

## Available Models

*(Click on model names to download the weights of ImageNet-trained models. Note that you do not need to untar them to use during testing.)*

| Name     | Description                                                              |
| -------- | ------------------------------------------------------------------------ |
| [CORnet-Z](https://s3.amazonaws.com/cornet-models/cornet_z_epoch25.pth.tar) | Our smallest, fastest model. Good neural fits                            |
| [CORnet-R](https://s3.amazonaws.com/cornet-models/cornet_r_epoch25.pth.tar) | Recurrent version of CORnet-Z. Better than CORnet-Z + recurrent but slow |
| [CORnet-S](https://s3.amazonaws.com/cornet-models/cornet_s_epoch43.pth.tar) | CORnet-R with ResNet-like blocks. Best overall but slow to train         |


## Quick Start

### Want to test on your own images?

`python run.py test --restore_path <path to model weights> - --model S --data_path <path to your image folder>`

**NOTE** the extra `-` between `--restore_path` and `--model`!

Add `-o <path to save features>` if you want model responses to be saved someplace.

### Want to train on ImageNet?

1. [Get ImageNet](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) if you don't have it yet. If you do, note that validation images need to be put in separate folders, just like train data. Follow the instructions in that link to do so easily.

2. `python run.py train - --model Z --workers 20`. On a single Titan X, it will train for about 20 hours. Models R and S will require 2 GPUs. **NOTE** the extra `-` between `train` and `--model`!

### If installation is needed

`pip install git+https://github.com/dicarlolab/CORnet`


# Details

## Longer Motivation

Deep artificial neural networks with spatially repeated processing (aka, deep convolutional ANNs) have been established as the best class of candidate models of visual processing in primate ventral visual processing stream. Over the past five years, these ANNs have evolved from a simple feedforward eight-layer architecture in AlexNet to extremely deep and branching NASNet architectures, demonstrating increasingly better object categorization performance and increasingly better explanatory power of both neural and behavioral responses. However, from the neuroscientist's point of view, the relationship between such very deep architectures and the ventral visual pathway is incomplete in at least two ways. On one hand, current state-of-the-art ANNs appear to be too complex (e.g. now over 100 levels) compared with the relatively shallow cortical hierarchy (4-8 levels), which makes it difficult to map their elements to those in the ventral visual stream and makes it difficult to understand what they are doing. On the other hand, current state-of-the-art ANNs appear to be not complex enough in that they lack recurrent connections and the resulting neural response dynamics that are commonplace in the ventral visual stream. Here we describe our ongoing efforts to resolve both of these issues by developing a "CORnet" family of deep neural network architectures. Rather than just seeking high object recognition performance (as the state-of-the-art ANNs above), we instead try to reduce the model family to its most important elements (CORnet-Z) and then gradually build new ANNs with recurrent and skip connections while monitoring both performance and the match between each new CORnet model and a large body of primate brain and behavioral data. We report here that our current best ANN model derived from this approach (CORnet-S) is among the top models on Brain-Score, a composite benchmark for comparing models to the brain, but is simpler than other deep ANNs in terms of the number of convolutions performed along the longest path of information processing in the model. All CORnet models are available at \url{github.com/dicarlolab/CORnet}, and we plan to update this manuscript and the available models in this family as they are produced.

Read more: [Kubilius\*, Schrimpf\*, et al. (biorxiv, 2018)](https://doi.org/10.1101/408385)

## Requirements

- Python 3.6+
- PyTorch 0.4.1+
- numpy
- pandas
- tqdm
- fire

## Citation

Kubilius, J., Schrimpf, M., Nayebi, A., Bear, D., Yamins, D.L.K., DiCarlo, J.J. (2018) *CORnet: Modeling the Neural Mechanisms of Core Object Recognition.* biorxiv. doi:10.1101/408385

## License

GNU GPL 3+


# FAQ

- Is CORnet-S *the* model of vision?

No. This is a constant work in progress. We display here our best current models for core object recognition but these models are constantly evolving.

- Why not "CoreNet"?

COR = Core Object Recognition. Also, CORnet has a nice connection to "cortex".

- My model is better than CORnet. Can I place it in this repository?

Exciting. Find a nice name for it and submit it to [Brain-Score.org](http://brain-score.org). This repository is only for CORnet family of models, while Brain-Score is a great place to show your model to the world and link to its own repository.

- Are hyperparameters arbitrary?

No. We tried many architectures and these are the ones that worked best. However, an exhaustive search has not been done. Simpler yet equally good models might exist, as well as more complicated but more predictive models.

- Why do you use classes for defining everything? Aren't functions enough?

Classes allow packaging functions into a single object, providing a good code organization.
