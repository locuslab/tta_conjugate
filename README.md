# Test-time Adaptation via Conjugate Pseudo-Labels
This repository provide the the implementation for replicating the results in the following paper:

Test-time Adaptation via Conjugate Pseudo-Labels. NeurIPS 2022. [[arXiv]](https://arxiv.org/abs/2207.09640), [[OpenReview]](https://openreview.net/forum?id=2yvUYc-YNUH)

[Sachin Goyal](https://saching007.github.io/)\*, [Mingjie Sun](https://eric-mingjie.github.io/)\*, [Aditi Raghunanthan](https://www.cs.cmu.edu/~aditirag/), [J. Zico Kolter](http://zicokolter.com/) (* equal contribution).


We provide the source training code (PolyLoss) in `cifar_source_train.py` and `imagenet_source_train.py`. For ImageNet training, we follow the setup of [pytorch ImageNet training](https://github.com/pytorch/examples/tree/main/imagenet). Put the source model in the [saved_models/pretrained](saved_models/pretrained) directory. Here we provide our PolyLoss with eps 6 model on CIFAR-10 as an example.

For test-time adaptation, specify the path in `MODEL.CKPT_PATH` argument in the yaml config file, for polyloss model, an extra parameter `MODEL.EPS` should be added. To use our conjugate pseudo-label loss, specify the optimizer parameter  `OPTIM.ADAPT` as `conjugate` in the config file.

For dataloading, be sure to update the PATH to ImageNet-C/R in the `get_imagenetc_loader` and `get_imagenetr_loader` functions.

## Meta Learning the Loss Function
For meta train a meta loss for test-time adaptation on CIFAR-10-C corruption:
```
python cifar10_meta_train.py --cfg cfgs/cifar.yaml
```

## Evaluation on CIFAR-10/100-C datasets
Train a source classifier on the cifar dataset using the following command. Specify the source loss function, dataset and save path in cfgs/source_train.yaml file.
```
python cifar_source_train.py --cfg cfgs/source_train.yaml 
```
Evaluation code for test-time adaptation on CIFAR-10/100-C. Specify the path of the pre-trained source classifier (MODEL.CKPT_PATH) and test-time-adaptation loss (OPTIM.ADAPT) in cfgs/cifar.yaml file.
```
python cifar_tta_test.py --cfg cfgs/cifar.yaml
```
For MEMO baseline:
```
python memo_cifar.py --cfg cfgs/cifar.yaml
```

## Evaluation on ImageNet-C/R dataset
Train a source classifier on the ImageNet dataset with PolyLoss using the following command.
```
python imagenet_source_train.py --eps 6.0
```

Evaluation code for test-time adaptation on ImageNet-C/R.
```
python imagenet_tta_test.py --cfg cfgs/imagenetc.yaml
```
For MEMO baseline:
```
python memo_imagenet.py --cfg cfgs/imagenetc.yaml
```

## Evaluation on SVHN --> MNIST Benchmark
Train a source classifier on the SVHN dataset using the following command. Specify the source loss function (for ex. polyloss) and save path in cfgs/source_train_svhn.yaml file.
```
python svhn_source_train.py --cfg cfgs/source_train_svhn.yaml 
```
Evaluation code for test-time adaptation on SVHN --> MNIST.
```
python svhn_mnist_tta_test.py --cfg cfgs/digit.yaml
```

## Credits  
This code has been built upon the code accompanying the paper "Tent: Fully Test-time Adaptation by Entropy Minimization" at https://github.com/DequanWang/tent. We are grateful to authors for releasing their code.