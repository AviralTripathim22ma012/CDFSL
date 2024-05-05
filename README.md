# Cross-Domain Few-Shot Learning

![Few-Shot Learning](https://img.shields.io/badge/Few--Shot%20Learning-Cross--Domain-orange.svg)
![Pretraining](https://img.shields.io/badge/Pretraining-Mini%20Imagenet-blue.svg)
![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-CDFSL-blue.svg)
![Testing](https://img.shields.io/badge/Testing-Code-green.svg)

Welcome to the Cross-Domain Few-Shot Learning project repository! This project focuses on modifying and extending the code from the original repository at [IBM/cdfsl-benchmark](https://github.com/IBM/cdfsl-benchmark) to enable pretraining a model on the Mini Imagenet dataset (64 classes) for subsequent fine-tuning on the task of Cross-Domain Few-Shot Learning (CDFSL).

## Overview

Cross-Domain Few-Shot Learning is a challenging problem in machine learning, where the model needs to learn to generalize across different domains and perform well with only a few examples per class.

## Features

This repository provides the following features:

1. **Pretraining**: Codes for pretraining a model on the Mini Imagenet dataset, which contains 64 classes.
2. **Fine-Tuning**: Codes for fine-tuning the pretrained model for the task of Cross-Domain Few-Shot Learning.
3. **Testing**: Code for evaluating the performance of the fine-tuned model on unseen data.
4. **Pretrained Models**: Best trained models are provided in the `checkpoint` folder for direct use or further experimentation.

## Usage

### Pretraining

To pretrain a model on the Mini Imagenet dataset, follow the instructions provided in the `pretraining` directory of this repository.

### Fine-Tuning

After pretraining the model, use the provided fine-tuning codes to adapt the pretrained model for the task of Cross-Domain Few-Shot Learning. Detailed instructions can be found in the `fine_tuning` directory.

### Testing

Evaluate the performance of the fine-tuned model using the testing code available in the `testing` directory. Instructions for running the evaluation are provided there.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- tqdm

## Citation

If you use this code in your research, please consider citing the original repository:

  ```bash
   @inproceedings{guo2020broader,
     title={A broader study of cross-domain few-shot learning},
     author={Guo, Yunhui and Codella, Noel C and Karlinsky, Leonid and Codella, James V and Smith, John R and Saenko, Kate       and Rosing, Tajana and Feris, Rogerio},
     year={2020},
     organization={ECCV}
   }
  ```


## License

This project is licensed under the [MIT License](LICENSE).

---

If you find this project useful, consider giving it a ⭐️!

[![GitHub stars](https://img.shields.io/github/stars/AviralTripathim22ma012/CDFSL.svg?style=social&label=Star)](https://github.com/AviralTripathim22ma012/CDFSL)
