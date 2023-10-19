# Implementation of Clippy AdaGrad from "Improving Training Stability for Multitask Ranking Models in Recommender Systems"

This repository contains an unofficial PyTorch implementation of the paper ["Improving Training Stability for Multitask Ranking Models in Recommender Systems"](https://arxiv.org/abs/2302.09178). 

Clippy is a method developed by Google to address the training instability problem they faced when training large deep learning models for recommender systems on YouTube data.

According to the paper:

*"Clippy offers larger gains when the model is more complex and trained with a larger learning rate."*

*"It has shown significant improvements on training stability in multiple ranking models for YouTube recommendations and is productionized in some large and complex models."*

## Requirements

- Python 3.6 or later
- PyTorch 2.1.0
- Numpy
- tqdm
- sklearn
- polars

## Dataset

The [AliExpress dataset](https://tianchi.aliyun.com/dataset/74690) used in this project is a real-world dataset gathered from the search system logs of AliExpress. 

The dataset contains both categorical and numerical data. The goal is to predict whether a user will click and purchase a product (2 tasks).

## Model

The model used in this project is a Shared Bottom Model, which is the type of multitask learning model used in the paper.

The dataset and model implementation comes from the [Multitask-Recommendation-Library](https://github.com/easezyc/Multitask-Recommendation-Library) repository.

The model is trained using the ClippyAdagrad optimizer and the Binary Cross Entropy loss function.

## Usage

To run the code, simply execute the test script. 

Make sure to replace "path_to_data" with the actual path to your dataset.

```python
python test.py
```

## License

This project is licensed under the MIT License.
