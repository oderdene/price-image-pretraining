# About
This repo's attempt is try to pre-train images of prices.

# Todo
- Write simple masking data augmenter since price image doesn't have much color
- Create an encoder model
- Do some contrastive training
- Plot some image's features on tSNE
- Freeze trained layer for preparation of finetune
- Fine-tune model for classification
- Evaluate classification result and also plot features

# References
- https://github.com/mwdhont/SimCLRv1-keras-tensorflow
- https://amitness.com/2020/03/illustrated-simclr/
- https://github.com/google-research/simclr


# Sample images

![price image 1](https://raw.githubusercontent.com/sharavsambuu/price-image-pretraining/master/images/2020-03-02_30_330.png)
![price image 2](https://raw.githubusercontent.com/sharavsambuu/price-image-pretraining/master/images/2020-03-02_30_300.png)
![price image 3](https://raw.githubusercontent.com/sharavsambuu/price-image-pretraining/master/images/2020-03-02_30_270.png)
