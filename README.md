# About
This repo's attempt is try to pre-train images of prices.

# Todo
- DONE : Write simple cropper data augmenter since price image doesn't have much color
- DONE : Create an encoder model
- DONE : Implement some contrastive training
- DONE : Implement model checkpointing
- DONE : Change RestNet50 to light weight layers
- ONGOING : Do actual training on bigger dataset
- Plot some image's features on tSNE
- Freeze trained layer for preparation of finetune
- Prepare sample dataset for classification
- Fine-tune model for classification
- Evaluate classification result and also plot features

# References
- https://github.com/mwdhont/SimCLRv1-keras-tensorflow
- https://amitness.com/2020/03/illustrated-simclr/
- https://github.com/google-research/simclr
- https://github.com/sayakpaul/SimCLR-in-TensorFlow-2

# Sample images
![augmented batch](images/crop_augmented_batch.png?raw=true "Augmented batch")
![price image 1](images/2020-03-02_30_330.png?raw=true "Price image 1")
![price image 2](images/2020-03-02_30_300.png?raw=true "Price image 2")
![price image 3](images/2020-03-02_30_270.png?raw=true "Price image 2")


