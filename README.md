# About
This repo's attempt is try to pre-train images of prices.

# Todo
- DONE : Write simple cropping augmenter since price image doesn't have much color
- DONE : Create an encoder model
- DONE : Implement contrastive training
- DONE : Implement model checkpointing
- DONE : Change a RestNet50 to the light weight layers
- DONE : Freeze trained layer for finetuning
- DONE : Fine-tune model for classification
- DONE : Compose LSTM classifier model with convolutional feature vectors from SimCLR
- DONE : Speed up dataset sampler
- DONE : Plot image features on tSNE
- ONGOING : Visualized learned CNN features
- Do actual training on bigger dataset
- Prepare sample dataset for classification
- Evaluate classification result and also plot features

# References
- https://github.com/mwdhont/SimCLRv1-keras-tensorflow
- https://amitness.com/2020/03/illustrated-simclr/
- https://github.com/google-research/simclr
- https://github.com/sayakpaul/SimCLR-in-TensorFlow-2

# Config
    [DEFAULT]
    BATCH_SIZE = 512
    EPOCHS = 100
    SAVE_STEPS = 500
    LEARNING_RATE = 0.1
    DECAY_STEPS = 1000
    DATASET_PATH = C:/Users/sharavsambuu/Downloads/EURUSD

# Sample images
![tSNE 3D](images/scattering3d.png?raw=true "tSNE 3D")
![tSNE](images/tSNE.png?raw=true "tSNE")
![grayscale augmented batch](images/Figure_1.png?raw=true "grayscale augmented batch")
![augmented batch](images/crop_augmented_batch.png?raw=true "Augmented batch")
![price image 1](images/2020-03-02_30_330.png?raw=true "Price image 1")
![price image 2](images/2020-03-02_30_300.png?raw=true "Price image 2")


