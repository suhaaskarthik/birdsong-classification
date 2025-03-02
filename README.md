# birdsong-classification
Classifies 184 different bird calls!!

This repository consists of two files, one of which classifies audio through simple mel-coefficients and averaging through the timestamps (reaching accuracies of mere 3-4%). The architecture used here is just a normal sequential model. Whereas the other consists of an upgraded version
that uses mel spectrograms, and works with images rather than arrays. So we leverage this to apply convolutional layers and fine-tuning complicated CNN architectures (efficientnet in this case). Augmentation, learning rate schedulers are applied. Data preprocessing, batching and shuffling are all done through tesnrflows graph based execution
Please check out my kaggle notebook for further clarifications (https://www.kaggle.com/code/suhaaskarthikeyan/audio-classification-tl)
