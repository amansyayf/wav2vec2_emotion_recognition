# wav2vec2_emotion_recognition
## Descriprion
Repository for exploration of the problem of emotion classification. For this task I implemented basic models that use Wav2Vec learned embedings and trained on [RAVDESS](https://paperswithcode.com/dataset/ravdess) dataset.
## Wav2Vec2
Wav2Vec2 is a pre-trained with self-supervised technique speech model. The model consists of three stages: local encoder, contextualized encoder and quantization module. Wav2Vec2 provides butter feature representation that can be helpful in small dataset case. In my solution I used wav2vec 2.0 [base](https://huggingface.co/facebook/wav2vec2-base) model pretrained in Librispeech and not fune-tuned on automatic speech recognition (ASR) and I used embedings after contextualized encoder. It is assumed that information that essential for emotion classification task (like pitch) is not so crucial for ASR.  
## Dataset
In this work only speech part of RAVDESS dataset was used. There are 24 actors enacting 2 statements with 8 different emotions. Following some research articles I used first 20 actors for training, 20-22 for validation and 23-24 for test. The benefit of such split is that model doesn't take information about validation and test speakers while training process. For computational reasons I used only 4 seconds of each audio.
## Metric
I chose the simplist metric accuracy. Advantage of this metric is its clarity and interpretability. But big drawback is some discrimination of classes in unbalanced data which we have in RAVDESS (neutral emotions two time fewer than others). 

## Model 
Model architecture is inspired by [this](https://arxiv.org/abs/2104.03502) article. Firstly, model takes embeddings from freezing Wav2Vec2. Then pass if them through two pointwise convolutional 1D layers with ReLU activation and dropout with a probability of 0.5. Then, two linear layers with the same activation and finally softmax activation returns probabilities for each class. 
In order to reduce date imbalance and overfitting, I applied the simplest data augmentation (random noise and stretch) and sampled an equal amount of data for each emotion category. Experiments showed that this had a beneficial effect on training process, validational curve became less noise and the test accuracy increased. Training curves can be finded [here](https://github.com/amansyayf/wav2vec2_emotion_recognition/tree/main/images).

## Results and discussion
In this work, I explored extracting and modeling features from pretrained Wav2Vec2 model for emotion classification. Trainde weights can be finded [here](https://drive.google.com/drive/folders/1RoH1pZM9VTY1HBQopxk252U_knrRqZ2_?usp=sharing). In order to compare my model I also trained baseline model - simple RandomForestClassifier over eGeMAPS features, which are commonly used in the literature.
Let's analyse training plots, both models trained quite slowly at the beginning, but then they picked up speed; it seems that by the 10th epoch the models were already begun to achieve convergence.
Finally comparison was made on test data. It seems that in my case data augmentation helped to learn representation.
![model comparison](https://github.com/amansyayf/wav2vec2_emotion_recognition/blob/main/images/metric_table.jpg)
In the future as part of this work it would be necessary to try to use features from eGeMAPS inside my model and may be to add LSTM layer, because it have shown to provide useful insights for emotion classification task. 


