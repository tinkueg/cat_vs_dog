# Train Dogs and Cats classification using PyTorch from scratch
This repository shows Dogs and Cats classification using PyTorch from scratch.
Here I use convnet for dogs vs. cats classification.

## Crete Virtual Environment
Create a virtiual environment.
```console
$conda create conda create --name ml_env_1.11
$conda activate ml_env_1.11
```
Install PyTorch
```console
$conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
Install matplotlib
```console
$conda install -c conda-forge matplotlib
```

## Data Prepration
Download data from https://www.kaggle.com/c/dogs-vs-cats/data and uncompress it.
Then execute data_prep.py to prepare train, test data directory with 2000 images
of dogs and cats.

## Training
To train the model on extracted data execute the following command.
```console
$python train.py
```

## Results
![image info](./plot.png)
![image info](./loss.png)
