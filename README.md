# StarGAN-v2 implementation.

## Preparations
Make sure you run on Python version 3.8 or 3.9

1. Clone the repo.
    ```
    git clone https://github.com/karimdzan/StarGANv2.git

    cd StarGANv2
    ```

2. Make sure to download the original dataset(https://www.kaggle.com/datasets/jessicali9530/celeba-dataset). 

3. Download model checkpoint.
    ```
    gdown 1qNTAlNszzmIUE-R7ldS8uhQXX4vvjqUs
    ```

## Training

6. To train your own model, use [train.py](./train.py):
    ```
    python3 train.py
    ```
   You will need to specify your wandb login key
   
   To resume training you will need to append path to [train.yaml](src/configs/train.yaml) file in configs as such:
    ```
    ...
    img_size: 256

    resume: path/to/checkpoint.pth
    
    StarGANv2:
    ...
    ```
    The configs should be the same as the original

## Inference

5. Place your checkpoint folder in test.yaml file. You can use [test.py](./test.py) to blend style from ref.jpg to org.jpg in [pictures](/pictures) folder, just put your pictures in ref and src folder respectively and run:
    ```
    python3 test.py
    ```
The results will be in output folder.

## Repository Structure

```
src/
├── configs/                        # Configuration files
├── datasets/                       # Dataset modules
├── logger/                         # Logging modules
├── loss/                           # Loss function modules
│   ├── AdvLoss.py
│   ├── Loss.py
│   ├── StarGANv2Loss.py
│   ├── __init__.py
├── metric/                         # Metric and evaluation modules
│   ├── base_metric.py
│   ├── lpips_.py
│   ├── __init__.py
├── model/                          # Model-related modules
│   ├── base_model.py
│   ├── layers.py
│   ├── model.py
│   ├── __init__.py
├── trainer/                        # Training and trainer modules
├── utils/                          # Utility modules
```

    
## Artifacts

Report: #TODO

Model checkpoints folder: https://drive.google.com/drive/folders/1JJo-omS6OjiTX6c9Ug3qjcE8PozB6Id5?usp=sharing
