# Detector Comparison

Comparison of feature detector algorithms on the Tartan Air Dataset https://theairlab.org/tartanair-dataset/

## Installation
Repository contains submodule for EfficientLoFTR so use recursive cloning
```shell
git clone --recursive git@github.com:nazcaspider/detector_comparison.git
cd detector_comparison
conda env create -f environment.yaml
conda activate detector_comparison
```
## Download datasets and pre-trained model
Download the Seasidetown sample from:
https://cmu.app.box.com/s/zzwyrrqm2ir2z0z75tqowpq91gny2sjk

Download the CarWelding sample from:
https://cmu.box.com/s/qpoikn7owhhj2v718m8u9cdmpsqmuq14

Extract datasets to folder: datasets/TartanAir
Path should look like: detector_comparison/datasets/tartanair/carwelding_sample_P007

Download Efficient LoFTR model from:
https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q?usp=sharing

Move model to weights, path should look like:
detector_comparison/weights/eloftr_outdoor.ckpt

## Execution
```shell
python evaluate.py
```
