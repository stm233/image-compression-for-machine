# image-compression-for-machine

## Installation

Right now this project is pravite. You need to setup a token for users to access.

The place to get a token:
github settings/Developer settings/Personal access tokens/

This method high depend on [CompressAI](https://github.com/InterDigitalInc/CompressAI), If you meet some problems for install compressai, please check their Doc firstly.
```bash
conda create -n ICM python=3.7
conda activate ICM
pip install compressai
pip install pybind11
git clone https://github.com/stm233/image-compression-for-machine.git ICM
cd ICM
pip install -e .
pip install -e '.[dev]'
pip install Cython
pip install scikit-image
pip3 install git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI
```

## Usage

### Objective Detection

#### Dataset

we choose the COCO2017 to train and test our model.

The way to download the COCO2017
```
mkdir ~/dataset/coco2017/ -p
cd ~/dataset/coco2017/
wget http://images.cocodataset.org/zips/train2017.zip ./
wget http://images.cocodataset.org/zips/val2017.zip ./
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip ./
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```

#### Model

cnn : CNNs model to compress Image (original paper model)

cnn2 : CNNs model to compress the feature map

stf: transformer model to compress Image (origianl paper mdoel)

stf2 : 3D zigzag model to compress Image

stf3 : apply the CLIP model to extract the feature map

stf4 : Masked Transformer for 3d zigzag

stf5 : 2d zigzag mode to compress Image

stf6 : 2d zigzag + LRCP (latent residual cross-attention prediction)

stf7 and stf8: different window size of Swin transformer based on stf6


### ICM's machine's Checkpoints
Classification, object detection, and Instance segmentation can be download from Detection2.
https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md



