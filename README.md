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
cd image-compression
pip install -e .
pip install -e '.[dev]'
```