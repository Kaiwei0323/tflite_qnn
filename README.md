# TFLITE_QNN

## Prerequisites
### Hardware
* Platform: QCS6490
    - CPU: Octa-Core Kyro 670 CPU
    - GPU: Qualcomm Adreno 643
 ### Software 
 * OS: Ubuntu 20.04
 * Qualcomm AI Engine Direct SDK: v2.26.0.240828
 * Model: YOLOv8-Detection-Quantized.tflite

## Qualcomm AI Engine Direct SDK Installation
[Qualcomm AI Engine Direct SDK](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)

## Environment Setup
```
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
pip3 install tensorflow
pip3 install tqdm
export LD_LIBRARY_PATH=/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828/lib/aarch64-ubuntu-gcc9.4
export ADSP_LIBRARY_PATH=/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828/lib/hexagon-v68/unsigned
```
