# Neuromorphic Lip-Reading with Signed Spiking Gated Recurrent Units

This repository contains the code associated with the paper "Neuromorphic Lip-Reading with Signed Spiking Gated Recurrent Units" published at CVPR Embedded Vision Workshop 2024.

Link to the paper [HERE](https://openaccess.thecvf.com/content/CVPR2024W/EVW/papers/Dampfhoffer_Neuromorphic_Lip-Reading_with_Signed_Spiking_Gated_Recurrent_Units_CVPRW_2024_paper.pdf).

If you use this code, please cite the paper:
```{bibtex}
@InProceedings{Dampfhoffer_2024_CVPR,
    author    = {Dampfhoffer, Manon and Mesquida, Thomas},
    title     = {Neuromorphic Lip-Reading with Signed Spiking Gated Recurrent Units},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {2141-2151}
}
```

## Setting the code and environnment
Use the file pytorchenv.yml to install the pytorch environment used to run this code.
Indicate the dataset path (train_data_root and test_data_root) in the main.py. The DVS-Lip dataset can be downloaded here: https://sites.google.com/view/event-based-lipreading
Note: The training was performed with GPU NVIDIA A100 with 40GB memory. To reduce the GPU memory usage, reduce the number of frames (--nbframes).

## Training the models
Below are the scripts used to train the different versions of the model. See the main.py for the details of the possible arguments.

### Main results
SNN SpikGRU2+ (hybrid: SpikeAct in the frontend and SpikeAct_signed in the backend)
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --hybridsign
```
ANN
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT -a
```

### Backend ablation
FC 
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --front
```
SpikGRU
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --singlegate
```
SpikGRU2
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN
```
SpiGRU+
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --hybridsign --singlegate
```
SpiGRU2+
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --hybridsign
```
ANN backend
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --hybridANN
```

### SNN activation function ablation
full SpikeAct_signed
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --ternact
```
full SpikeAct
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN
```
hybrid
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --hybridsign
```

### Data augmentation ablation
Temporal and spatial
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT -a
```
Temporal
```
python main.py -e 100 -b 32 --nbframes 90 --augT -a
```
Spatial
```
python main.py -e 100 -b 32 --nbframes 90 --augS -a
```
Temporal with different hyperaparameters
For temporal masking with different number of masks and mask length: set --Tnbmask (number of masks) --Tmaxmasklength (maximum mask length)
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT -a --Tnbmask 6 --Tmaxmasklength 18
```

## Fine-tuning the models
The SNN needs 100 epochs of training and then 100 epochs of finetuning to obtain the results of the paper (see paper for details). You can finetune a trained model with the command --finetune, indicating the model's name with -f:
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --hybridsign -f [modelname] --finetune
```

## Testing a model
You can test a trained model with the command -t:
```
python main.py -e 100 -b 32 --nbframes 90 --augS --augT --useBN --hybridsign -f [modelname] -t
```
