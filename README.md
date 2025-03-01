# **Document Shadow Removal**

Document Shadow Removal is a lightweight **Hybrid Document Shadow Removal** architecture inspired by recent research on high-resolution shadow removal. The architecture is inspired from the 2 recent research papers on high-resolution shadow removal:

**Inspired by the following research papers:**
1. SHARDS: Efficient SHAdow Removal using Dual Stage Network for High-Resolution Images
2. High-Resolution Document Shadow Removal via A Large-Scale Real-World Dataset and A Frequency-Aware Shadow Erasing Net

By combining the architecture of LSRNet and DRNet from the first paper with FSENet from the second, this project delivers an effective solution for removing shadows from high-resolution document images while still maintaining the light-weight of the architecture.

## Overview
The core idea is to leverage frequency decomposition to separate low-frequency global illumination (shading) from high-frequency details (e.g., text edges and fine textures). Two dedicated modules are employed:

* *Low-Frequency Module:* Processes the global shading using conventional convolutions.
* *High-Frequency Module:* Restores fine details with dilated convolutions that capture a larger receptive field.

The outputs from these modules are then fused to produce a final high-resolution, shadow-free document image. A hyperbolic tangent (TanH) activation is applied to ensure the output values lie within a normalized range.

## Architecture Details

* Frequency Decomposition:
The input image is decomposed using a Laplacian approach. The low-frequency component is obtained by downsampling (via average pooling) and then upsampling back to the original size. The high-frequency details are computed as the difference between the original image and its low-frequency approximation.

* LowFrequencyModule:
Inspired by LSRNet, this module uses a series of convolutional layers with ReLU activations to correct global illumination issues.

* HighFrequencyModule:
Inspired by DRNet and FSENet, this module uses dilated convolutions to capture and restore fine details that are lost during shadowing.

* Fusion:
The processed outputs from both branches are concatenated along the channel dimension and passed through an additional convolutional layer to produce the final output image.

## Dataset
The repository is set up to work with the [**SD7K dataset**](https://github.com/CXH-Research/DocShadow-SD7K). The expected folder structure is as follows:
```
SD7K/
├── train/
│   ├── input/      # High-resolution shadow images
│   └── target/     # Corresponding shadow-free images
└── test/
    ├── input/
    └── target/
```
Each image is resize to a target resolution of 1024x1024 during data loading.

## Usage
1. Clone the Repository:
```
git clone https://github.com/mohitmarvania/Document_Shadow_Removal.git
cd Document_Shadow_Removal
```

2. Prepare the Dataset:
Place the SD7K dataset in the repository root, ensuring it follows the directory structure mentioned above.


3. Train the Model:
Run the main training script:
```
python main.py
``` 
*NOTE: By default, the script runs for 5 epochs with a batch size of 2. Adjust the parameters as needed.*

## References
1. [SHARDS: Efficient SHAdow Removal using Dual Stage Network for
High-Resolution Images](https://openaccess.thecvf.com/content/WACV2023/papers/Sen_SHARDS_Efficient_Shadow_Removal_Using_Dual_Stage_Network_for_High-Resolution_WACV_2023_paper.pdf)
2. [High-Resolution Document Shadow Removal via A Large-Scale Real-World
Dataset and A Frequency-Aware Shadow Erasing Net](https://arxiv.org/pdf/2308.14221v4)

## Acknowledgments

* Thank you to the authors of the SHARDS and High-Resolution Document Shadow Removal papers for inspiring this project.

## Issues

If you find any issue or suggestion please let me know by opening a new issue. Thank you for your time :)
