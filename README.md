# Low Light Image Enhancement with Neural Network - Experimentation with RetinexFormer model - Group 66

![University of Pennsylvania](reade_me_assets/UniversityofPennsylvania_FullLogo_RGB.png)

**Course:** ESE 5460-001 202430 Principles Of Deep Learning

## Authors
- Tri Le: [trilev@seas.upenn.edu](mailto:trilev@seas.upenn.edu)
- Navjot Singh Chahal: [nschahal@seas.upenn.edu](mailto:nschahal@seas.upenn.edu)
- Margarita Granat: [mgranat@upenn.edu](mailto:mgranat@upenn.edu)

## Abstract
This project explores low-light image enhancement (LLIE) using RetinexFormer, a transformer-based model built on Retinex Theory. We replicated baseline results and introduced modifications to improve performance, including experimenting with different loss functions, integrating the Gradient Graph Laplacian Regularizer (GGLR), and incorporating the Lion optimizer. Our findings suggest that RetinexFormer performs well on curated datasets but may benefit from integrating generative approaches like GANs for broader applications.

## Introduction
LLIE is a computer vision task that seeks to improve the visibility and quality of photos or video streams captured in poorly lit conditions. This project focuses on RetinexFormer, a transformer-based architecture leveraging insights from Retinex Theory to perform LLIE tasks.

## Approach
We inherited the RetinexFormer codebase and training pipeline, originally built on the open-source BasicSR framework. We focused our experiments on three primary modifications:
1. Training with different loss functions,
2. Introducing the Gradient Graph Laplacian Regularizer (GGLR),
3. Incorporating the Lion optimizer.

## Results
The primary quantitative metric used is Peak Signal-to-Noise Ratio (PSNR). Charbonnier Loss and L1 Loss yielded the best performance. Subjectively, the GGLR+L1LuminanceLoss configuration provided more accurate enhancements, balancing improved illumination fidelity with better color and geometry retention.

## Conclusion
Overall, we are satisfied with both the vanilla RetinexFormer and some of our tweaked models in LLIE tasks. However, the computational efficiency of the model leaves some to be desired. We think the best use for our model would be as a low-resolution image enhancer module that works in tandem with other feature extraction modules.

## Acknowledgements
We would like to express our deepest gratitude to everyone who supported us throughout this project. 

**Instructor:**
We are immensely grateful to Pratik Chaudhari ([pratikac@seas.upenn.edu](mailto:pratikac@seas.upenn.edu)), whose guidance, expertise, and encouragement were invaluable throughout the course ESE 546: Principles of Deep Learning (Fall 2024). His insights and feedback greatly enhanced our understanding and execution of this project. [Website](https://pratikac.github.io)

**Teaching Assistants:**
We would also like to extend our heartfelt thanks to the teaching assistants for their continuous support and assistance:
- Fiona Luo ([fionaluo@seas.upenn.edu](mailto:fionaluo@seas.upenn.edu))
- Michael Yao ([myao2199@seas.upenn.edu](mailto:myao2199@seas.upenn.edu))
- Rohit Jena ([rjena@seas.upenn.edu](mailto:rjena@seas.upenn.edu))
- Zhaojin Sun ([zjsun@seas.upenn.edu](mailto:zjsun@seas.upenn.edu))
- Alexander Kyimpopkin ([alxkp@seas.upenn.edu](mailto:alxkp@seas.upenn.edu))
- Aalok Patwa ([apatwa@seas.upenn.edu](mailto:apatwa@seas.upenn.edu))
- Gokul Nair ([gokuln@seas.upenn.edu](mailto:gokuln@seas.upenn.edu))
- Tanaya Gupte ([tanayag@seas.upenn.edu](mailto:tanayag@seas.upenn.edu))
- Karthikeya Jayarama ([jkarthik@seas.upenn.edu](mailto:jkarthik@seas.upenn.edu))
- Georgios Mentzelopoulos ([gment@seas.upenn.edu](mailto:gment@seas.upenn.edu))
- Justin Qiu ([jsq@seas.upenn.edu](mailto:jsq@seas.upenn.edu))

Their dedication and willingness to help at every step were crucial to the successful completion of this project.


## For Setup and Installation refer to [READEME_BASE_PAPER.md](README_BASE_PAPER.md)

### Original Video
You can view the original video below:

### Original Video
You can view the original video below:

[Original Video](Enhancement/video.mp4)

### Enhanced Video Output
You can view the enhanced video output below (Reduced to 5 FPS for computational efficiency / constraints):

[Enhanced Video Output](Enhancement/output_video.mp4)

## Some more Real-world testing results in G-drive repository:
Enhanced Video directory
Models Used .pth files
Our Data Sets
[Our_Gdrive_ForExperiment_Results_of_late_night_dark_shot_and_movie_scenes](https://drive.google.com/drive/folders/1MOwvRiOAB3fAoSF83KVJW8EBdpXAcVKZ?usp=drive_link)

### Discussions as we progreesed through the project:

#### [Implemented ReduceLROnPlateau learning rate scheduler](https://github.com/your-repo/commit/55766df)
- **Reasoning**: To improve the training process and achieve better convergence, we implemented the ReduceLROnPlateau learning rate scheduler. This scheduler reduces the learning rate when a metric has stopped improving, helping the model to converge more effectively.
  - [`Options/RetinexFormer_LOL_v1_ReduceLROnPlateau.yml`](Options/RetinexFormer_LOL_v1_ReduceLROnPlateau.yml): Configuration file for ReduceLROnPlateau.
  - [`basicsr/models/base_model.py`](basicsr/models/base_model.py): Added ReduceLROnPlateau scheduler.
  - [`basicsr/models/lr_scheduler.py`](basicsr/models/lr_scheduler.py): Defined ReduceLROnPlateau class.
  - [`basicsr/version.py`](basicsr/version.py): Updated version file.

#### [Added sample data from testing](https://github.com/your-repo/commit/1851527)
- **Reasoning**: Providing sample data from testing helps in validating the model's performance and ensures reproducibility of results. It also serves as a reference for future experiments and comparisons.
  - [`Enhancement/output_video.mp4`](Enhancement/output_video.mp4): Enhanced video output.
  - [`Enhancement/video.mp4`](Enhancement/video.mp4): Original video input.
  - [`Options/RetinexFormer_LOL_v1_ReduceLROnPlateau.yml`](Options/RetinexFormer_LOL_v1_ReduceLROnPlateau.yml): Updated configuration file.
  - [`Options/RetinexFormer_LOL_v1_ReduceLROnPlateau_01.yml`](Options/RetinexFormer_LOL_v1_ReduceLROnPlateau_01.yml): New configuration file for additional testing.

#### [Image inferencing, mixed precision training, and LuminanceL1 Loss](https://github.com/your-repo/commit/6781b4a)
- **Reasoning**: Enhancing the model's inference capabilities and training efficiency is crucial for practical applications. Mixed precision training reduces memory usage and speeds up training, while the LuminanceL1 Loss improves the model's performance on low-light images.
  - [`Enhancement/test_from_dataset.py`](Enhancement/test_from_dataset.py): Script for testing from dataset.
  - [`Options/RetinexFormer_LOL_v1.yml`](Options/RetinexFormer_LOL_v1.yml): Configuration file for LOL-v1 dataset.
  - [`basicsr/models/losses/__init__.py`](basicsr/models/losses/__init__.py): Added LuminanceL1Loss import.
  - [`basicsr/models/losses/losses.py`](basicsr/models/losses/losses.py): Defined LuminanceL1Loss class.
  - [`basicsr/train.py`](basicsr/train.py): Updated training script.

#### [GGLR regularizer - trained with GGLR+LuminanceL1 loss. Lion optimizer, and will train with the same setting](https://github.com/your-repo/commit/2e02bac)
- **Reasoning**: Introducing the Gradient Graph Laplacian Regularizer (GGLR) and the Lion optimizer aims to enhance the model's performance and stability. GGLR helps in preserving image structure, while the Lion optimizer improves convergence and generalization.
  - [`Options/RetinexFormer_LOL_v1.yml`](Options/RetinexFormer_LOL_v1.yml): Configuration file for LOL-v1 dataset.
  - [`basicsr/models/archs/RetinexFormer_arch.py`](basicsr/models/archs/RetinexFormer_arch.py): Updated architecture with GGLR and Lion optimizer.
  - [`basicsr/models/image_restoration_model.py`](basicsr/models/image_restoration_model.py): Added Lion optimizer support.
  - [`basicsr/models/losses/__init__.py`](basicsr/models/losses/__init__.py): Added GradientGraphLaplacianRegularizer import.
  - [`basicsr/models/losses/losses.py`](basicsr/models/losses/losses.py): Defined GradientGraphLaplacianRegularizer class.
  - [`basicsr/train.py`](basicsr/train.py): Updated training script to include GGLR loss.


## Citation
```shell
@InProceedings{Cai_2023_ICCV,
    author    = {Cai, Yuanhao and Bian, Hao and Lin, Jing and Wang, Haoqian and Timofte, Radu and Zhang, Yulun},
    title     = {Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {12504-12513}
}
```