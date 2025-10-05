# Disentangled Latent Gating for Expression-Preserving StyleCLIP  
(ICCV 2021 Reimplementation & Improvement Project)

Contributors:  
KAIST
Calvin Samuel (20210899), Himani Paudayal (20220776), Lunar Sebastian Widjaja (20220933), Samuel Dawit Assefa (20230858), Yomori Achiza (20230931)

---

## Poster  
You can view our project poster here:  
[https://drive.google.com/file/d/1hD8vL7_lSOwZ1lC2b5JY6oTJNRUGiw8h/view?usp=sharing]

---

## Base Paper  
StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery (ICCV 2021)  
https://arxiv.org/abs/2103.17249

---

## Description  
This repository contains our reimplementation and improvement of StyleCLIP, focusing on expression preservation and disentanglement in latent space.  
Our proposed method, Disentangled Latent Gating, splits the latent vector into multiple segments, each trained with static or dynamic gating to regulate the influence of edits.  
This approach allows localized control over features such as expression, hair, and geometry, improving the realism and consistency of generated images.

---

## Training Setup  
To begin training, install all required dependencies and prepare the dataset and pretrained weights as indicated in the args parser.  

Training methods implemented in this project:

- Static Mapper Gates: Used for identity-preserving transformations such as "Hillary Clinton" and "Beyonce".  
- Dynamic Mapper Gates: Used for subtle or expression-based edits such as "Surprised" and "Annoyed".  

Training data is located in `/StyleCLIP/data`.  
Pretrained weights are located in `/StyleCLIP/pretrained_models/`.  
Trained results are automatically saved in the following folders:

- `train_1.py/` for static mapper  
- `train_2.py/` for dynamic mapper  

If there are path issues, modify them directly in the argument parser of the corresponding training scripts.

---

## Inference Setup  
To run inference, use the provided pretrained mapper checkpoints and specify the corresponding text prompts.  

Example prompt–model mapping:  
- Beyonce_Best.pt → "Beyonce Knowles"  
- Clinton_Best.pt → "Hillary Clinton"  
- Annoyed_Best.pt → "Annoyed face"  
- Surprised.pt → "Surprised face"  

Inference results are saved automatically in the following folders:
- `inference_1_results/`  
- `inference_2_results/`

If path errors occur, adjust the file directories in the args parser accordingly.

---

## Extra Information  
This implementation introduces additional loss functions such as color, correlation, and orthogonality losses to improve edit quality and disentanglement.  
Training time was reduced significantly with selective gate training, while maintaining better preservation of expression and identity.  
Recommended parameters can be tuned in the args file for balancing CLIP loss, ID loss, and other terms depending on the editing prompt.

---

## Missing Files  
The following files are not included in this repository due to size and licensing restrictions:  
- train.pt  
- test.pt  
- stylegan2-ffhq-config-f.pt  
- model_ir_se50.pth (ArcFace model)

To obtain them, please visit the official StyleCLIP repository:  
https://github.com/orpatashnik/StyleCLIP  
or email himanipaudayal07@kaist.ac.kr for access.

---

## Acknowledgements  
Pretrained models and datasets are borrowed from the original StyleCLIP paper.  
All improvements, including gating mechanisms, training structure, and additional loss terms, are original contributions made by our team.
