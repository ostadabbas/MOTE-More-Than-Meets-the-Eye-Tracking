

# MOTE: More Than Meets the Eyes

**Accepted in ICML 2025**

This repository contains the official implementation of MOTE (More Than Meets the Eyes) for end-to-end multi-object tracking.

**Project Page:** [MOTE Project Page](https://bishoymoussa.github.io/mot-research.github.io/)

**Paper:** Coming Soon

**Hugging Face Demo:** Coming Soon

## Overview

![MOTE](figs/MOTE.png)

MOTE integrates optical flow estimation and softmax splatting to enhance multi-object tracking, especially in challenging scenarios with frequent occlusions.


## Features

- End-to-end multi-object tracking
- Incorporates optical flow estimation, using the RAFT pretrained weights model
- Utilizes softmax splatting for occlusion handling, generating the dissocclusion matrices that are integrated in the ETEM module with the track queries and positional encodings of the subject

<div style="display: flex; flex-direction: row;"> <img src="figs/viz_gif_1.gif" alt="Visualization GIF 1" style="width: 49%; margin-right: 1%;"> <img src="figs/viz_gif_2.gif" alt="Visualization GIF 2" style="width: 49%;"> </div>

## Installation

Clone the repository and install the required dependencies:

```bash
git clone MOTE.git
cd MOTE
pip install -r requirements.txt
```

## Preparing the Optical Flow Estimator

To prepare the optical flow estimator using the RAFT model, follow these steps:


1. Install the required dependencies to run the whole framework, including the RAFT model:

    ```bash
    conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 matplotlib tensorboard scipy opencv -c pytorch
    ```

3. Download the pre-trained weights for the RAFT model from [this link](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT?usp=sharing).

4. Move the downloaded weights to the appropriate experiment directory.

The code for integrating the RAFT model into the MOTE framework is already included within the repository. Refer to the [RAFT GitHub repository](https://github.com/princeton-vl/RAFT?tab=readme-ov-file) for more details on the optical flow estimator.



## Usage

### Training and Evaluation

#### Training on a Single Node

You can download COCO pretrained weights from Deformable DETR. Then, train MOTE on 8 GPUs as follows:

```bash
sh configs/r50_motr_train.sh
```

#### Evaluation on MOT17

You can download the pretrained model of MOTE (the link is in the "Main Results" section), then run the following command to evaluate it on the MOT17 train dataset:

```bash
sh configs/r50_motr_eval.sh
```

For visualizing in demo video, you can enable `vis=True` in `eval.py` like:

```python
det.detect(vis=True)
```

#### Evaluation on MOT17

You can use the pretrained model of MOTE, then run the following command to evaluate it on the MOT17 test dataset (submit to server):

```bash
sh configs/r50_motr_submit.sh
```

**Note:** Make sure to change the directory paths in the config files according to your setup.


## Results

Our experiments show that MOTE outperforms existing methods in handling occlusions and maintaining tracking accuracy. Detailed results are presented below.

### Performance Comparison on the MOT17 Dataset

The table below compares MOTE with other state-of-the-art multi-object tracking methods on the MOT17 dataset. The best results for each metric are in bold.

| Methods | HOTA↑ | AssA↑ | DetA↑ | IDF1↑ | MOTA↑ | IDS↓ |
|---------|-------|-------|-------|-------|-------|------|
| **CNN-based:** |
| Tracktor++ | 44.8 | 45.1 | 44.9 | 52.3 | 53.5 | 2072 |
| CenterTrack | 52.2 | 51.0 | 53.8 | 64.7 | 67.8 | 3039 |
| TraDeS | 52.7 | 50.8 | 55.2 | 63.9 | 69.1 | 3555 |
| QDTrack | 53.9 | 52.7 | 55.6 | 66.3 | 68.7 | 3378 |
| GSDT | 55.5 | 54.8 | 56.4 | 68.7 | 66.2 | 3318 |
| FairMOT | 59.3 | 58.0 | 60.9 | 72.3 | 73.7 | 3303 |
| CorrTracker | 60.7 | 58.9 | 62.9 | 73.6 | 76.5 | 3369 |
| GRTU | 62.0 | 62.1 | 62.1 | 75.0 | 74.9 | 1812 |
| MAATrack | 62.0 | 60.2 | 64.2 | 75.9 | 79.4 | 1452 |
| StrongSORT | 63.5 | 63.7 | 63.6 | 78.5 | 78.3 | 1446 |
| ByteTrack | 63.1 | 62.0 | 64.5 | 77.3 | 80.3 | 2196 |
| **Transformer-based:** |
| TrackFormer | - | - | - | 63.9 | 65.0 | 3528 |
| TransTrack | 54.1 | 47.9 | 61.6 | 63.9 | 74.5 | 3663 |
| MOTE | 57.8 | 55.7 | 60.3 | 68.6 | 73.4 | 2439 |
| MOTRv2 | 62.0 | 60.6 | 63.8 | 75.0 | 78.6 | - |
| **MOTE (Ours)** | **66.3** | **67.8** | **65.4** | **80.3** | **82.0** | **1412** |

### Ablation Studies on MOT17 Dataset

#### Comparing Linear Splatting and Softmax Splatting

Results of an ablation study comparing linear splatting and softmax splatting (models trained for 5 epochs):

| Method | HOTA↑ | MOTA↑ | IDF1↑ | IDS↓ |
|--------|-------|-------|-------|------|
| Linear Splatting | 55.2 | 61.3 | 65.7 | 2450 |
| **Softmax Splatting** | **58.4** | **64.9** | **69.2** | **2134** |

#### Impact of `iters` Parameter in Optical Flow Estimation

Results of varying the `iters` parameter in the forward method of optical flow estimation:

| iters | HOTA↑ | MOTA↑ | IDF1↑ | IDS↓ |
|-------|-------|-------|-------|------|
| 15 | 56.1 | 62.4 | 66.8 | 2300 |
| 20 | **58.3** | 63.7 | **69.0** | 2205 |
| 25 | 57.4 | **64.5** | 68.1 | **2150** |

### Extended Results on MOT20 Dataset

| Methods | HOTA↑ | AssA↑ | DetA↑ | MOTA↑ | IDF1↑ |
|---------|-------|-------|-------|-------|-------|
| FairMOT | 54.6 | 54.7 | 54.7 | 61.8 | 67.3 |
| ByteTrack | 61.3 | 59.6 | 63.4 | 77.8 | 75.2 |
| OC-SORT | 62.4 | 62.5 | - | 75.9 | 76.4 |
| MOTRv2 | 60.3 | 58.1 | 62.9 | 76.2 | 72.2 |
| StrongSORT | 61.5 | 63.2 | 59.9 | 72.2 | 75.9 |
| **MOTE (Ours)** | **65.8** | **66.9** | **64.9** | **81.7** | **79.8** |

### Performance on DanceTrack Dataset

| Methods | HOTA↑ | AssA↑ | DetA↑ | MOTA↑ | IDF1↑ |
|---------|-------|-------|-------|-------|-------|
| FairMOT | 39.7 | 23.8 | 66.7 | 82.2 | 40.8 |
| CenterTrack | 41.8 | 22.6 | 78.1 | 86.8 | 35.7 |
| TraDeS | 43.3 | 25.4 | 74.5 | 86.2 | 41.2 |
| QDTrack | 54.2 | 38.7 | 81.0 | 87.7 | 50.4 |
| ByteTrack | 47.7 | 31.0 | 71.0 | 91.5 | 48.8 |
| OC-SORT | 55.1 | 38.3 | 80.3 | 92.0 | 54.6 |
| TransTrack | 45.5 | 27.5 | 75.9 | 88.4 | 45.2 |
| GTR | 48.0 | 31.9 | 72.5 | 89.7 | 50.3 |
| MOTRv2 | 69.9 | 59.0 | 83.0 | 91.9 | 71.7 |
| MOTRv2* | 73.4 | 64.4 | **83.7** | 92.1 | **76.0** |
| **MOTE (Ours)** | **74.2** | **65.2** | 82.6 | **93.2** | 75.2 |

*Note: MOTRv2* denotes MOTRv2 with extra association, adding validation set for training, and test ensemble.*
### Extended Results on the MOT15 Dataset

The table below provides extended results comparing MOTE and MOTE on the MOT15 dataset, with models trained on the MOT17 dataset. Metrics reported include higher order tracking accuracy (HOTA), multi-object tracking accuracy (MOTA), ID F1 score (IDF1).

| Methods           | HOTA↑ | MOTA↑ | IDF1↑ |
|-------------------|-------|-------|-------|
| MOTE              | 28.4  | 32.5  | 36.3  |
| **MOTE (Ours)**   | **57.2** | **63.2**  | **68.8** |


Below is a demo video showcasing the performance of MOTE. The video loops non-stop for continuous viewing.

<img src="figs/mote_demo.gif" alt="Visualization GIF 3" style="width: 100%"> 

#### Video Demo

We also provide a demo interface which allows for a quick processing of a given video.

```bash
python3 demo.py \
    --meta_arch mote \
    --dataset_file e2e_mot \
    --pretrained ${EXP_DIR}/mote_final.pth \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --merger_dropout 0 \
    --dropout 0 \
    --query_interaction_layer 'ETEM' \
    --extra_track_attn \
    --resume ${EXP_DIR}/mote_final.pth \
    --input_video figs/video_input.avi
```




## SportsMOT Dataset

### Overview

SportsMOT is a large-scale multi-object tracking dataset consisting of 240 video clips from 3 sports categories:
- Basketball
- Football (Soccer)
- Volleyball

The dataset focuses on tracking players on the playground (excluding spectators, referees, and coaches) in various sports scenes. It provides diverse scenes including indoor and outdoor environments, different camera angles, and various player densities and movement patterns.

<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin: 20px 0;">
  <img src="figs/v_-6Os86HzwCs_c009.gif" alt="SportsMOT Basketball" style="width: 80%; margin-bottom: 15px; border: 1px solid #ddd;">
  <p style="text-align: center; margin-bottom: 20px;"><em>Basketball tracking</em></p>
  
  <img src="figs/v_ApPxnw_Jffg_c016.gif" alt="SportsMOT Football" style="width: 80%; margin-bottom: 15px; border: 1px solid #ddd;">
  <p style="text-align: center; margin-bottom: 20px;"><em>Football (Soccer) tracking</em></p>
  
  <img src="figs/v_gQNyhv8y0QY_c013.gif" alt="SportsMOT Volleyball" style="width: 80%; margin-bottom: 15px; border: 1px solid #ddd;">
  <p style="text-align: center;"><em>Volleyball tracking</em></p>
</div>
<p style="text-align: center;"><em>Sample clips from SportsMOT dataset</em></p>

### Dataset Statistics

- 240 video clips across 3 sports categories
- Average of 485 frames per clip
- 720P resolution, 25 FPS
- Diverse scenes: indoor/outdoor, different viewing angles
- Annotations follow MOT Challenge format

### How to Download SportsMOT

1. Sign up on the CodaLab platform: [https://codalab.lisn.upsaclay.fr/](https://codalab.lisn.upsaclay.fr/)
2. Participate in the SportsMOT competition: [https://codalab.lisn.upsaclay.fr/competitions/12424](https://codalab.lisn.upsaclay.fr/competitions/12424)
3. Download the dataset from the "Participate/Get Data" section of the competition page

Alternatively, you can access the dataset through Hugging Face:
[https://huggingface.co/datasets/MCG-NJU/SportsMOT](https://huggingface.co/datasets/MCG-NJU/SportsMOT)

### Dataset Structure

The SportsMOT dataset follows the MOT Challenge 17 format structure:

```
splits_txt (video-split mapping)
    - basketball.txt
    - volleyball.txt
    - football.txt
    - train.txt
    - val.txt
    - test.txt
scripts
    - mot_to_coco.py
    - sportsmot_to_trackeval.py
dataset (in MOT challenge format)
    - train
      - VIDEO_NAME1
        - gt
        - img1
          - 000001.jpg
          - 000002.jpg
        - seqinfo.ini
    - val (same hierarchy as train)
    - test
      - VIDEO_NAME1
        - img1
          - 000001.jpg
          - 000002.jpg
        - seqinfo.ini
```

The ground truth annotations in the `gt` directory follow the MOT Challenge format:
```
<frame_id>, <track_id>, <x>, <y>, <width>, <height>, <confidence>, <class_id>, <visibility>
```

### Format Conversion

For format conversion utilities, refer to the scripts provided in the SportsMOT repository:
[https://github.com/MCG-NJU/SportsMOT/blob/main/codes/conversion](https://github.com/MCG-NJU/SportsMOT/blob/main/codes/conversion)

## Using demo_splatted.py for Feature Visualization

### Overview

The `demo_splatted.py` script generates visualizations of optical flow and disocclusion matrices from video sequences. This helps in understanding how the tracking algorithm handles occlusions and player movements.

### Prerequisites

- Python 3.7+
- PyTorch
- OpenCV
- Matplotlib
- RAFT optical flow module (make sure the path is correctly set in the script)

### How to Run

1. Make sure you have the SportsMOT dataset downloaded and properly organized
2. Ensure the MOTR model weights are available at the specified path
3. Run the script with the following command:

```bash
python demo_splatted.py --mot17_data_path /path/to/sportsmot/dataset/video_sequence --resume /path/to/model_weights/model_motr_final.pth --output_dir /path/to/output_directory
```

Alternatively, you can modify the default paths in the script:

```python
args.mot17_data_path = '/path/to/sportsmot/dataset/video_sequence'
args.resume = '/path/to/model_weights/model_motr_final.pth'
args.output_dir = '/path/to/output_directory'
```

### Output

The script generates the following outputs in the specified output directory:

1. **Optical Flow Visualizations**: Colored representations of player movement between frames
   - Saved in the `flow` subdirectory

2. **Disocclusion Matrix Visualizations**: Heat maps showing areas where players become visible after being occluded
   - Saved in the `disocclusion` subdirectory

3. **Combined Visualizations**: Side-by-side views of optical flow and disocclusion matrices
   - Saved in the `combined` subdirectory

4. **Original Frames**: The input video frames
   - Saved in the `frames` subdirectory

5. **Visualization Videos**: Compiled videos of the combined visualizations
   - Created every 50 frames or at the end of processing

### Understanding the Visualizations

- **Optical Flow**: Colors represent the direction and magnitude of player movement
- **Disocclusion Matrix**: Red intensity shows areas where players become visible after being occluded

### Customizing the Visualization

You can customize the visualization by modifying the following functions in the script:

- `save_flow_visualization`: Change how optical flow is visualized
- `save_disocclusion_visualization`: Adjust the disocclusion matrix visualization
- `create_combined_visualization`: Modify how the combined visualization is created

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

We thank the authors of the MOT17 dataset, SportsMOT dataset, and the community for their valuable contributions. We thank the contributors of Deformable-DETR as well as MOTE. 


## Citation

If you use MOTE in your research, please cite our paper:

```css
@inproceedings{galoaa2025mote,
  title={More Than Meets the Eye: Enhancing Multi-Object Tracking Even with Prolonged Occlusions},
  author={Galoaa, Bishoy and Amraee, Somaieh and Ostadabbas, Sarah},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  pages={--},
  year={2025},
  editor={--},
  volume={267},
  series={Proceedings of Machine Learning Research},
  url={https://github.com/ostadabbas/MOTE-More-Than-Meets-the-Eye-Tracking}
}
```
