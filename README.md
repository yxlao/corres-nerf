<h2 align="center" style="border-bottom: none;">
CorresNeRF: Image Correspondence Priors for Neural Radiance Fields
</h2>

<p align="center">
    <a href="https://yxlao.github.io/">Yixing Lao</a>
    ·
    <a href="https://xiaogang00.github.io/">Xiaogang Xu</a>
    ·
    <a href="https://zhipengcai.github.io/">Zhipeng Cai</a>
    ·
    <a href="https://xh-liu.github.io/">Xihui Liu</a>
    ·
    <a href="https://hszhao.github.io/">Hengshuang Zhao</a>
    <br>
    (NeurIPS 2023)
</p>

<p align="center">
   <a href="https://yxlao.github.io/corres-nerf/"><img src="https://img.shields.io/badge/Project-gray?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8%2BCjxzdmcKICAgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwIDAgMzIgMzIiCiAgIGlkPSJFZGl0YWJsZS1saW5lIgogICB2ZXJzaW9uPSIxLjEiCiAgIHZpZXdCb3g9IjAgMCAzMiAzMiIKICAgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIKICAgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIgogICB4bWxuczpzdmc9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcwogICAgIGlkPSJkZWZzNyIgLz48cGF0aAogICAgIGQ9IiAgTTI3LDI5SDVWMTdIMy4yMzVjLTEuMTM4LDAtMS42NjktMS40MTktMC44MTItMi4xNjhMMTQuMTMxLDMuNzQ1YzEuMDQ4LTAuOTkzLDIuNjg5LTAuOTkzLDMuNzM3LDBsMTEuNzA3LDExLjA4NyAgQzMwLjQzMywxNS41OCwyOS45MDIsMTcsMjguNzYzLDE3SDI3VjI5eiIKICAgICBmaWxsPSJub25lIgogICAgIGlkPSJYTUxJRF8xXyIKICAgICBzdHJva2U9IiMwMDAwMDAiCiAgICAgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIgogICAgIHN0cm9rZS1saW5lam9pbj0icm91bmQiCiAgICAgc3Ryb2tlLW1pdGVybGltaXQ9IjEwIgogICAgIHN0cm9rZS13aWR0aD0iMiIKICAgICBzdHlsZT0ic3Ryb2tlOiNmZmZmZmYiIC8%2BPHBhdGgKICAgICBkPSIgIE0yMCwyOWgtOHYtNmMwLTIuMjA5LDEuNzkxLTQsNC00aDBjMi4yMDksMCw0LDEuNzkxLDQsNFYyOXoiCiAgICAgZmlsbD0ibm9uZSIKICAgICBpZD0iWE1MSURfMl8iCiAgICAgc3Ryb2tlPSIjMDAwMDAwIgogICAgIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIKICAgICBzdHJva2UtbGluZWpvaW49InJvdW5kIgogICAgIHN0cm9rZS1taXRlcmxpbWl0PSIxMCIKICAgICBzdHJva2Utd2lkdGg9IjIiCiAgICAgc3R5bGU9InN0cm9rZTojZmZmZmZmIiAvPjwvc3ZnPg%3D%3D&labelColor=4682b4" alt="Project Page"></a>
   <a href="https://arxiv.org/abs/2312.06642"><img src="https://img.shields.io/badge/Paper-gray?style=for-the-badge&logo=arxiv&labelColor=b91e1e"
      alt="Paper"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/yxlao/corres-nerf/gh-pages/assets/demo_video.gif" height=320>
</p>

<p align="center">
  For more details, please visit our <a href="https://yxlao.github.io/corres-nerf/">project page</a> and <a href="https://arxiv.org/abs/2312.06642">paper</a>.
</p>

## Installation

```bash
# Environment
conda create -n corresnerf python=3.10
conda activate corresnerf

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/Parskatt/DKM.git
```

## Data

- Download [LLFF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) for NeRF and extract to `data/nerf_llff`.
- Download [DTU dataset](https://www.dropbox.com/scl/fo/um3wj3ctiuoottbfmqmgb/ABZRltszDvWHJ824UL6DHw0?rlkey=3vjok0aivnoiaf8z5j6w05k92&e=1&dl=0) for NeuS and extract to `data/neus_dtu`.

After extraction, we shall have:

```bash
$ tree -lL 2 data
data
|-- nerf_llff
|   |-- fern
|   |-- flower
|   # ... more scenes
|   |-- room
|   `-- trex
`-- neus_dtu
    |-- dtu_scan105
    |-- dtu_scan106
    # ... more scenes
    |-- dtu_scan83
    `-- dtu_scan97
```

## Pre-trained weights

We release [pre-trained weights](https://drive.google.com/file/d/1C4sNNVd43pWP9qGvKdEu7RdgU-hSJ41Z/view?usp=sharing) for the paper for each scene.
Extract `pretrained_weights.zip` to `logs/` and we shall have:

```bash
$ tree -lL 4 logs
logs
|-- nerf_llff
|   |-- fern
|   |   `-- 050000.tar
|   |-- flower
|   |   `-- 050000.tar
|   # ... more scenes
|   |-- room
|   |   `-- 050000.tar
|   `-- trex
|       `-- 050000.tar
`-- neus_dtu
    |-- dtu_scan105
    |   `-- checkpoints
    |       `-- ckpt_200000.pth
    |-- dtu_scan106
    |   `-- checkpoints
    |       `-- ckpt_200000.pth
    # ... more scenes
    |-- dtu_scan83
    |   `-- checkpoints
    |       `-- ckpt_200000.pth
    `-- dtu_scan97
        `-- checkpoints
            `-- ckpt_200000.pth
```

## Inference with the pre-trained weights

We use the `fern` scene for NeRF on LLFF and the `dtu_scan24` scene for NeuS on DTU as example.

```bash
# Inference with NeRF. This will load the latest checkpoint (050000.tar in this case).
# - The rendered images will be saved in logs/nerf_llff/fern/renderall_050000.
python -m src.tools.run_nerf --config configs/nerf_llff.txt --scene fern --factor 8 --render_all

# Inference with NeuS. This will load the specified checkpoint (ckpt_200000.pth in this case).
# - The rendered images will be saved in logs/neus_dtu/dtu_scan24/renders.
# - The extracted mesh will be saved in logs/neus_dtu/dtu_scan24/meshes.
python -m src.tools.run_neus --conf configs/neus_dtu.conf --case dtu_scan24 --from_checkpoint ckpt_200000.pth render --resolution_level 4
python -m src.tools.run_neus --conf configs/neus_dtu.conf --case dtu_scan24 --from_checkpoint ckpt_200000.pth extract_mesh
```

## Correspondence matching and training

The full pipeline includes:

- Extract correspondences.
- Train NeRF / NeuS models.
- Render with the trained NeRF / NeuS.
- Extract mesh with the trained NeuS.

For NeRF on LLFF, we use the `fern` scene as example.

```bash
# Matcher
python -m src.tools.run_matcher --enable_matcher --enable_filter --matcher_name dkm --config_name default --num_views 3 --dataset llff --scene_dir data/nerf_llff/fern --corres_dir data_corres/nerf_llff/fern

# Train
python -m src.tools.run_nerf --config configs/nerf_llff.txt --scene fern

# Render
python -m src.tools.run_nerf --config configs/nerf_llff.txt --scene fern --factor 8 --render_all
```

For NeuS on DTU, we use the `dtu_scan24` scene as example.

```bash
# Match
python -m src.tools.run_matcher --enable_matcher --enable_filter --matcher_name dkm --config_name default --num_views 3 --dataset dtu --scene_dir data/neus_dtu/dtu_scan24 --corres_dir data_corres/neus_dtu/dtu_scan24

# Train
python -m src.tools.run_neus --conf configs/neus_dtu.conf --case dtu_scan24 --is_continue train

# Render & extract mesh
python -m src.tools.run_neus --conf configs/neus_dtu.conf --case dtu_scan24 --from_checkpoint ckpt_200000.pth render --resolution_level 4
python -m src.tools.run_neus --conf configs/neus_dtu.conf --case dtu_scan24 --from_checkpoint ckpt_200000.pth extract_mesh
```

## Citation

If you find our work useful in your research, please consider citing:

```bib
@inproceedings{lao2023corresnerf,
  title     = {{CorresNeRF}: Image Correspondence Priors for Neural Radiance Fields},
  author    = {Lao, Yixing and Xu, Xiaogang and Cai, Zhipeng and Liu, Xihui and Zhao, Hengshuang},
  booktitle = {NeurIPS},
  year      = {2023}
}
```

This project is built with [CamTools](https://github.com/yxlao/camtools).

<p>
<a href="https://github.com/yxlao/camtools"><img src="https://raw.githubusercontent.com/yxlao/camtools/master/camtools/assets/built_with_camtools_dark.svg" width=240 alt="Built with CamTools"></a>
</p>
