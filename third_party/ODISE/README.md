# ODISE: Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models

**ODISE**: **O**pen-vocabulary **DI**ffusion-based panoptic **SE**gmentation unifies pre-trained text-image diffusion and discriminative models to perform open-vocabulary panoptic segmentation.
It leverages the frozen representation of both these models to perform panoptic segmentation of any category in the wild. 

This repository is the official implementation of ODISE introduced in the paper:

[**Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models**](https://arxiv.org/abs/2303.04803)

[*Jiarui Xu*](https://jerryxu.net),
[*Sifei Liu**](https://research.nvidia.com/person/sifei-liu),
[*Arash Vahdat**](http://latentspace.cc/),
[*Wonmin Byeon*](https://wonmin-byeon.github.io/),
[*Xiaolong Wang*](https://xiaolonw.github.io/),
[*Shalini De Mello*](https://research.nvidia.com/person/shalini-gupta)

CVPR 2023.

[[`arXiv`](https://arxiv.org/abs/2303.04803)] [[`Project Page`](https://jerryxu.net/ODISE/)] [[`BibTeX`](#citation)]

![teaser](figs/teaser.jpg)

## Environment Setup

Install dependencies by running:

```bash
conda create -n odise python=3.9
conda activate odise
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
or (conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch)
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev
git clone git@github.com:NVlabs/ODISE.git 
cd ODISE
pip install -e .
```

(Optional) install [xformers](https://github.com/facebookresearch/xformers) for efficient transformer implementation:
One could either install the pre-built version

```
pip install xformers==0.0.16
```

or build from latest source 

```bash
# (Optional) Makes the build much faster
pip install ninja
# Set TORCH_CUDA_ARCH_LIST if running and building on different GPU types
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
# (this can take dozens of minutes)
```

## Model Zoo

We only fine-tuned 28.1M parameters, the download link is provided in the table below. The code will automatically download the pre-trained model from the link.

<table>
<thead>
  <tr>
    <th align="center"></th>
    <th align="center" style="text-align:center" colspan="3">ADE20K(A-150)</th>
    <th align="center" style="text-align:center" colspan="3">COCO</th>
    <th align="center" style="text-align:center">ADE20K-Full <br> (A-847)</th>
    <th align="center" style="text-align:center">Pascal Context 59 <br> (PC-59)</th>
    <th align="center" style="text-align:center">Pascal Context 459 <br> (PC-459)</th>
    <th align="center" style="text-align:center">Pascal VOC 21 <br> (PAS-21) </th>
    <th align="center" style="text-align:center">download </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center"></td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
  </tr>
  <tr>
    <td align="center"><a href="configs/Panoptic/odise_label_coco_50e.py"> ODISE(label) </a></td>
    <td align="center">22.6</td>
    <td align="center">14.4</td>
    <td align="center">29.9</td>
    <td align="center">55.4</td>
    <td align="center">46.0</td>
    <td align="center">65.2</td>
    <td align="center">11.1</td>
    <td align="center">57.3</td>
    <td align="center">14.5</td>
    <td align="center">84.6</td>
    <td align="center"><a href="https://github.com/NVlabs/ODISE/releases/download/v1.0.0/odise_label_coco_50e-b67d2efc.pth"> checkpoint </a></td>
  </tr>
  <tr>
    <td align="center"><a href="configs/Panoptic/odise_caption_coco_50e.py"> ODISE(caption) </a></td>
    <td align="center">23.4</td>
    <td align="center">13.9</td>
    <td align="center">28.7</td>
    <td align="center">45.6</td>
    <td align="center">38.4</td>
    <td align="center">52.4</td>
    <td align="center">11.0</td>
    <td align="center">55.3</td>
    <td align="center">13.8</td>
    <td align="center">82.7</td>
    <td align="center"><a href="https://github.com/NVlabs/ODISE/releases/download/v1.0.0/odise_caption_coco_50e-853cc971.pth"> checkpoint </a></td>
  </tr>
</tbody>
</table>

## Get Started
See [Preparing Datasets for ODISE](datasets/README.md).

See [Getting Started with ODISE](GETTING_STARTED.md).
## Demo

* Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/xvjiarui/ODISE)

* Run the demo on Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVlabs/ODISE/blob/master/demo/demo.ipynb)

* To run the demo from the command line:

    ```shell
    python demo/demo.py --input demo/examples/coco.jpg --output demo/coco_pred.jpg --vocab "black pickup truck, pickup truck; blue sky, sky"
    ```
    The output is saved in `demo/coco_pred.jpg`.
  
* To run gradio demo locally:
    ```shell
    python demo/app.py
    ```

## Citation

If you find our work useful in your research, please cite:

```BiBTeX
@article{xu2022odise,
  author    = {Xu, Jiarui and Liu, Sifei and Vahdat, Arash and Byeon, Wonmin and Wang, Xiaolong and De Mello, Shalini},
  title     = {{ODISE: Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models}},
  journal   = {arXiv preprint arXiv: 2303.04803},
  year      = {2023},
}
```

## Acknowledgement

Code is largely based on [Detectron2](https://github.com/facebookresearch/detectron2), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [Mask2Former](https://github.com/facebookresearch/Mask2Former), [OpenCLIP](https://github.com/mlfoundations/open_clip), [GLIDE](https://github.com/openai/glide-text2im)

Thank you all for the great open-source projects!