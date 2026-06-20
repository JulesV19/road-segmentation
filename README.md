# Road Segmentation — Cityscapes

Semantic segmentation of urban driving scenes. UNet + EfficientNet-B4 encoder trained on [Cityscapes](https://www.cityscapes-dataset.com/), 8 classes, **0.836 mIoU**.

---

## Demo

<table>
  <tr>
    <td><img src="assets/demo_stuttgart00.gif"/></td>
  </tr>
  <tr>
    <td><img src="assets/demo_stuttgart01.gif"/></td>
  </tr>
</table>

*Left: original frame — Right: segmentation overlay (α = 0.5)*

---

## Results

| Class | IoU |
|---|---|
| flat | 0.965 |
| construction | 0.869 |
| nature | 0.878 |
| sky | 0.907 |
| vehicle | 0.877 |
| human | 0.621 |
| object | 0.521 |
| **mean** | **0.836** |

Trained for 22 epochs · EfficientNet-B4 encoder · 1024×512 · Colab A100

---

## Examples

<img src="assets/examples/example_00.png" width="100%"/>
<img src="assets/examples/example_03.png" width="100%"/>
<img src="assets/examples/example_06.png" width="100%"/>

---

## Quick Start

**1. Preprocess** *(run once locally)*
```bash
python scripts/preprocess.py \
  --data_root /path/to/cityscapes \
  --output_dir /path/to/preprocessed \
  --width 1024 --height 512
```

**2. Train** *(Colab)*

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), upload the zipped preprocessed data to Drive, run all cells.

**3. Inference on video**
```bash
python scripts/infer_video.py \
  --checkpoint path/to/best.pth \
  --frames_dir path/to/frames/ \
  --output     path/to/output.mp4
```

---

## Architecture

| | |
|---|---|
| Model | UNet |
| Encoder | EfficientNet-B4 (ImageNet pretrained) |
| Loss | 0.5 × CrossEntropy + 0.5 × Dice |
| Optimizer | AdamW + cosine warmup |
| Input | 1024×512 |
| Classes | 8 (flat · construction · object · nature · sky · human · vehicle) |

---

## Class Palette

<table>
  <tr>
    <td align="center"><img width="20" height="20" src="https://placehold.co/20x20/804080/804080"/> flat</td>
    <td align="center"><img width="20" height="20" src="https://placehold.co/20x20/464646/464646"/> construction</td>
    <td align="center"><img width="20" height="20" src="https://placehold.co/20x20/999999/999999"/> object</td>
    <td align="center"><img width="20" height="20" src="https://placehold.co/20x20/6b8e23/6b8e23"/> nature</td>
    <td align="center"><img width="20" height="20" src="https://placehold.co/20x20/4682b4/4682b4"/> sky</td>
    <td align="center"><img width="20" height="20" src="https://placehold.co/20x20/dc143c/dc143c"/> human</td>
    <td align="center"><img width="20" height="20" src="https://placehold.co/20x20/00008e/00008e"/> vehicle</td>
  </tr>
</table>
