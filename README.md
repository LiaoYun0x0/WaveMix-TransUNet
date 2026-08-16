# WaveMix-TransUNet

PyTorch training code for WaveMix-TransUNet on the ISPRS Vaihingen dataset.

## File structure

```text
WaveMix-TransUNet/
|-- model/
|   |-- r50_vit_encoder.py
|   `-- wavemix_transunet.py
|-- train/
|   `-- train_vaihingen.py
|-- utils.py
|-- requirements.txt
`-- README.md
```

## Environment

Install the PyTorch build that matches the CUDA version on the training
server, and then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Vaihingen dataset

Place the Vaihingen files in the following structure:

```text
WaveMix-TransUNet/
`-- ISPRS_dataset/
    `-- Vaihingen/
        |-- top/
        |   |-- top_mosaic_09cm_area1.tif
        |   |-- top_mosaic_09cm_area3.tif
        |   `-- ...
        |-- dsm/
        |   |-- dsm_09cm_matching_area1.tif
        |   |-- dsm_09cm_matching_area3.tif
        |   `-- ...
        |-- gts_for_participants/
        |   |-- top_mosaic_09cm_area1.tif
        |   |-- top_mosaic_09cm_area3.tif
        |   `-- ...
        `-- gts_eroded_for_participants/
            |-- top_mosaic_09cm_area1_noBoundary.tif
            |-- top_mosaic_09cm_area3_noBoundary.tif
            `-- ...
```

The training script uses the following split:

- Training areas: `1, 3, 23, 26, 7, 11, 13, 28, 17, 32, 34, 37`
- Validation areas: `5, 21, 15, 30`

## Pretrained weights

Place the TransUNet `R50+ViT-B_16.npz` pretrained weights in:

```text
WaveMix-TransUNet/
`-- pretrain/
    `-- R50+ViT-B_16.npz
```

## Start training

Run the following command from the `WaveMix-TransUNet` directory:

```bash
python train/train_vaihingen.py \
  --data-root ./ISPRS_dataset/Vaihingen \
  --pretrained ./pretrain/R50+ViT-B_16.npz
```
