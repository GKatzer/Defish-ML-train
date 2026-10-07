# Frozen embeddings vs deployed YOLO-cls

Unique crops: 187 (train 128, eval 59).

## A. Matched split (train on 128 unique train crops, test on unique valid/test crops)

| system | subset | task | n | acc | acc_ci | bal_acc | macro_f1 | sick_recall | healthy_recall |
|---|---|---|---|---|---|---|---|---|---|
| YOLO-cls deployed (server preproc) | all eval | 9-class | 59 | 0.22 | [0.13, 0.34] | 0.36 | 0.20 | 0.81 | 0.12 |
| YOLO-cls deployed (server preproc) | all eval | 7-class | 45 | 0.29 | [0.18, 0.43] | 0.34 | 0.22 | 0.79 | 0.38 |
| YOLO-cls deployed (native preproc) | all eval | 9-class | 59 | 0.36 | [0.25, 0.48] | 0.47 | 0.34 | 0.93 | 0.31 |
| YOLO-cls deployed (native preproc) | all eval | 7-class | 45 | 0.38 | [0.25, 0.52] | 0.42 | 0.36 | 0.90 | 0.44 |
| dinov2_base_224 + LR | all eval | 9-class | 59 | 0.59 | [0.47, 0.71] | 0.47 | 0.43 | 0.84 | 0.75 |
| dinov2_base_224 + LR | all eval | 7-class | 45 | 0.51 | [0.37, 0.65] | 0.46 | 0.42 | 0.79 | 0.75 |
| dinov2_base_448 + LR | all eval | 9-class | 59 | 0.58 | [0.45, 0.69] | 0.47 | 0.44 | 0.86 | 0.62 |
| dinov2_base_448 + LR | all eval | 7-class | 45 | 0.53 | [0.39, 0.67] | 0.49 | 0.44 | 0.90 | 0.69 |
| bioclip_224 + LR | all eval | 9-class | 59 | 0.49 | [0.37, 0.62] | 0.37 | 0.36 | 0.74 | 0.69 |
| bioclip_224 + LR | all eval | 7-class | 45 | 0.40 | [0.27, 0.55] | 0.37 | 0.35 | 0.69 | 0.62 |
| YOLO-cls deployed (server preproc) | eval, photo not in train | 9-class | 30 | 0.13 | [0.05, 0.30] | 0.28 | 0.13 | 0.86 | 0.00 |
| YOLO-cls deployed (server preproc) | eval, photo not in train | 7-class | 29 | 0.21 | [0.10, 0.38] | 0.25 | 0.16 | 0.75 | 0.22 |
| YOLO-cls deployed (native preproc) | eval, photo not in train | 9-class | 30 | 0.37 | [0.22, 0.54] | 0.47 | 0.34 | 1.00 | 0.22 |
| YOLO-cls deployed (native preproc) | eval, photo not in train | 7-class | 29 | 0.38 | [0.23, 0.56] | 0.40 | 0.38 | 0.90 | 0.33 |
| dinov2_base_224 + LR | eval, photo not in train | 9-class | 30 | 0.50 | [0.33, 0.67] | 0.52 | 0.46 | 0.86 | 0.56 |
| dinov2_base_224 + LR | eval, photo not in train | 7-class | 29 | 0.48 | [0.31, 0.66] | 0.45 | 0.40 | 0.80 | 0.56 |
| dinov2_base_448 + LR | eval, photo not in train | 9-class | 30 | 0.50 | [0.33, 0.67] | 0.52 | 0.46 | 0.86 | 0.56 |
| dinov2_base_448 + LR | eval, photo not in train | 7-class | 29 | 0.48 | [0.31, 0.66] | 0.45 | 0.40 | 0.95 | 0.56 |
| bioclip_224 + LR | eval, photo not in train | 9-class | 30 | 0.40 | [0.25, 0.58] | 0.42 | 0.42 | 0.62 | 0.67 |
| bioclip_224 + LR | eval, photo not in train | 7-class | 29 | 0.38 | [0.23, 0.56] | 0.34 | 0.34 | 0.65 | 0.56 |


## B. Stratified group 4-fold CV x5 seeds over all unique crops (LR, C=1)

| embedder | task | n | acc | bal_acc | macro_f1 | acc_sd |
|---|---|---|---|---|---|---|
| dinov2_base_224 | 9-class | 187 | 0.63 | 0.60 | 0.58 | 0.01 |
| dinov2_base_224 | 7-class | 138 | 0.57 | 0.58 | 0.56 | 0.03 |
| dinov2_base_224 | healthy-vs-sick | 138 | 0.81 | 0.74 | 0.76 | 0.02 |
| dinov2_base_448 | 9-class | 187 | 0.60 | 0.56 | 0.53 | 0.01 |
| dinov2_base_448 | 7-class | 138 | 0.58 | 0.59 | 0.56 | 0.03 |
| dinov2_base_448 | healthy-vs-sick | 138 | 0.81 | 0.75 | 0.76 | 0.01 |
| bioclip_224 | 9-class | 187 | 0.52 | 0.46 | 0.45 | 0.02 |
| bioclip_224 | 7-class | 138 | 0.42 | 0.45 | 0.43 | 0.02 |
| bioclip_224 | healthy-vs-sick | 138 | 0.71 | 0.65 | 0.65 | 0.02 |


## C1. Gold-trained model (7 classes) applied to external panda992 images
`predicted_healthy` = share of images the model calls healthy. For `healthy_fish` it should be high.

| embedder | silver_class | n | predicted_healthy | top_pred |
|---|---|---|---|---|
| dinov2_base_224 | bacterial_diseases_aeromoniasis | 250 | 0.43 | healthy |
| dinov2_base_224 | bacterial_gill_disease | 249 | 0.29 | hexamitosis |
| dinov2_base_224 | bacterial_red_disease | 248 | 0.42 | healthy |
| dinov2_base_224 | fungal_diseases_saprolegniasis | 250 | 0.54 | healthy |
| dinov2_base_224 | healthy_fish | 250 | 0.54 | healthy |
| dinov2_base_224 | parasitic_diseases | 249 | 0.55 | healthy |
| dinov2_base_224 | viral_diseases_white_tail_disease | 250 | 0.42 | healthy |
| dinov2_base_448 | bacterial_diseases_aeromoniasis | 250 | 0.14 | hexamitosis |
| dinov2_base_448 | bacterial_gill_disease | 249 | 0.09 | hexamitosis |
| dinov2_base_448 | bacterial_red_disease | 248 | 0.20 | hexamitosis |
| dinov2_base_448 | fungal_diseases_saprolegniasis | 250 | 0.25 | dermatomycosis |
| dinov2_base_448 | healthy_fish | 250 | 0.48 | healthy |
| dinov2_base_448 | parasitic_diseases | 249 | 0.29 | hexamitosis |
| dinov2_base_448 | viral_diseases_white_tail_disease | 250 | 0.29 | fin_rot |
| bioclip_224 | bacterial_diseases_aeromoniasis | 250 | 0.12 | oodiniosis |
| bioclip_224 | bacterial_gill_disease | 249 | 0.15 | oodiniosis |
| bioclip_224 | bacterial_red_disease | 248 | 0.21 | mycobacteriosis |
| bioclip_224 | fungal_diseases_saprolegniasis | 250 | 0.15 | oodiniosis |
| bioclip_224 | healthy_fish | 250 | 0.49 | healthy |
| bioclip_224 | parasitic_diseases | 249 | 0.22 | oodiniosis |
| bioclip_224 | viral_diseases_white_tail_disease | 250 | 0.09 | oodiniosis |


## C2. Does adding silver data help? Group CV on gold, silver added to training folds only

| embedder | config | acc | bal_acc | macro_f1 | sick_recall | healthy_recall |
|---|---|---|---|---|---|---|
| dinov2_base_224 | gold only | 0.57 | 0.58 | 0.56 | 0.83 | 0.66 |
| dinov2_base_224 | +healthy x100 | 0.59 | 0.60 | 0.57 | 0.86 | 0.66 |
| dinov2_base_224 | +healthy x100 +fungal->dermatomycosis x100 | 0.57 | 0.60 | 0.57 | 0.87 | 0.60 |
| dinov2_base_448 | gold only | 0.58 | 0.59 | 0.56 | 0.85 | 0.68 |
| dinov2_base_448 | +healthy x100 | 0.59 | 0.60 | 0.57 | 0.87 | 0.66 |
| dinov2_base_448 | +healthy x100 +fungal->dermatomycosis x100 | 0.57 | 0.59 | 0.57 | 0.87 | 0.61 |
| bioclip_224 | gold only | 0.42 | 0.45 | 0.43 | 0.77 | 0.51 |
| bioclip_224 | +healthy x100 | 0.40 | 0.43 | 0.41 | 0.74 | 0.47 |
| bioclip_224 | +healthy x100 +fungal->dermatomycosis x100 | 0.41 | 0.44 | 0.43 | 0.80 | 0.50 |


### Confusion matrix (seed 0, out-of-fold): dinov2_base_448, 9-class

```
pred             dermatomycosis  fin_rot  healthy  hexamitosis  many_fish  mycobacteriosis  not_a_fish  oodiniosis  plistophorosis
true                                                                                                                              
dermatomycosis                5        2        1            1          0                2           0           1               1
fin_rot                       2       15        4            1          0                2           0           0               0
healthy                       2        4       24            2          2                2           3           2               1
hexamitosis                   2        1        1           11          0                2           0           0               0
many_fish                     0        0        6            0          2                0           0           0               1
mycobacteriosis               2        5        4            1          0                8           0           1               0
not_a_fish                    0        0        2            0          0                0          36           0               2
oodiniosis                    1        2        3            2          0                0           0           6               1
plistophorosis                0        0        0            0          0                0           0           0               6
```

### Confusion matrix (seed 0, out-of-fold): dinov2_base_448, 7-class

```
pred             dermatomycosis  fin_rot  healthy  hexamitosis  mycobacteriosis  oodiniosis  plistophorosis
true                                                                                                       
dermatomycosis                4        1        3            0                1           3               1
fin_rot                       0       14        4            0                6           0               0
healthy                       2        4       29            2                3           2               0
hexamitosis                   3        1        2           11                0           0               0
mycobacteriosis               0        6        4            0                9           1               1
oodiniosis                    1        1        1            2                2           6               2
plistophorosis                0        0        0            0                0           0               6
```
