# LHYP

A kiinduló forrás tartalmaz kódot a con filok beolvasására valamint a rövid tengely (sa) képek olvasására. Ki is lehet próbálni, van egy kis példa program hozzá, ami rárajzolja a kontúrt a képre. (A saját gépen érdemes futtatni a scriptet, mivel a szerver nem tud képet megjeleníteni.)

Mindenki fork-olja ezt a repót, majd készítsen egy saját branch-et. Időnként, mikor találunk valami érdekeset, szinkronizáljuk majd az egyes repókat.

A nano szerveren elérhetők példa adatok, amik alapján az adatbeolvasás és később a modellhez az adatbetöltés megírható. Javasolt a modellhez szükséges adatok (minták) pickle fájlba írása. Egy pácienshez egy pickle, minden szükséges információval, ami szükséges róla.

# Önlab összefoglalás

## Data

Input size: 224x224<br>
Normalization: mean=0.449, std=0.226

### SA:

- channels: 6 (2 systole, 2 diastole, 2 inbetween)
- ROI: largest bounding box from contours + 5px padding around

### SALE:

- channels: lowest across all samples (4)

## Optimizer

SGD: LR=0.01, momentum=0.9

## LR scheduler

ReduceLROnPlateau: factor=10, patience=10

## Loss function

BCEWithLogitLoss

## Architectures

SimmCNN: Batch size: 256 [impl]()<br>
LongCNN: BS: 256 [impl]()<br>
ResNet18: BS: 256 [PyTorch impl](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)<br>
ResNet34: BS: 164 [PyTorch impl](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

## Results

### Test set accuracy:

![Accuracy](ml\statistics\test_results.png)

### Loss plots (red line=lowest val loss=last saved checkpoint):

SimmCNN:
| SA | SALE |
| :---: | :---: |
|![SimmCNN SA loss](ml\statistics\loss\SimmCNN_bs256_lr0.01_sa_loss.png) | ![SimmCNN SALE loss](ml\statistics\loss\SimmCNN_bs256_lr0.01_sa_loss.png)|

LongCNN:
| SA | SALE |
| :---: | :---: |
|![LongCNN SA loss](ml\statistics\loss\LongCNN_bs256_lr0.01_sa_loss.png) | ![LongCNN SALE loss](ml\statistics\loss\LongCNN_bs256_lr0.01_sa_loss.png)|

ResNet18:
| SA | SALE |
| :---: | :---: |
| ![ResNet18 SA loss](ml\statistics\loss\ResNet18_bs256_lr0.01_sa_loss.png) | ![ResNet18 SALE loss](ml\statistics\loss\ResNet18_bs256_lr0.01_sa_loss.png) |

ResNet34:
| SA | SALE |
| :---: | :---: |
| ![ResNet34 SA loss](ml\statistics\loss\ResNet34_bs164_lr0.01_sa_loss.png) | ![ResNet34 SALE loss](ml\statistics\loss\ResNet34_bs164_lr0.01_sa_loss.png) |
