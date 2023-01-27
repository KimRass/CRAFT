# What is 'CRAFT'?
- [CRAFT: Character-Region Awareness For Text detection](https://github.com/clovaai/CRAFT-pytorch)

# Directory Structure
```
CRAFT-pytorch
├── LICENSE (x)
├── README.md (x)
├── basenet
│   ├── __init__.py (x)
│   └── vgg16_bn.py (o)
├── craft.py (o)
├── craft_utils.py (o)
├── figures
│   └── craft_example.gif (x)
├── file_utils.py (x)
├── imgproc.py (o)
├── refinenet.py (o)
├── requirements.txt (x)
└── test.py (o)
```

# Model Architecture
- Model architecture
  - <img src="https://miro.medium.com/max/1400/1*b6I-Bdj5itX7tllJ5HRKbg.png" width="400">
```
CRAFT(
  (basenet): vgg16_bn(
    (slice1): Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (9): ReLU(inplace=True)
      (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (slice2): Sequential(
      (12): ReLU(inplace=True)
      (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (16): ReLU(inplace=True)
      (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (slice3): Sequential(
      (19): ReLU(inplace=True)
      (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (22): ReLU(inplace=True)
      (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (26): ReLU(inplace=True)
      (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (slice4): Sequential(
      (29): ReLU(inplace=True)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (32): ReLU(inplace=True)
      (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (36): ReLU(inplace=True)
      (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (slice5): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      (1): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
      (2): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (upconv1): double_conv(
    (conv): Sequential(
      (0): Conv2d(1536, 512, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
  (upconv2): double_conv(
    (conv): Sequential(
      (0): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
  (upconv3): double_conv(
    (conv): Sequential(
      (0): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
  (upconv4): double_conv(
    (conv): Sequential(
      (0): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
    )
  )
  (conv_cls): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): ReLU(inplace=True)
    (6): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(16, 2, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

# Paper Review
- Paper: [Character Region Awareness for Text Detection](https://arxiv.org/pdf/1904.01941.pdf)
- Reference: https://medium.com/@msmapark2/character-region-awareness-for-text-detection-craft-paper-%EB%B6%84%EC%84%9D-da987b32609c
- OCR을 Character-level로 수행합니다.
- Input으로 주어진 이미지의 각 픽셀마다 "Region score"와 "Affinity score"를 예측합니다.
  - Region score: 해당 픽셀이 어떤 문자의 중심일 확률.
  - Affiniy score: 해당 픽셀이 어떤 문자와 다른 어떤 문자의 중심일 확률. 이것을 통해 여러 문자들을 묶어서 하나의 텍스트로 인식할지 여부가 결정됩니다.
- 다음의 세 가지를 예측할 수 있습니다; Character boxes, Word boxes, Polygons
- Inference stage: Character region을 바탕으로 위에서 말한 세 가지를 추론하는 단계
  - Word-level QuadBox (word-level bounding box) Inference
    - Polygon Inference
      - <img src="https://miro.medium.com/max/1400/1*_EyygIYQyQPqUk-w-OaKjw.png" width="400">
      - "Local maxima along scanning direction" (Blue. "Control points of text polygon"의 후보들) -> "Center line of local maxima" (Yellow) -> "Line of control points" (Red. 문자 기울기 반영) -> 양 끝에 있는 문자들을 커버하기 위해 그들에 대한 "Control points of text polygon"을 정하고 최종적으로 "Polygin text instance"를 정함

# Score Maps
- Text score map | Link score map | Line score map
  - <img src="https://i.imgur.com/g2xxnuI.jpg" width="1000">

# Text Score Map Difference by Colors from Image
- Text score map from original image | Text score map from inverted image
  - <img src="https://i.imgur.com/6EnenSj.jpg" width="800">
- 이미지를 반전한 후에 콤마 (';')를 더 잘 Detect하는 것을 볼 수 있습니다. 아마 학습에 사용된 이미지들에서 하얀색보다는 검은색의 텍스트가 더 많이 존재했기 때문일 것으로 추측됩니다.

# Processing Super High Resolution Images
- 해상도가 매우 큰 이미지는 그 자체로는 메모리 부족 등의 이유로 CRAFT를 사용하여 Infer할 수 없습니다. 따라서 이미지를 적당한 크기로 분할하여 처리해야 합니다.
- 단 이미지를 분할할 때 하나의 문자가 둘 이상으로 나뉘어질 수 있습니다. 이렇게 되면 Text detection이 제대로 수행될 수 없으므로 유의해야 합니다.
- Super high resolution image sample (7,256 x 13,483)
  - <img src="https://i.imgur.com/w0ELNTk.jpg" width="300">
## Algorithm
1. 이미지를 절반의 해상도로 분할하되 다음과 같이 3개로 분리합니다. 이를 통해 하나의 문자가 둘 이상으로 분리되어 Detect되지 못하는 것을 방지할 수 있습니다. 이들 각각을 CRAFT를 사용하여 Infer하고 그 결과들을 합칩니다.
  - Image splitting (Red -> Green -> Blue)
    - <img src="https://i.imgur.com/9Gnmet6.jpg" width="300">
2. 분할된 이미지가 지정된 해상도보다 크다면 각각을 다시 분할합니다. 지정된 해상도 미만이 될 때까지 이것을 반복하여 수행합니다.
- Text score map
  - <img src="https://i.imgur.com/ZbXWURG.jpg" width="300">
