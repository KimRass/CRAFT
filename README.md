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

# CRAFT (Character Region Awareness For Text detection)
- Paper: https://arxiv.org/abs/1904.01941
- References: https://github.com/clovaai/CRAFT-pytorch, https://github.com/LoveGalaxy/Character-Region-Awareness-for-Text-Detection-, https://github.com/ducanh841988/Kindai-OCR, https://medium.com/@msmapark2/character-region-awareness-for-text-detection-craft-paper-%EB%B6%84%EC%84%9D-da987b32609c
- OCR을 Character-level로 수행합니다.
- Input으로 주어진 이미지의 각 픽셀마다 "Region score"와 "Affinity score"를 예측합니다.
  - Region score: 해당 픽셀이 어떤 문자의 중심일 확률.
  - Affiniy score: 해당 픽셀이 어떤 문자와 다른 어떤 문자의 중심일 확률. 이것을 통해 여러 문자들을 묶어서 하나의 텍스트로 인식할지 여부가 결정됩니다.
- 다음의 세 가지를 예측할 수 있습니다; Character boxes, Word boxes, Polygons
- Inference stage: Character region을 바탕으로 위에서 말한 세 가지를 추론하는 단계
  - Word-level QuadBox (word-level bounding box) Inference
    - Polygon Inference
      - ![polygon_inference](https://miro.medium.com/max/1400/1*_EyygIYQyQPqUk-w-OaKjw.png)
      - "Local maxima along scanning direction" (Blue. "Control points of text polygon"의 후보들) -> "Center line of local maxima" (Yellow) -> "Line of control points" (Red. 문자 기울기 반영) -> 양 끝에 있는 문자들을 커버하기 위해 그들에 대한 "Control points of text polygon"을 정하고 최종적으로 "Polygin text instance"를 정함
- Architecture
  - ![architecture](https://miro.medium.com/max/1400/1*b6I-Bdj5itX7tllJ5HRKbg.png)

# Score Maps
## Line Score Map
- ![036_line_score_map](https://i.imgur.com/4jQnSpN.png)