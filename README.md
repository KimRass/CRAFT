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

# Paper Summary
- Paper: [Character Region Awareness for Text Detection](https://arxiv.org/pdf/1904.01941.pdf)
- Reference: https://medium.com/@msmapark2/character-region-awareness-for-text-detection-craft-paper-%EB%B6%84%EC%84%9D-da987b32609c
## Character-level Awareness;
  - These methods mainly train their networks to localize wordlevel bounding boxes. However, they may suffer in difficult cases, such as texts that are curved, deformed, or extremely long, which are hard to detect with a single bounding box.
  - Alternatively, character-level awareness has many advantages when handling challenging texts by linking the successive characters in a bottom-up manner.
  - In this paper, we propose a novel text detector that localizes the individual character regions and links the detected characters to a text instance.
  - Most methods detect text with words as its unit, but defining the extents to a word for detection is non-trivial since words can be separated by various criteria, such as meaning, spaces or color. In addition, the boundary of the word segmentation cannot be strictly defined, so the word segment itself has no distinct semantic meaning. This ambiguity in the word annotation dilutes the meaning of the ground truth for both regression and segmentation approaches.
## Train
- Unfortunately, most of the existing text datasets do not provide characterlevel annotations, and the work needed to obtain characterlevel ground truths is too costly.
## Architecture
- Our framework, referred to as CRAFT for Character Region Awareness For Text detection, is designed with a convolutional neural network producing the character region score and affinity score. The region score is used to localize individual characters in the image, and the affinity score is used to group each character into a single instance.
- The final output has two channels as score maps: the region score and the affinity score.
## Inference
- The final output has two channels as score maps: the region score and the affinity score.
- *At the inference stage, the final output can be delivered in various shapes, such as word boxes or character boxes, and further polygons*.
- Inference
  - <img src="https://miro.medium.com/max/1400/1*_EyygIYQyQPqUk-w-OaKjw.png" width="400">
## Link Refinement
- In the CTW-1500 dataset’s case, two difficult characteristics coexist, namely annotations that are provided at the line-level and are of arbitrary polygons. To aid CRAFT in such cases, a small link refinement network, which we call the LinkRefiner, is used in conjunction with CRAFT.
- *The input of the LinkRefiner is a concatenation of the region score, the affinity score, and the intermediate feature map of CRAFT, and the output is a refined affinity score adjusted for long texts. To combine characters, the refined affinity score is used instead of the original affinity score*, then the polygon generation is performed in the same way as it was performed for TotalText.
- Only LinkRefiner is trained on the CTW-1500 dataset while freezing CRAFT. The detailed implementation of LinkRefiner is addressed in the supplementary materials. As shown in Table 2, the proposed method achieves state-of-the-art performance.
- Atrous Spatial Pyramid Pooling (ASPP) in is adopted to ensure a large receptive field for combining distant characters and words onto the same text line.

# Region Score Map Difference by Colors from Image
- Region score map from original image | Region score map from inverted image
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
- Region score map
  - <img src="https://i.imgur.com/ZbXWURG.jpg" width="300">
