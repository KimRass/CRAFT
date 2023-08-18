# `normalizeMeanVariance`
- Numpy array 형태의 이미지를 Normalize합니다.
- 다음으로 대체가 가능합니다. 단 Pytorch tensor로 변환됩니다.
    ```python
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    z = transform(img)
    ```

# `resize_aspect_ratio`
```python
height, width, channel = img.shape

# magnify image size
target_size = mag_ratio * max(height, width)

# set original image size
if target_size > square_size:
    target_size = square_size

ratio = target_size / max(height, width)    

target_h, target_w = int(height * ratio), int(width * ratio)
proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)
```
- Line number 38 ~ 50
- 이 부분은 `square_size`와 `mag_ratio`에 따라 이미지의 가로와 세로 비율을 그대로 유지한 채 크기를 줄입니다.
- 그런데 이 코드가 정확히 왜 필요한지 잘 모르겠습니다. 사용하지 않아도 Text detection 성능에는 전혀 지장이 없는 것으로 보입니다.
```python
# make canvas and paste image
target_h32, target_w32 = target_h, target_w
if target_h % 32 != 0:
    target_h32 = target_h + (32 - target_h % 32)
if target_w % 32 != 0:
    target_w32 = target_w + (32 - target_w % 32)
resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
resized[0:target_h, 0:target_w, :] = proc
target_h, target_w = target_h32, target_w32

size_heatmap = (int(target_w/2), int(target_h/2))
```
- Line number 53 ~ 63
- 이미지의 가로와 세로 각각을 32의 배수가 되도록 Zero pad합니다.

# `cvt2HeatmapImg`
- 1-channel 이미지의 Colormap을 변환합니다. 값이 큰 픽셀은 빨간색으로, 작은 픽셀은 파란색으로 나타냅니다.