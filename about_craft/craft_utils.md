# `getDetBoxes_core`
```python
ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
```
- `low_text`: Text score_map을 Text mask로 변환할 때의 Threshold를 의미합니다. 이 값이 작을수록 Text mask의 영역의 넓이는 증가합니다.
- `link_threshold`: Link score_map을 Link mask로 변환할 때의 Threshold를 의미합니다. 이 값이 작을수록 Link mask의 영역의 넓이는 증가합니다.
```python
text_score_comb = np.clip(text_score + link_score, 0, 1)
```
- Text mask와 Link mask을 합하여 단어 단위의 Word mask를 생성합니다.
```python
nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
```
```python
# size filtering
size = stats[k, cv2.CC_STAT_AREA]
if size < 10: continue
```
- `stats`는 
- `cv2.CC_STAT_AREA`는 `4`와 같습니다. `stats`의 Index 4에 해당하는 컬럼은 각 Label별 Pixel count를 나타냅니다. 즉 Pixel count 10 이하의 너무 작은 영역을 제거합니다.
```python
# thresholding
if np.max(textmap[labels==k]) < text_threshold: continue
```
- `text_threshold`: Text score map에서 Label `k`에 해당하는 Region만을 봅시다. 이 Region 내에서 Maximum text score가 `text_threshold`가 작다면 그 Region은 무시합니다. 즉 Text detection에 있어서 각 텍스트가 `text_threshold` 이상의 Text score를 최대값으로 가질 경우 `low_text`를 Threshold로 하는 Region을 그 텍스트가 차지하는 공간이라고 할 수 있습니다.
```python
# make segmentation map
segmap = np.zeros(textmap.shape, dtype=np.uint8)
segmap[labels==k] = 255
segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
```
1. `textmap`과 같은 Shape을 갖는 `segmap`을 생성합니다. 이것은 일종의 검은색 Canvas 역할을 합니다.
2. `labels==`k를 만족하는 Region만 255 (White)의 값을 갖도록 합니다.
3. Link score가 1이고 Text score가 0인 Region (완전한 Link를 나타내는 픽셀)은 제외합니다.
```python
x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
# boundary check
if sx < 0 : sx = 0
if sy < 0 : sy = 0
if ex >= img_w: ex = img_w
if ey >= img_h: ey = img_h
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
```
- `stats`
- `cv2.CC_STAT_LEFT`, `cv2.CC_STAT_TOP`, `cv2.CC_STAT_WIDTH`, `cv2.CC_STAT_HEIGHT`는 각각 `0`, `1`, `2`, `3`과 같습니다.
- Word mask (`text_score_comb`)의 서로 연결되지 않은 각 영역에 대해서 그 영역을 최소 넓이로 포함하고 두 변이 축에 평행한 직사각형을 나타냅니다.
- `niter`는 일종의 Margin입니다. `niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)`의 2에 더 큰 숫자를 넣으면 Margin이 커져서 Word mask의 각 Region을 여유있게 포함하도록 직사각형의 크기가 커집니다.
- `sx`, `sy`, `ex`, `ey`는 x좌표와 y좌표가 이미지를 벗어나지 않도록 하기 위한 변수입니다.