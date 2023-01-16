# `copyStateDict`
# `test_net`
```python
img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
```
- CRAFT의 Input으로 넣기 위해 이미지를 Pad합니다.
```python
# forward pass
with torch.no_grad():
    y, feature = net(x)

# make score and link map
score_text = y[0,:,:,0].cpu().data.numpy()
score_link = y[0,:,:,1].cpu().data.numpy()

# refine link
if refine_net is not None:
    with torch.no_grad():
        y_refiner = refine_net(y, feature)
    score_link = y_refiner[0,:,:,0].cpu().data.numpy()
```
- CRAFT의 Output으로 `y`와  `feature`라는 변수가 생성됩니다. `feature`는 CRAFT Refiner의 Input으로만 사용됩니다.
- `refine_net=None`을 사용하면 (CRAFT Refiner를 사용하지 않으면) Text score map (`score_text`)와 Link score map (`score_link`)가 산출되고, CRAFT Refiner를 사용하면 `score_link`가 Line score map이 되어 산출됩니다.