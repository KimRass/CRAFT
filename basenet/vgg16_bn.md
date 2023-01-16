# `vgg16_bn`
## `__init__`
- `torchvision` 0.13 이상부터 `torchvision.model`의 여러 모델들에 대해서 `pretrained` parameter가 없어지고 `weights` parameter가 새로 생겼습니다. 따라서 `DeprecationWarning`을 방지하기 위해 코드를 수정할 필요가 있습니다.
- References: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/model/modules.py, https://pytorch.org/vision/stable/models.html
- As-is
    ```python
    super(vgg16_bn, self).__init__()
    model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
    vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
    ```
- To-be
    ```python
    super(vgg16_bn, self).__init__()
    if version.parse(torchvision.__version__) >= version.parse("0.13"):
        vgg_pretrained_features = models.vgg16_bn(
            weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
        ).features
    else:
        model_urls["vgg16_bn"] = model_urls["vgg16_bn"].replace("https://", "http://")
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
    ```