```python
class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
```
```python
class vgg16_bn(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        if version.parse(torchvision.__version__) >= version.parse("0.13"):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else:
            model_urls["vgg16_bn"] = model_urls["vgg16_bn"].replace("https://", "http://")
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
```
- Reference: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/model/modules.py