import cv2
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
# import settings


def each_char(image_anno):
    for block in image_anno['annotations']:
        for char in block:
            yield char


with open("/Users/jongbeomkim/Downloads/ctw-annotations/train.jsonl") as f:
open("/Users/jongbeomkim/Downloads/ctw-annotations/train.jsonl").readline()
    anno = json.loads(f.readline())
anno["file_name"]
anno.keys()


# path = os.path.join(settings.TRAINVAL_IMAGE_DIR, anno['file_name'])
# assert os.path.exists(path), 'file not exists: {}'.format(path)
# img = cv2.imread(path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = load_image("/Users/jongbeomkim/Downloads/ctw-trainval-01-of-26/0000174.jpg")

plt.figure(figsize=(10, 10))
ax = plt.gca()
plt.imshow(img)
for instance in each_char(anno):
    instance
    color = (0, 1, 0) if instance['is_chinese'] else (1, 0, 0)
    ax.add_patch(
        patches.Polygon(instance['polygon'], fill=False, color=color)
    )
plt.show()

instance['polygon']

imp
np.random.multivariate_normal(mean=(0, 0), cov=np.array([[10, 10], [20, 20]]), size=10000).shape

