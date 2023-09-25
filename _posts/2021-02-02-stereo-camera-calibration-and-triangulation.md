---
layout: post
title:  "Title"
date:   2023-09-25 10:00:00 +0000
categories: Category1 Category2
---

You can add a link as follows: [Joel Mwaka](https://github.com/joelmwaka)

Adding image:

<p align="center">
  <img src="https://github.com/TemugeB/temugeb.github.io/blob/main/_posts/images/triangulate.png?raw=true">
</p>

List:

  1. Point 1.
  2. Point 2.
  3. Point 3.

**Subtitle**

Add python code:

```python
import cv2 as cv
import glob
import numpy as np
 
images_folder = 'D2/*'
images_names = sorted(glob.glob(images_folder))
images = []
for imname in images_names:
    im = cv.imread(imname, 1)
    images.append(im)    

```

**References**

1. Ref 1.
2. Ref 2.
3. Ref 3.
