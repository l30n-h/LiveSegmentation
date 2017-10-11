#Install caffe via docker:
```bash
sudo docker build --memory=2g --memory-swap=-1 -t caffe:cpu .
```

#Start caffe via docker:
```bash
sudo docker run --rm -it -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) --memory=2g --memory-swap=-1 caffe:cpu bash
```

#Install src:
```bash
cd src && cmake . && make && cd ..
```

#Used Models:
https://github.com/shelhamer/fcn.berkeleyvision.org

#Pretrained caffemodel:
http://dl.caffe.berkeleyvision.org/nyud-fcn32s-color-heavy.caffemodel

#Test (c++):
```bash
src/nyu_classification models/deploy.prototxt models/pretrained.caffemodel "116.190" "97.203" "92.318" data/cafe1a.ppm
```

#Test (ipython):
ipython

```python
import numpy as np
from PIL import Image
import caffe

MODEL_FILE = 'models/deploy.prototxt'
PRETRAINED = 'models/pretrained.caffemodel'
IMAGE_FILE = 'data/cafe1a.ppm'
caffe.set_mode_cpu()

im = Image.open(IMAGE_FILE)
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((116.190, 97.203, 92.318), dtype=np.float32)
in_ = in_.transpose((2,0,1))

net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

out = net.forward()

np.save('out', out)

Image.fromarray((out['loss'][0].argmax(axis=0)).astype('uint8')).save('class.png')
Image.fromarray((out['loss'][0].max(axis=0)*100).astype('uint8')).save('prob.png')

quit()
```
