# 3D-CNNs-Model-Hub

This repository includes 3D implementations of commonly used 2D CNNs, filling the gap of readily available 3D CNN models in most deep learning libraries.

## Libraries Needed:
```ruby
* Python 3.x
* Tensorflow 2.X
* Numpy
* random
```

## 3D-Models Available:

### Classification:
```ruby
* Resnet_3D.py
* DenseNet_3D.py
* VGG_3D.py
* Inception_3D.py
```
### Segmentation:
```ruby
* DenseVnet3D.py
* Unet3D.py
```

References for the above are available at:

* DenseVnet3D.py -> (https://github.com/myubabe/DenseVNet3D_Chest_Abdomen_Pelvis_Segmentation_tf2/)
  
* Unet3D.py-> (https://github.com/myubabe/3DUnet_tensorflow2.0/)

### How to Run
Configure the models based on your needs and GPU's running capability using *config.py*:
```ruby
##-----Network Configuration----#