Deep Cascade of Convolutional Neural Networks for MR Image Reconstruction
=========================================================================

Reconstruct MR images from its undersampled measurements using Deep Cascade of
Convolutional Neural Networks (DC-CNN). This repository contains the
implementation of DC-CNN using Theano and Lasagne and the simple demo. Note that
the library requires the dev version of Lasagne and Theano, as well as pygpu
backend for using CUFFT Library. Some of the toy dataset borrowed from
<http://mridata.org>.

1. 2D Reconstruction
====================

Usage::

  python main_2d.py --num_epoch 5 --batch_size 2 


----


2. Dynamic Reconstruction
=========================================================================

Reconstruct dynamic MR images from its undersampled measurements using DC-CNN
with Data Sharing layer. Note that the library requires CUDNN in addition to the
requirement specified above.

Usage::

  python main_3d.py --acceleration_factor 4


----


Citation and Acknowledgement
============================

If you use the code for your work, or if you found the code useful, please cite the following works.

2D Reconstruction::

  Schlemper, J., Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D. A Deep Cascade of Convolutional Neural Networks for MR Image Reconstruction. Information Processing in Medical Imaging (IPMI), 2017

----

The paper is also available on arXiv: <https://arxiv.org/pdf/1703.00555.pdf>


Dynamic Reconstruction::

  Schlemper, J., Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D. A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction. ArXiv 1704.02422

----

The paper is also available on arXiv: <https://arxiv.org/pdf/1704.02422.pdf>
