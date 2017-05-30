Deep Cascade of Convolutional Neural Networks for MR Image Reconstruction
=========================================================================

Reconstruct MR images from its undersampled measurements using Deep Cascade of
Convolutional Neural Networks. This repository contains the implementation of
DC-CNN using Theano and Lasagne and the simple demo on toy dataset borrowed from
<http://mridata.org>. Note that the library requires the dev version of Lasagne
and Theano, as well as pygpu backend for using CUFFT Library.

Usage::

  python main_2d.py --num_epoch 5 --batch_size 2 


----

If you used this code for your work, you must cite the following work::

  Schlemper, J., Caballero, J., Hajnal, J. V., Price, A., & Rueckert, D. A Deep Cascade of Convolutional Neural Networks for MR Image Reconstruction. Information Processing in Medical Imaging (IPMI), 2017

----

The paper is also available on arXiv: <https://arxiv.org/pdf/1703.00555.pdf>
