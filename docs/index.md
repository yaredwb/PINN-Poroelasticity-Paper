---
layout: default
title: Physics-informed Deep Learning for Flow and Deformation in Poroelastic Media
mathjax: true
---

#### Author: [Yared W. Bekele](https://yaredwb.github.io/)


### Abstract

A physics-informed neural network is presented for poroelastic problems with coupled flow and deformation processes. The governing equilibrium and mass balance equations are discussed and specific derivations for two-dimensional cases are presented. A fully-connected deep neural network is used for training. Barry and Mercer's source problem with time-dependent fluid injection/extraction in an idealized poroelastic medium, which has an exact analytical solution, is used as a numerical example. A random sample from the analytical solution is used as training data and the performance of the model is tested by predicting the solution on the entire domain after training. The deep learning model predicts the horizontal and vertical deformations well while the error in the predicted pore pressure predictions is slightly higher because of the sparsity of the pore pressure values.

# Introduction

In recent years, physics-informed neural networks (PINNs) have created a new trend at the intersection of machine learning and computational modeling research. Such models involve physical governing equations as constraints in the neural network such that training is performed both on example data and the governing equations. In addition to PINNs, various names are used by different researchers to refer to the concept and the most common ones include \emph{physics-based deep learning}, \emph{theory-guided data science} and \emph{deep hidden physics models}. In general, the aims of these applications include improving the efficiency, accuracy and generalization capability of numerical methods for the solution of PDEs.

Since the pioneering work on PINNs by Raissi et al. 2019, where well-known partial differential equations (PDEs) such as Burgers' equation and Schroedinger's equation are investigated, the concept has been applied to various problems in computational science and engineering. The application areas are increasing rapidly with different variations in the general methodology.

In this paper, application of PINNs to problems of poroelasticity is presented. The governing equations of poroelasticity describe the coupled flow and deformation processes in porous media. Barry and Mercer's poroelastic problem with an 'exact' analytical solution is used to train a deep neural network model where the poroelastic governing equations are applied as constraints. In the following, the coupled governing PDEs of poroelasticity are first described. The architecture of the deep learning model is then described. A numerical example is then presented to show the performance of the deep learning model in comparison with the existing analytical solution.
