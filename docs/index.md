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

Since the pioneering work on PINNs by Raissi et al. (2019), where well-known partial differential equations (PDEs) such as Burgers' equation and Schroedinger's equation are investigated, the concept has been applied to various problems in computational science and engineering. The application areas are increasing rapidly with different variations in the general methodology.

In this paper, application of PINNs to problems of poroelasticity is presented. The governing equations of poroelasticity describe the coupled flow and deformation processes in porous media. Barry and Mercer's poroelastic problem with an 'exact' analytical solution is used to train a deep neural network model where the poroelastic governing equations are applied as constraints. In the following, the coupled governing PDEs of poroelasticity are first described. The architecture of the deep learning model is then described. A numerical example is then presented to show the performance of the deep learning model in comparison with the existing analytical solution.

# Governing Equations of Poroelasticity

The governing equations of poroelasticity are a combination of the overall mass balance equation, the equilibrium or linear momentum balance equation and the linear elastic constitutive equations for stress-strain relationships.

## Mass Balance Equation

The general mass balance equation for a two-phase porous medium (fluid saturated porous solid matrix) under isothermal conditions, obtained from superposition of the individual phase mass balance equations, is given by

$$
\begin{equation}
\alpha \nabla \cdot \dot{\boldsymbol{\overline{u}}} + \left( \frac{\alpha-n}{K_\mathrm{s}} + \frac{n}{K_\mathrm{f}} \right) \frac{\partial \bar{p}}{\partial \bar{t}} + \nabla \cdot \boldsymbol{\overline{w}} = Q, \label{eq:genmasbal} 
\end{equation}
$$

where $ \alpha = 1 - K/K_\mathrm{s} $ is Biot's coefficient, $ \boldsymbol{\overline{u}} $ is the solid deformation vector, $ n $ is the porosity, $ K_\mathrm{s} $ is bulk modulus of the solid, $ K_\mathrm{f} $ is the bulk modulus of the fluid, $ K $ is total bulk modulus of the porous medium, $ \bar{p} $ is the pore fluid pressure, $ \boldsymbol{\overline{w}} $ is the fluid velocity vector and $ Q $ is a fluid source or sink term. For a porous medium with incompressible constituents, i.e. $ 1/K_\mathrm{s} = 1/K_\mathrm{f} = 0 $, the mass balance equation reduces to

$$
\begin{equation}
\nabla \cdot \dot{\boldsymbol{\overline{u}}} + \nabla \cdot \boldsymbol{\overline{w}} = Q. \label{eq:masbalincompressible} 
\end{equation}
$$

The fluid velocity as described by Darcy's law, assuming flow in the porous medium is driven by pressure gradients only, can be expressed as

$$
\begin{equation}
\boldsymbol{\overline{w}} = -\frac{\mathbf k}{\gamma_\mathrm{f}} \nabla \bar{p},
\label{eq:darcyslaw}
\end{equation}
$$

where $ \mathbf k $ is the hydraulic conductivity matrix and $ \gamma_\mathrm{f} $ is the unit weight of the fluid. For an isotropic porous medium, the hydraulic conductivity is the same in all directions of flow and the magnitude of the hydraulic conductivity, $ k $, replaces the hydraulic conductivity matrix in the equation above. For a two-dimensional problem, the deformation vector is $ \boldsymbol{\overline{u}} = \left\lbrace \bar{u}, \bar{v} \right\rbrace^\intercal $ where $\bar{u}$ and $\bar{v}$ are the deformations along the $x$ and $z$ directions, respectively. Introducing the deformation vector and combining equations \eqref{eq:genmasbal} and \eqref{eq:darcyslaw} gives

$$
\begin{equation}
\frac{\partial}{\partial \bar{t}} \left( \frac{\partial \bar{u}}{\partial \bar{x}} + \frac{\partial \bar{v}}{\partial \bar{z}} \right)  - \frac{k}{\gamma_\mathrm{f}} \left( \frac{\partial^2 \bar{p}}{\partial \bar{x}^2} + \frac{\partial^2 \bar{p}}{\partial \bar{z}^2} \right) = Q
\label{eq:masbalxz}
\end{equation}
$$

wherein the hydraulic conductivity is assumed to be isotropic as described earlier.
