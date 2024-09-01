---
type: page
layout: distill
title: Course overview
author: A.Belcaid
permalink: /introduction/
---


<p align="center">
  <img src="{{ '/_images/intro_deep_learning.jpg' | relative_url }}" alt="Data Mining Image" style="width: 50%; height: auto;">
  <br>
</p>


----

## Aim of this course

- This course will provide you will an introduction to the functioning of **modern deep
learning systems**.

- You will learn about the underlying concepts of modern deep learning systems like
**automatic differentiation**, **neural network architectures**, **optimization**, and efficient
operations on systems like GPUs.

- To solidify your understanding, along the way (in your homeworks), you will build
**(from scratch)** all the basic architectures and models like a **fully connected model** or a **CNN** for image classification.

----

## Why Study Deep learning

Here is a list of important Breakthroughs  that uses Deep learning.

1. **Image Classification**


<p align="center">
  <img src="{{'/_images/image_classification.png' | relative_url }}" alt="Data Mining Image">
  <br>

  <a href="https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf" target="_blank">AlexNet 2012 Paper</a> by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton
</p>


<br>
<br>

<p align="center">
  <img src="{{ '/_images/alpha_go.png' | relative_url }}" alt="AlphaGo Image" style="width: 90%; height: auto;">
  <br>
  <em>Figure 1: Visualization of AlphaGo Process</em>
  <br>
  <a href="https://www.nature.com/articles/nature16961" target="_blank">AlphaGo 2016 Paper</a> by David Silver, Aja Huang, Chris J. Maddison, et al.
</p>

<br>
<br>

<p align="center">
  <img src="{{ '/_images/diffusion.png' | relative_url }}" alt="Diffusion Model Image" style="width: 100%; height: auto;">
  <br>
  <em>Figure 1: Visualization of Diffusion Model Process</em>
  <br>
  <a href="https://arxiv.org/abs/2006.11239" target="_blank">Diffusion Models Paper (2020)</a> by Jonathan Ho, Ajay Jain, Pieter Abbeel
</p>

----

## Why Study Deep learning

Deep learning has revolutionized the field of artificial intelligence by enabling machines to learn from **vast amounts of data**, often surpassing human capabilities in complex tasks. By studying deep learning, we can unlock the potential to **solve some of the most challenging problems** across various domains, including natural language processing, computer vision, healthcare, and scientific research. Deep learning models are the **backbone** of many groundbreaking applications that are transforming industries, enhancing productivity, and creating new possibilities for innovation. Understanding deep learning is essential for anyone looking to contribute to the future of technology, as it holds the key to advancements in automation, personalized services, and the development of intelligent systems that can adapt, evolve, and improve over time.


- ChatGpt

<p align="center">
  <img src="{{ '/_images/chat_gpt.png' | relative_url }}" alt="ChatGPT Image" style="width: 50%; height: auto;">
  <br>
  <em>Figure 1: ChatGPT – Revolutionizing Natural Language Processing</em>
  <br>
  <a href="https://arxiv.org/abs/2005.14165" target="_blank">ChatGPT 2022 Paper</a> by Tom B. Brown, Benjamin Mann, Nick Ryder, et al.
</p>
ChatGPT, powered by OpenAI's GPT-3, has revolutionized how machines understand and generate human language. This model, which is based on deep learning techniques, has demonstrated the ability to perform a wide range of tasks, from writing essays to answering complex questions, making it a powerful tool in natural language processing.

<br>

- AlphaFold 2 (2021)
<p align="center">
  <img src="{{ '/_images/alpha_fold.png' | relative_url }}" alt="AlphaFold 2 Image" style="width: 50%; height: auto;">
  <br>
  <em>Figure 2: AlphaFold 2 – Advancing Protein Structure Prediction</em>
  <br>
  <a href="https://www.nature.com/articles/s41586-021-03819-2" target="_blank">AlphaFold 2 Paper (2021)</a> by John Jumper, Richard Evans, Alexander Pritzel, et al.
</p>
AlphaFold 2, developed by DeepMind, represents a major breakthrough in the field of bioinformatics. By applying deep learning, AlphaFold 2 was able to predict protein structures with unprecedented accuracy, addressing a fundamental challenge in biology that had remained unsolved for decades.

<br>

- Stable Diffusion (2022)
<p align="center">
  <img src="{{ '/_images/Renaissance-Astronaut-Portrait-Best-Stable-Diffusion-Prompts.webp' | relative_url }}" alt="Stable Diffusion Image" style="width: 50%; height: auto;">
  <br>
  <em>Figure 3: Stable Diffusion – Transforming Image Generation</em>
  <br>
  <a href="https://arxiv.org/abs/2207.12598" target="_blank">Stable Diffusion Paper (2022)</a> by Robin Rombach, Andreas Blattmann, Dominik Lorenz, et al.
</p>
Stable Diffusion is an advanced generative model that leverages diffusion processes to create high-quality images from text descriptions. This innovation in deep learning allows for the generation of stunning visuals and has opened new avenues in art, design, and media production. The image is generated by the prompt "Portrait de style Renaissance d'un astronaute dans l'espace, fond étoilé détaillé, casque réfléchissant."

### Trends in Deep Learning from 2008 to Present

Deep learning has undergone significant evolution since 2008, marked by key milestones that have shaped the field. The early years saw the resurgence of neural networks with the advent of powerful GPUs, leading to the development of **AlexNet in 2012**, which demonstrated the potential of deep learning in image recognition. This breakthrough ushered in the modern era of deep learning, followed by the creation of **Keras in 2015**, which made building and experimenting with neural networks accessible to a broader audience through its user-friendly interface. The release of **TensorFlow** in the same year by Google provided a robust, scalable platform for deploying deep learning models in production environments. More recently, **PyTorch**, introduced in 2016, has become a favorite among researchers and developers for its flexibility and ease of use, enabling rapid prototyping and experimentation. These tools and frameworks have been instrumental in advancing deep learning, leading to its widespread adoption across industries and its integration into cutting-edge technologies.

<p align="center">
  <img src="{{ '/_images/trends_deep_learning.png' | relative_url }}" alt="Stable Diffusion Image" style="width: 90%; height: auto;">
  <br>
  <em>Figure 3: Trends for Deep learning</em>
  <br>
</p>

## Evolution of Deep learning libraries

- In the first era, working on deep learning involve writing **heavy** code to define your model and specification and spending an important amount of time for training.


<p align="center">
  <img src="{{ '/_images/working_deep_learning_before.png' | relative_url }}" alt="Stable Diffusion Image" style="width: 90%; height: auto;">
  <br>
</p>

- But now, things have really changed with some automatic libraries like [Torch lightning](https://lightning.ai/docs/pytorch/stable/) 



<p align="center">
  <img src="{{ '/_images/working_deep_learning_after.png' | relative_url }}" alt="Stable Diffusion Image" style="width: 90%; height: auto;">
  <br>
</p>

## Overview of the course:

In this course, we will try to touch on the following chapters and points:

- Machine Learning Refresher.
- Back propagation and  automatic differentiation.
- Neural Networks: Architecture
- Neural Networks: Data and the loss
- Neural Networks: Data and the loss
- Neural Networks: Learning and Evaluation.
- Convolutional neural Networks
- Classical Models zoology
- Recurrent Neural Networks.
