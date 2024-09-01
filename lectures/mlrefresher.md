---
type: page
layout: distill
title: Course overview
author: A.Belcaid
permalink: /mlrefresher/
---


<p align="center">
  <img src="{{ '/_images/ml_refresher.webp' | relative_url }}" alt="Data Mining Image" style="width: 50%; height: auto;">
  <br>
</p>


----


# Basic Machine Learning
## Machine Learning as Data-Driven Programming
Machine Learning (ML) is fundamentally a form of **data-driven programming**. Unlike traditional programming, where a human explicitly defines the rules and logic to solve a problem, machine learning leverages **data** to automatically discover patterns and relationships. This approach is particularly powerful when dealing with complex tasks that are difficult to code manually, such as recognizing objects in images, translating languages, or predicting future trends.

One illustrative example of data-driven programming is the **Optical Character Recognition (OCR)** problem, specifically the classification of handwritten digits. Imagine you are tasked with developing a system that can recognize and classify digits (0-9) written by different people. Manually defining rules for every possible way someone might write a "7" or "5" is impractical due to the immense variability in handwriting styles. Instead, with machine learning, we can train a model to learn these patterns from data.

A well-known dataset used to tackle this problem is the [**MNIST dataset**](http://yann.lecun.com/exdb/mnist/), which consists of 60,000 training images and 10,000 test images of handwritten digits. Each image is a 28x28 pixel grayscale image, and the task is to classify each image into one of the 10 digit classes.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" alt="Digit Classification using MNIST">
  <br>
  <a href="https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf" target="_blank">AlexNet 2012 Paper</a> by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton
  <br>
  <a href="http://yann.lecun.com/exdb/mnist/" target="_blank">MNIST Dataset</a>
</p>

In this scenario, machine learning models are trained on the **labeled data** from the MNIST dataset, learning to associate pixel patterns with specific digits. Once trained, these models can generalize to accurately classify new, unseen images of digits, effectively automating the OCR task.

This example highlights the essence of machine learning: creating algorithms that **learn** from data rather than being explicitly programmed, making it possible to solve complex problems in an efficient and scalable manner.

---
## Elements of an ML Algorithm

The **(supervised)** machine learning approach involves **collecting a training set** of images with **known labels**. These labeled images are then fed into a **machine learning algorithm**, which will (if done well), **automatically produce** a "program" that solves the task.

<p align="center">
  <img src="{{ '/_images/ml_model_diagram.png' | relative_url }}" alt="Data Mining Image" style="width: 90%; height: auto;">
  <br>
</p>

Every machine learning algorithm consists of three different elements:

### Hypothesis Class
In the context of machine learning, the hypothesis class refers to the set of all possible **functions or models** that the algorithm can choose from to map inputs (such as images of digits) to outputs (like class labels or probabilities of different class labels). This "program structure" is defined by a set of **parameters** that the algorithm will learn during the training process.

Think of the hypothesis class as a **blueprint** for building the model. It defines the general form or structure that the final model will take, but not the specific details. For example, in a simple linear classifier, the hypothesis class might include all possible straight lines that can separate different classes of data points. The exact line chosen will depend on the parameters that are learned from the data.

The flexibility of the hypothesis class is **crucial**

- it determines the types of patterns the model can learn.
- A more complex hypothesis class might allow the model to capture more intricate relationships in the data, but it also risks overfitting (learning the noise instead of the true signal). 

Thus, choosing the right hypothesis class is a balance between flexibility and simplicity.

### **Example 1: Linear Classifier Hypothesis Class**

In a **linear classifier**, the hypothesis class consists of all possible linear functions that can separate different digit classes. For instance, suppose you have two-dimensional features extracted from the digit images (e.g., pixel intensity and another feature). The linear classifier would attempt to find a straight line (or a hyperplane in higher dimensions) that separates one digit class from another. 

- **Simple Hypothesis Class**: Here, the hypothesis class is restricted to straight lines. This simplicity makes it easier to learn from the data, but it also means that the classifier can only correctly classify the digits if they are linearly separable—if the digits can be distinguished by a straight line or boundary in the feature space. For example, it might work well for distinguishing between digits like "1" and "7," where the distinguishing feature might be as simple as a vertical stroke versus a slanted stroke.

### **Example 2: Neural Network Hypothesis Class**

In contrast, consider a **neural network** as your hypothesis class. Neural networks, especially deep ones, have a much more complex hypothesis class because they consist of multiple layers of neurons that can capture non-linear relationships between input features.

- **Complex Hypothesis Class**: Here, the hypothesis class includes a vast range of non-linear functions. This flexibility allows the model to capture intricate patterns and relationships in the data that a linear classifier would miss. For instance, a neural network could easily distinguish between digits like "8" and "3," which have complex curves and loops, by learning non-linear boundaries in the feature space. The complexity of the hypothesis class allows the network to adapt to the subtle variations in handwriting styles, curvature, and orientation that might be present in the digits.

### **Summary**

These two examples highlight how the hypothesis class can vary significantly:

- A **linear classifier** represents a **simple** hypothesis class, with limitations in capturing complex patterns but is easier to train and interpret.
- A **neural network** represents a **complex** hypothesis class, capable of learning more intricate relationships but requiring more data and computational resources to train effectively.

The choice between these two (or other) hypothesis classes depends on the problem at hand, the complexity of the data, and the resources available for training.

### Loss Function

<p align="center">
  <img src="https://jtuckerk.github.io/images/loss_landscape_big.png" alt="Digit Classification using MNIST">
  <br>
  <a href="https://arxiv.org/abs/1712.09913" target="_blank">Visualizing the loss function of neural network</a>  <br>
  <a href="https://losslandscape.com/" target="_blank">Visualizing the landscape of loss function for neural networks</a>
</p>



The **loss function** is a crucial component of any machine learning algorithm. It acts as a **measure of error** or **discrepancy** between the model’s predictions and the actual labels in the training data. In essence, the loss function quantifies how well or poorly the model is performing at any given point during training.

The goal of the machine learning algorithm is to find the set of parameters in the hypothesis class that **minimizes** this loss function. By minimizing the loss, the model becomes more accurate in making predictions on new, unseen data.

#### **How It Works:**

- **Prediction vs. Reality**: After the model makes a prediction based on the current set of parameters, the loss function compares this prediction to the actual label. If the prediction is far from the actual label, the loss function assigns a high penalty (high loss). If the prediction is accurate, the penalty is low (low loss).

- **Example 1: Mean Squared Error (MSE)**: In a regression setting, a common loss function is the Mean Squared Error. This loss function calculates the average of the squares of the differences between predicted and actual values. It penalizes larger errors more heavily, making it suitable for tasks where it’s important to minimize large deviations.

- **Example 2: Cross-Entropy Loss**: In classification tasks, like digit classification, cross-entropy loss (or log loss) is often used. This loss function measures the difference between two probability distributions—the predicted probability distribution output by the model and the actual distribution (usually represented as a one-hot encoded vector). If the model assigns a high probability to the correct class, the cross-entropy loss will be low. Conversely, if the model assigns a high probability to the wrong class, the loss will be high.

#### **Intuitive Explanation Using Digit Classification:**

Imagine you’re training a model to classify handwritten digits from the MNIST dataset. If the model predicts that a particular digit image is a "7" with 90% confidence, but the actual label is "3," the loss function will penalize this prediction, resulting in a high loss value. On the other hand, if the model correctly predicts the digit "3" with high confidence, the loss value will be low, indicating a good prediction.

#### **Optimization Goal:**

<div style="display: flex; justify-content: center; align-items: center;">
  <div style="margin-right: 20px;">
    <img src="{{ '/_images/gradient_descent_0.1.gif' | relative_url }}" alt="Gradient Descent with Step Size 0.1" style="max-width: 100%;">
  </div>
  <div>
    <img src="{{ '/_images/gradient_descent_vis.gif' | relative_url }}" alt="Gradient Descent with Step Size 0.5" style="max-width: 100%;">
  </div>
</div>
<p align="center"><strong>Figure:</strong> Comparison of Gradient Descent with Step Size 0.1 (left) and Step Size 0.5 (right).</p>

The **objective** of the machine learning algorithm is to **minimize the loss function** across all examples in the training set. During training, the algorithm adjusts the parameters of the model to reduce the loss, thereby improving the model’s predictions. The process of minimizing the loss function is what ultimately allows the model to generalize well to new, unseen data.

### Optimization Method

The **optimization method** is the process or algorithm used to adjust the parameters of the model to minimize the loss function. In simpler terms, it’s the strategy that the machine learning algorithm employs to find the best possible set of parameters within the hypothesis class that results in the lowest error on the training data.

Optimization is at the heart of training a machine learning model. Without an effective optimization method, even the best-defined hypothesis class and loss function would not yield a useful model.

#### **How It Works:**

- **Initial Parameters**: When a machine learning model is first initialized, its parameters are typically set randomly or based on some heuristic. At this stage, the model’s predictions are usually far from correct, leading to a high loss.

- **Gradient Descent**: One of the most commonly used optimization methods is **Gradient Descent**. The idea behind gradient descent is to iteratively adjust the model’s parameters in the direction that reduces the loss the most. This direction is determined by the gradient of the loss function with respect to the parameters. In other words, the gradient tells us how to change the parameters to decrease the loss.

  - **Learning Rate**: The step size for each update is controlled by a parameter called the **learning rate**. A larger learning rate means bigger steps, which can speed up convergence but might overshoot the optimal solution. A smaller learning rate means smaller, more precise steps but can make the training process slower.

- **Stochastic Gradient Descent (SGD)**: In large datasets like MNIST, computing the gradient over the entire dataset can be computationally expensive. **Stochastic Gradient Descent (SGD)** addresses this by updating the parameters using a small, randomly selected subset of the data (called a mini-batch) in each iteration. This makes the optimization process faster and more scalable.

- **Advanced Optimizers**: There are more sophisticated variants of gradient descent, like **Adam**, **RMSprop**, and **Adagrad**, which adaptively adjust the learning rate and incorporate momentum, allowing for faster and more reliable convergence. These optimizers are especially useful in training deep neural networks where the optimization landscape can be complex.

# Image Classification: Example
- **Motivation**. In this section we will introduce the Image Classification problem, which is the task of assigning an input image one label from a fixed set of categories. This is one of the core problems in Computer Vision that, despite its simplicity, has a large variety of practical applications. Moreover, as we will see later in the course, many other seemingly distinct Computer Vision tasks (such as object detection, segmentation) can be reduced to image classification.

- **Example**. For example, in the image below an image classification model takes a single image and assigns probabilities to 4 labels, {cat, dog, hat, mug}. As shown in the image, keep in mind that to a computer an image is represented as one large 3-dimensional array of numbers. In this example, the cat image is 248 pixels wide, 400 pixels tall, and has three color channels Red,Green,Blue (or RGB for short). Therefore, the image consists of 248 x 400 x 3 numbers, or a total of 297,600 numbers. Each number is an integer that ranges from 0 (black) to 255 (white). Our task is to turn this quarter of a million numbers into a single label, such as “cat”.


---
<p align="center">
  <img src="{{ '/_images/classify_example_four_classes.png' | relative_url }}" alt="Image classification example">
  <br>
<small>
The task in Image Classification is to predict a single label (or a distribution over labels as shown here to indicate our confidence) for a given image. Images are 3-dimensional arrays of integers from 0 to 255, of size Width x Height x 3. The 3 represents the three color channels Red, Green, Blue.
</small>
</p>
---
- **Challenges**. Since this task of recognizing a visual concept (e.g. cat) is relatively trivial for a human to perform, it is worth considering the challenges involved from the perspective of a Computer Vision algorithm. As we present (an inexhaustive) list of challenges below, keep in mind the raw representation of images as a 3-D array of brightness values:
- **Viewpoint variation**. A single instance of an object can be oriented in many ways with respect to the camera.
Scale variation. Visual classes often exhibit variation in their size (size in the real world, not only in terms of their extent in the image).
- **Deformation**. Many objects of interest are not rigid bodies and can be deformed in extreme ways.
- **Occlusion**. The objects of interest can be occluded. Sometimes only a small portion of an object (as little as few pixels) could be visible.
- **Illumination conditions**. The effects of illumination are drastic on the pixel level.
- **Background clutter**. The objects of interest may blend into their environment, making them hard to identify.
Intra-class variation.

The classes of interest can often be relatively broad, such as chair. There are many different types of these objects, each with their own appearance.
A good image classification model must be invariant to the cross product of all these variations, while simultaneously retaining sensitivity to the inter-class variations.


---
<p align="center">
  <img src="{{ '/_images/challenges_image_classification.jpeg' | relative_url }}" alt="Image classification example">
  <br>
<small>
Data-driven approach. How might we go about writing an algorithm that can classify images into distinct categories? Unlike writing an algorithm for, for example, sorting a list of numbers, it is not obvious how one might write an algorithm for identifying cats in images. Therefore, instead of trying to specify what every one of the categories of interest look like directly in code, the approach that we will take is not unlike one you would take with a child: we’re going to provide the computer with many examples of each class and then develop learning algorithms that look at these examples and learn about the visual appearance of each class. This approach is referred to as a data-driven approach, since it relies on first accumulating a training dataset of labeled images. Here is an example of what such a dataset might look like:
</small>
<br>
<img src="{{ '/_images/trainset_example_classification.jpg' | relative_url }}" alt="Image classification example">
</p>
---

## CIFAR 10

One popular toy image classification dataset is the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

- This dataset consists of **60,000** tiny images that are 32 pixels high and wide.
- Each image is labeled with one of **10** classes (for example “airplane, automobile, bird, etc”).
- These **60,000** images are partitioned into a training set of **50,000** images and a test set of **10,000** images.
- In the image below you can see 10 random example images from each one of the 10 classes:

---
<p align="center">
  <img src="{{ '/_images/example_cifar_10.jpg' | relative_url }}" alt="Image classification example">
  <br>
<small>
Example images from the <a href="https://www.cs.toronto.edu/~kriz/cifar.html">CIFAR-10 dataset</a>. Right: first column shows a few test images and next to each we show the top 10 nearest neighbors in the training set according to pixel-wise difference.
</small>
</p>
---

Let's consider some notations which follows the standard in Machine Learning:

- Training data : 

$$
x^i \in \mathbb{R}^n \quad y^i \in \{1, 2, \ldots, k\}\quad \forall i = 1,\ldots, m
$$

- $n$ is the **dimentionality** of the input data.
- $k$ is the number of classes.
- $m$ is the number of points in the training set.

So for the **CIFAR** we have:

- $n = 32\times 32 = 1024$
- $m = 50000$
- $k = 10$.

> Can you think of an **hypothesis function** for this problem?

Our hypothesis functions maps inputs $x\in \mathbb{R}^n$ to $k-$dimentional vectors:

$$
h: \mathbb{R}^n \rightarrow \mathbb{R}^k
$$

where $h_i(x)$ indicate the measure of **belief** on how much likely the label is to be the class $i$.

For example, a **Linear hypothesis function** will have the following form:

$$
h_\theta(x) = \theta^T x
$$

where $\theta \in \mathbb{R}^{n \times k}$

We should also choose a **loss function** to train the model. We will start with the simplest loss function to use in classification is just the classification error, i.e.,whether the classifier makes a **mistake** a or **no**.

$$
l_{\text{err}}(h(x), y) =  \left\{\begin{array}{ll}0& \text{if } argmax_i h_i(x) = y\\1&\text{otherwise}\end{array}\right.
$$

> Unfortunately, the error is a bad loss function to use for optimization, i.e., selecting
the best parameters, because it is not differentiable

#### **Softmax Function**

The softmax function is used in the final layer of a neural network to convert the raw output scores (also known as logits) into probabilities. It does this by normalizing the output scores so that they sum to 100%, effectively representing a probability distribution over all possible classes.

Given an input vector of scores $ z = [z_1, z_2, \dots, z_k] $ for $ k $ classes, the softmax function is defined as:

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}
$$

This equation converts the score $ z_i $ for each class into a probability $ p_i $, where $ 0 \leq p_i \leq 1 $ and $ \sum_{i=1}^{k} p_i = 1 $.

#### **Cross-Entropy Loss**

Once the probabilities are obtained from the softmax function, the **cross-entropy loss** is used to quantify the difference between the predicted probability distribution and the actual distribution (which is usually a one-hot encoded vector representing the true class).

For a single training example, where the true label is $ y $ and the predicted probability for each class is $p_i $, the cross-entropy loss is defined as:

$$
\text{Loss} = -\log(p_y)
$$

If the true label is \( y \), this loss function penalizes the model based on how much probability it assigned to the correct class \( y \). If the model assigns a high probability to the correct class, the loss is low. Conversely, if the model assigns a low probability to the correct class, the loss is high.

#### **Combined Softmax and Cross-Entropy**

When the softmax function and cross-entropy loss are combined, the result is a single, differentiable loss function that can be used to optimize the model’s parameters via gradient descent or other optimization algorithms. This combination is particularly powerful because:

- The **softmax function** ensures that the model outputs are interpretable as probabilities.
- The **cross-entropy loss** provides a natural way to measure the performance of these probabilistic predictions, penalizing incorrect predictions based on the confidence level of the model.

This loss function is widely used in tasks such as image classification, where the goal is to assign an input image to one of several possible categories.


Now we move on the optimizer. For starter we could use [**Gradient descent**](https://en.wikipedia.org/wiki/Gradient_descent)

$$
\theta = \theta - \alpha \nabla_\theta f(\theta)
$$

Of course we could use other optimisation methods that we will cover in up-coming lectures.



