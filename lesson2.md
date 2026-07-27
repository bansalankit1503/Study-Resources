# Lesson 2 — How Does a Neural Network Learn?

> **Goal:** By the end of this lesson, you'll understand how a computer makes decisions from numbers and why **weights** are the foundation of every neural network, including Transformers and SAM.

---

# Recap

From Lesson 1, we learned:

```
Real World
     │
     ▼
Image / Text / Audio
     │
     ▼
Numbers (Tensor)
```

But now comes the important question...

## The computer has numbers.

Now what?

How does it know whether this image is a **dog** or a **cat**?

---

# Imagine You Are Hiring Someone

Suppose you want to hire an employee.

You don't make your decision based on just one thing.

You consider several factors:

| Feature        | Importance |
| -------------- | ---------: |
| Experience     |       High |
| Education      |     Medium |
| Communication  |       High |
| Certifications |        Low |

Notice something?

Every feature has a different **importance**.

In machine learning, this importance is called a **weight**.

---

# A Computer Thinks the Same Way

Suppose we want to predict the price of a house.

The computer receives:

```
House Size
Bedrooms
Bathrooms
Age
Location Score
```

Example:

```
House Size = 1800
Bedrooms = 3
Bathrooms = 2
Age = 5
Location = 9
```

The computer doesn't know which feature matters more.

So it learns numbers called **weights**.

---

# Weight = Importance

Suppose after training, the model learns:

| Feature   | Weight |
| --------- | ------ |
| Size      | 0.8    |
| Bedrooms  | 0.2    |
| Bathrooms | 0.1    |
| Age       | -0.3   |
| Location  | 1.5    |

Interpretation:

* Bigger house → increases price
* Better location → increases price a lot
* Older house → decreases price

The weights capture these relationships.

---

# The Simplest Neural Network

Imagine just one neuron.

```
          x₁
           \
            \
             \
              ○ -----> Output
             /
            /
          x₂
```

Mathematically:

```
Output = x₁·w₁ + x₂·w₂
```

Where:

* `x` = input
* `w` = weight

---

# Add More Inputs

Now imagine four inputs.

```
Size --------\
Bedrooms -----\
Bathrooms -----○
Age ----------/
```

The neuron computes:

```
Output

=

(Size × Weight₁)

+

(Bedrooms × Weight₂)

+

(Bathrooms × Weight₃)

+

(Age × Weight₄)
```

This is just a weighted sum.

---

# Why Do We Need a Bias?

Imagine a teacher grading students.

Suppose a student scores:

```
70
80
90
```

But the teacher decides to give everyone **+5 bonus marks**.

That extra bonus is like the **bias**.

So now:

```
Output

=

Weighted Sum

+

Bias
```

Mathematically:

```
y = xw + b
```

where:

* `x` = input
* `w` = weight
* `b` = bias

The bias lets the neuron shift its output independently of the inputs.

---

# Real Example

Suppose

```
Size = 2000
Weight = 0.5

Bias = 100
```

Then

```
Output

=

2000 × 0.5

+

100

=

1100
```

---

# What Does the Neuron Actually Do?

Think of it like this:

```
Input Numbers

↓

Multiply by Importance

↓

Add Everything

↓

Add Bias

↓

Decision
```

That's all a neuron does before applying an activation function.

---

# But There's a Problem...

Suppose we want to recognize faces.

Could we solve that with just one neuron?

No.

A face depends on many complex patterns:

* Eyes
* Nose
* Hair
* Skin texture
* Lighting
* Angles

A single weighted sum isn't expressive enough.

We need many neurons working together.

---

# Layer of Neurons

Instead of one neuron:

```
Input

↓

Neuron

↓

Output
```

we build layers:

```
Input

↓

○ ○ ○ ○

↓

○ ○ ○

↓

○

↓

Output
```

Each neuron learns different patterns.

---

# Why Is It Called a Neural Network?

Because it's inspired (loosely) by the brain.

A biological neuron:

```
Signals

↓

Neuron

↓

Signal
```

Artificial neuron:

```
Numbers

↓

Weighted Sum

↓

Output Number
```

The resemblance is conceptual rather than biological.

---

# Where Does the Learning Happen?

Here's the key question:

Who chooses the weights?

```
Weight = 0.8

Weight = -1.3

Weight = 2.4
```

Do we set them manually?

No.

The model **learns** them from data.

This learning process is what training is all about.

---

# Example: Dog vs Cat

Suppose the input features are:

```
Ear Length

Tail Length

Weight

Fur Density
```

Initially, the weights are random:

```
0.32

-1.4

0.09

2.1
```

The prediction is poor.

During training, the model repeatedly adjusts the weights.

Eventually, they might become:

```
2.4

0.8

-0.2

3.6
```

Now the model makes much better predictions.

---

# How Does the Model Know It Is Wrong?

Imagine the model predicts:

```
Dog
```

But the correct answer is:

```
Cat
```

It computes an **error** (also called a **loss**).

```
Prediction

↓

Compare with Truth

↓

Error

↓

Adjust Weights
```

This cycle repeats many times.

---

# The Learning Cycle

This is the heartbeat of machine learning:

```
Input

↓

Prediction

↓

Calculate Error

↓

Update Weights

↓

Better Prediction

↓

Repeat
```

Every deep learning model follows this loop.

---

# How Does This Connect to Transformers?

People often think Transformers are something completely different.

They're not.

Every Transformer layer still contains **millions or billions of weights**.

Examples:

* Query projection weights
* Key projection weights
* Value projection weights
* MLP weights
* Output projection weights

The architecture is more sophisticated, but learning still means **adjusting weights**.

---

# How Does This Connect to LoRA?

Now think about your SAM project.

Normally, fine-tuning updates all these weights:

```
W
```

LoRA keeps the original weights fixed and learns a small update:

```
W + ΔW
```

where the update is represented efficiently using low-rank matrices.

We'll derive that mathematically later in the course.

For now, remember:

> LoRA works because neural networks learn through weights—and LoRA changes **how** those weights are updated, not **whether** they exist.

---

# PyTorch Example

Here's what a single neuron looks like in PyTorch:

```python
import torch
import torch.nn as nn

# One neuron
neuron = nn.Linear(4, 1)

# Four input features
x = torch.tensor([[1800.0, 3.0, 2.0, 5.0]])

# Forward pass
output = neuron(x)

print(output)
```

`nn.Linear(4, 1)` creates a neuron with:

* 4 input features
* 4 learnable weights
* 1 learnable bias

---

# Key Takeaways

By the end of this lesson, you should remember:

1. A neuron computes a weighted sum of its inputs.
2. A weight represents how important an input is.
3. A bias lets the neuron shift its output.
4. Learning means adjusting weights and biases to reduce prediction errors.
5. Transformers, Vision Transformers, SAM, and LoRA are all built on these same learnable parameters.

---

# Mini Exercise

Try answering these before moving on:

1. What is the purpose of a weight?
2. Why is a bias useful?
3. Why can't a single neuron recognize complex objects like faces?
4. What does it mean for a neural network to "learn"?
5. In LoRA, what kind of model parameters are ultimately being adapted?

---

# Next Lesson (Lesson 3)

We'll answer one of the most important questions in deep learning:

> **How does a neural network decide whether to activate a neuron or ignore it?**

You'll learn:

* Activation functions
* Why linear models are limited
* Sigmoid
* Tanh
* ReLU
* Why ReLU dominates modern deep learning
* Why Transformers use **GELU** instead of ReLU
* How activation functions appear inside SAM and Vision Transformers

This lesson will explain why deep networks can model complex patterns instead of behaving like a single linear equation.
