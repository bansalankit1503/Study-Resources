# Lesson 3 — Why Activation Functions Exist

> **Goal:** By the end of this lesson, you'll understand why stacking many linear layers **doesn't** make a model smarter, why activation functions are essential, and why **GELU** is used in Transformers, ViT, and SAM.

This is one of the most important lessons in deep learning.

---

# Recap

In Lesson 2, we learned that a neuron computes:

[
y = Wx + b
]

Where:

* **x** = input
* **W** = weights
* **b** = bias

Suppose:

```text
Input = 5

Weight = 2

Bias = 1
```

The neuron computes:

```text
Output = (5 × 2) + 1 = 11
```

Easy.

But here's the big question...

---

# Can We Build Intelligence Using Only This?

Imagine building a huge neural network:

```text
Input

↓

Linear Layer

↓

Linear Layer

↓

Linear Layer

↓

Linear Layer

↓

Output
```

Looks powerful, right?

Actually...

**No.**

This is one of the biggest surprises in deep learning.

---

# Why Doesn't This Work?

Let's simplify.

Suppose the first layer computes:

```text
y₁ = 2x + 3
```

The second layer computes:

```text
y₂ = 4y₁ + 1
```

Substitute the first equation:

```text
y₂ = 4(2x + 3) + 1
```

Expand:

```text
y₂ = 8x + 13
```

Notice something?

Two layers became...

**One layer.**

Let's add another.

```text
y₃ = 5y₂ + 7
```

Substitute:

```text
y₃ = 5(8x + 13) + 7
```

Result:

```text
y₃ = 40x + 72
```

Again...

Still just one linear equation.

---

# The Big Discovery

No matter how many **linear layers** you stack,

```text
Linear

↓

Linear

↓

Linear

↓

Linear
```

they collapse into

```text
One Linear Layer
```

This means:

> **Without activation functions, a 100-layer neural network is mathematically equivalent to a single linear transformation.**

It cannot learn complex patterns.

---

# Real-Life Analogy

Imagine a factory.

Every machine only doubles a number.

```text
Machine 1

5 → 10

↓

Machine 2

10 → 20

↓

Machine 3

20 → 40
```

No matter how many machines you add...

The whole factory is just multiplying by one larger number.

Nothing fundamentally new happens.

To build intelligence, we need something that **changes the behavior**.

That's exactly what an activation function does.

---

# What Is an Activation Function?

Instead of

```text
Input

↓

Linear

↓

Output
```

we do

```text
Input

↓

Linear

↓

Activation

↓

Output
```

Now the neuron becomes:

```text
Output

=

Activation(Wx + b)
```

The activation introduces **non-linearity**, allowing the network to model much more complex relationships.

---

# Imagine a Security Guard

A neuron calculates a score:

```text
Score = 3.2
```

The activation function is like a guard deciding what passes through.

Depending on the activation:

* Allow everything
* Block negatives
* Compress values
* Scale outputs

The activation controls the flow of information.

---

# Activation Function #1 — Sigmoid

The sigmoid function maps any number into the range **0 to 1**.

Conceptually:

```text
Very Negative → close to 0

0 → 0.5

Very Positive → close to 1
```

It behaves like a smooth probability.

Examples:

```text
Input   Output

-10     0.000045

0       0.5

10      0.99995
```

---

## Why Was Sigmoid Popular?

Suppose you're predicting:

```text
Spam

or

Not Spam
```

Returning

```text
0.97
```

is naturally interpreted as

> 97% confidence.

That's why early neural networks used sigmoid extensively.

---

# The Problem with Sigmoid

Imagine this graph:

```text
          _________

       /

     /

___/
```

At the extremes, it becomes almost flat.

When learning, flat regions produce **very small gradients**.

Small gradients mean:

* tiny parameter updates
* extremely slow learning

This is called the **vanishing gradient problem**.

---

# Activation Function #2 — Tanh

Tanh is similar to sigmoid but outputs values between **-1 and 1**.

```text
Negative → -1

Zero → 0

Positive → 1
```

Unlike sigmoid, it's centered around zero, which often makes optimization easier.

---

# The Problem with Tanh

Although better than sigmoid in some cases,

it still saturates for very large positive or negative inputs.

So gradients can still become very small.

---

# Activation Function #3 — ReLU

ReLU stands for:

**Rectified Linear Unit**

It's surprisingly simple:

```text
If x > 0

Return x

Else

Return 0
```

Examples:

```text
Input    Output

-5       0

-1       0

0        0

2        2

8        8
```

---

# Why Did ReLU Change Deep Learning?

Imagine electricity flowing through wires.

Negative signals:

```text
Blocked
```

Positive signals:

```text
Allowed
```

ReLU behaves similarly.

Benefits:

* Very fast
* Easy to compute
* Doesn't saturate for positive values
* Helps train deep networks efficiently

This is why CNNs such as ResNet became practical.

---

# The Problem with ReLU

Suppose a neuron always receives negative inputs.

```text
-2

↓

ReLU

↓

0
```

Next iteration:

```text
-3

↓

0
```

Again:

```text
-4

↓

0
```

The neuron never activates again.

Its output stays zero, and it may stop learning.

This is called the **dying ReLU** problem.

---

# Then Why Don't Transformers Use ReLU?

Transformers require smoother behavior.

They don't just want neurons to be:

```text
ON

OFF
```

They want outputs that change smoothly, even for slightly negative inputs.

That leads us to...

---

# GELU (Gaussian Error Linear Unit)

Modern Transformers—including **BERT**, **Vision Transformer (ViT)**, **GPT**, and **SAM**—typically use **GELU** instead of ReLU.

Intuition:

Instead of

```text
Negative

↓

Always Zero
```

GELU says:

> "Small negative values might still contain useful information."

It suppresses them smoothly rather than cutting them off abruptly.

Conceptually:

```text
ReLU

Negative

↓

0
```

versus

```text
GELU

Negative

↓

Small Negative

↓

Gradually Increasing

↓

Positive
```

---

# Why GELU Is Better for Transformers

Transformers work with rich representations such as word embeddings and image patch embeddings.

A small negative value doesn't necessarily mean "bad information."

GELU lets the model retain subtle information while still emphasizing strong positive signals.

That's one reason it performs well in Transformer architectures.

---

# Where Does GELU Appear in SAM?

A simplified Transformer block looks like this:

```text
Input
   │
   ▼
Multi-Head Attention
   │
   ▼
Add & LayerNorm
   │
   ▼
Linear
   │
   ▼
GELU
   │
   ▼
Linear
   │
   ▼
Add & LayerNorm
```

Notice something?

The activation function isn't inside the attention mechanism.

It's inside the **MLP (Feed-Forward Network)** that follows attention.

When you inspect the SAM source code, you'll find GELU used in these feed-forward layers.

---

# PyTorch Example

```python
import torch
import torch.nn as nn

x = torch.tensor([-3., -1., 0., 1., 3.])

relu = nn.ReLU()
gelu = nn.GELU()

print("Input :", x)
print("ReLU  :", relu(x))
print("GELU  :", gelu(x))
```

You'll observe:

* ReLU sets all negative values to exactly zero.
* GELU produces smooth negative outputs for small negative inputs.

---

# Why This Matters for LoRA

Suppose you're fine-tuning SAM.

LoRA updates certain weight matrices.

Those updated weights feed into layers like:

```text
Linear

↓

GELU

↓

Linear
```

If you don't understand the role of GELU, it's difficult to reason about how information flows through the model after your LoRA updates.

So even though LoRA only changes weights, those weights are interpreted through activation functions.

---

# Summary

Every neuron computes:

```text
Output

=

Activation(Wx + b)
```

The activation function is what makes deep learning truly "deep."

Without it:

* Deep networks collapse into one linear transformation.
* They cannot model complex patterns.

With it:

* Networks learn highly non-linear relationships.
* Vision models recognize objects.
* Language models understand context.
* SAM learns rich image representations.

---

# Key Takeaways

By the end of this lesson, you should remember:

1. A stack of linear layers is still just one linear transformation.
2. Activation functions introduce non-linearity, enabling complex learning.
3. Sigmoid outputs values between 0 and 1 but suffers from vanishing gradients.
4. ReLU is simple and efficient but can produce "dead" neurons.
5. GELU is smoother than ReLU and is widely used in modern Transformer-based models such as ViT and SAM.

---

# Mini Exercise

Answer these without looking back:

1. Why can't a network made only of linear layers learn complex functions?
2. What problem does an activation function solve?
3. Why did ReLU become more popular than sigmoid?
4. What is the "dying ReLU" problem?
5. Why do Transformer models like SAM prefer GELU over ReLU?

---

# Next Lesson (Lesson 4)

We'll dive into **how neural networks actually learn** by exploring:

* What is a **loss function**?
* How does a model measure its mistakes?
* Mean Squared Error (MSE)
* Cross-Entropy Loss
* Binary vs. Multi-class classification
* The intuition behind optimization

This lesson will prepare you for **backpropagation**, which is the algorithm that updates the weights of every neural network—from a tiny perceptron to the billions of parameters inside SAM.
