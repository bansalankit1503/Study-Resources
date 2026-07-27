# Lesson 5 — Gradient Descent: How Does a Neural Network Learn?

> **Goal:** By the end of this lesson, you'll understand exactly what **gradients** are, why **calculus** is used in deep learning, what `loss.backward()` computes, and why **Gradient Descent** is the engine behind training models like SAM, ViT, GPT, and LoRA.

---

# Recap

So far, we know this training loop:

```text
Image
   │
   ▼
Neural Network
   │
   ▼
Prediction
   │
   ▼
Loss
```

But something is missing.

After computing the loss...

**How does the model become better?**

How does it know which weight to change?

---

# Imagine You're Climbing a Mountain... Backwards

Suppose you're standing on top of a mountain.

Your goal is to reach the **lowest point** (the valley).

```
           /\      ← Mountain
          /  \
         /    \
        /      \
_______/        \_______
 Valley         Valley
```

Imagine it's very foggy.

You cannot see the whole mountain.

You can only look at the ground around your feet.

Question:

**How do you reach the bottom?**

Simple.

Take one small step in the direction that goes downward.

Take another.

Then another.

Eventually...

You reach the valley.

---

# Training a Neural Network is Exactly Like This

Replace the mountain with the **Loss Function**.

```
High Loss

        ▲

        |

        |

        |

        |

Low Loss
```

Your model starts somewhere on the mountain.

Its job is to reach the minimum loss.

---

# What Tells Us Which Way is Down?

Imagine standing here.

```
        *
      /
    /
  /
/
```

How do you know which direction decreases the height?

You measure the **slope**.

In mathematics,

the slope is called the **Derivative**.

In machine learning,

we usually call it the **Gradient**.

---

# What is a Gradient?

A gradient answers one simple question:

> **If I change this weight a little, how will the loss change?**

That's it.

Suppose:

```
Weight = 2.0
```

Increase it slightly:

```
Weight = 2.1
```

What happened?

* Loss increased?
* Loss decreased?
* Stayed the same?

The gradient tells us.

---

# Everyday Example

Imagine adjusting your shower.

```
Hot

↓

Warm

↓

Cold
```

You slightly turn the knob.

If the water gets hotter,

you know which direction you're moving.

If it gets colder,

you turn back.

This feedback is exactly what gradients provide.

---

# One Weight Example

Suppose our model is:

[
y = wx
]

Input:

```
x = 5
```

Current weight:

```
w = 2
```

Prediction:

```
10
```

Actual answer:

```
20
```

Loss is high.

Should we increase the weight?

Or decrease it?

The gradient answers that question.

---

# Positive Gradient

Suppose

```
Gradient = +4
```

This means:

> Increasing the weight increases the loss.

So...

We should move in the opposite direction.

Decrease the weight.

---

# Negative Gradient

Suppose

```
Gradient = -7
```

This means

Increasing the weight decreases the loss.

Great!

Increase the weight.

---

# The Golden Rule

We always move **opposite the gradient**.

That's why it's called

**Gradient Descent**

We descend the loss surface.

---

# The Weight Update Rule

This single equation powers almost all deep learning:

[
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
]

Let's break it down.

* (w) = current weight
* (L) = loss
* (\frac{\partial L}{\partial w}) = gradient
* (\eta) (eta) = learning rate

This says:

> New Weight = Old Weight − (Step Size × Slope)

---

# Example

Suppose

Current weight

```
5
```

Gradient

```
2
```

Learning rate

```
0.1
```

Then

```
New Weight

=

5

-

0.1 × 2

=

4.8
```

The weight moves a little.

Not a lot.

Just a little.

---

# Why Not Take Giant Steps?

Suppose you're descending a mountain.

Small steps:

```
🙂

↓

🙂

↓

🙂
```

Safe.

Now imagine taking huge jumps.

```
😱

↓

Overshoot

↓

Back

↓

Overshoot

↓

Back
```

You'll keep jumping across the valley.

Never settling.

This is exactly what happens with a **large learning rate**.

---

# Learning Rate

The learning rate controls

> **How big each update should be.**

Small learning rate

```
Tiny steps
```

Training is stable but slow.

Large learning rate

```
Huge jumps
```

Training becomes unstable.

---

# Visualizing Learning Rate

Small LR

```
●

↓

●

↓

●

↓

●

↓

Minimum
```

Large LR

```
●

↓

↓

Minimum

↑

↓

↑

↓

Keeps bouncing
```

Choosing the right learning rate is one of the most important parts of training.

---

# Why Calculus?

People often ask:

> "Why do I need calculus for Machine Learning?"

Now you know.

Because we need to answer:

> **How does changing a weight affect the loss?**

That question is answered by derivatives.

Without derivatives,

the model wouldn't know how to improve.

---

# Millions of Weights

So far,

we've talked about one weight.

Real models have many.

Example:

```
Weight 1

Weight 2

Weight 3

...

Weight 1,000,000,000
```

Each weight has its own gradient.

Each one gets updated.

Every training step.

---

# Where Does `loss.backward()` Fit?

Suppose your code is:

```python
loss.backward()
```

What actually happens?

The model computes:

```
Gradient of Weight 1

Gradient of Weight 2

Gradient of Weight 3

...

Gradient of Every Trainable Parameter
```

These gradients are stored in each parameter's `.grad` field.

Nothing is updated yet.

---

# Then What Does `optimizer.step()` Do?

Now we have gradients.

The optimizer applies the update rule:

```
Weight

↓

Weight - LR × Gradient

↓

New Weight
```

This happens for every trainable parameter.

---

# Complete Training Cycle

Now you understand the whole loop:

```
Image

↓

Prediction

↓

Loss

↓

Compute Gradients

↓

Update Weights

↓

Better Prediction

↓

Repeat
```

This repeats thousands or millions of times.

---

# What About LoRA?

This is where your project becomes interesting.

Normal fine-tuning:

```
Loss

↓

Gradient

↓

Update ALL weights
```

LoRA:

```
Loss

↓

Gradient

↓

Only LoRA weights updated

↓

Original model stays frozen
```

This is why LoRA is much faster and uses less memory.

The gradient still exists.

It simply flows only through the trainable LoRA parameters.

---

# PyTorch Example

```python
import torch
import torch.nn as nn

model = nn.Linear(2, 1)

x = torch.tensor([[2.0, 3.0]])
y = torch.tensor([[8.0]])

criterion = nn.MSELoss()

prediction = model(x)

loss = criterion(prediction, y)

loss.backward()

print(model.weight.grad)
print(model.bias.grad)
```

Notice:

After `loss.backward()`

you'll see gradients printed.

The weights themselves have **not changed yet**.

Now update them:

```python
optimizer.step()
```

Only then do the weights change.

---

# A Common Misunderstanding

Many beginners think:

```python
loss.backward()
```

updates the model.

It does **not**.

It only computes gradients.

The update happens here:

```python
optimizer.step()
```

This distinction is very important.

---

# How SAM Learns

When training SAM:

```
Image

↓

Image Encoder (ViT)

↓

Prompt Encoder

↓

Mask Decoder

↓

Predicted Mask

↓

Loss

↓

Backward Pass

↓

Gradients

↓

Optimizer

↓

Updated Parameters
```

If you're using LoRA:

* ViT weights stay frozen.
* LoRA matrices receive gradients.
* The optimizer updates only those LoRA matrices.

---

# Summary

Think of training like hiking downhill in dense fog:

* The **loss** tells you how high you are.
* The **gradient** tells you which direction goes downhill.
* The **learning rate** determines how large a step you take.
* **Gradient Descent** repeatedly takes small downhill steps until the model reaches a low-loss solution.

---

# Key Takeaways

1. A gradient measures how the loss changes when a weight changes.
2. Gradient Descent updates weights in the opposite direction of the gradient.
3. The learning rate controls the update size.
4. `loss.backward()` computes gradients—it does **not** update weights.
5. `optimizer.step()` applies those gradients to update trainable parameters.
6. LoRA uses the same training process but updates only the LoRA parameters.

---

# Mini Exercise

Answer these without looking back:

1. What is a gradient in your own words?
2. Why do we move opposite the gradient?
3. What happens if the learning rate is too large?
4. What is the difference between `loss.backward()` and `optimizer.step()`?
5. During LoRA fine-tuning, which parameters receive gradient updates?

---

# 🔥 Before We Continue

From the next lesson onward, we enter what I consider the **core of deep learning**.

Everything you've learned so far (neurons, activations, loss, gradients) prepares you for one algorithm that made modern AI possible.

> **Backpropagation**

Without backpropagation:

* CNNs wouldn't work.
* Vision Transformers wouldn't work.
* SAM wouldn't work.
* LoRA wouldn't work.
* GPT wouldn't exist.

Many tutorials treat backpropagation as a few equations. We won't.

We'll build it **from scratch**, starting with a single neuron and tracing every derivative by hand, so that when you later see `loss.backward()`, you'll know exactly what PyTorch is computing internally. This understanding will make debugging and modifying models like SAM much easier.
