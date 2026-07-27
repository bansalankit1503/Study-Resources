# Lesson 4 — How Does a Neural Network Know It's Wrong?

> **Goal:** By the end of this lesson, you'll understand what a **loss function** is, why it is the "teacher" of a neural network, and how it drives learning in models ranging from a simple classifier to **SAM** and **LoRA**.

This is one of the most important concepts in machine learning.

---

# Recap

So far we know:

```text
Image
   │
   ▼
Tensor
   │
   ▼
Neural Network
   │
   ▼
Prediction
```

Suppose the network predicts:

```text
Cat
```

But the correct answer is:

```text
Dog
```

How does the network know it made a mistake?

---

# Imagine Taking an Exam

Suppose you answer 10 questions.

After finishing, your teacher checks your paper.

| Question | Your Answer | Correct |
| -------- | ----------- | ------- |
| 1        | ✅           | ✅       |
| 2        | ❌           | ✅       |
| 3        | ✅           | ✅       |
| 4        | ❌           | ✅       |

Finally your teacher says:

> **Score: 8/10**

That score tells you **how wrong** you were.

A neural network needs something similar.

That score is called the **Loss**.

---

# What is Loss?

Loss is simply

> **A number that tells the model how bad its prediction is.**

Small loss

↓

Good prediction

Large loss

↓

Bad prediction

---

# Example

Suppose we're predicting house prices.

Actual price:

```text
$500,000
```

Prediction:

```text
$495,000
```

Error:

```text
$5,000
```

Very good.

Now another prediction:

```text
$300,000
```

Error:

```text
$200,000
```

Very bad.

Loss converts this difference into a mathematical value.

---

# Loss is Like GPS

Imagine driving to your destination.

Google Maps constantly tells you

> "You are 15 miles away."

As you drive,

```text
15

↓

10

↓

5

↓

1

↓

0
```

You're getting closer.

Training is exactly like this.

The destination is

> Minimum Loss.

---

# The Training Loop

Every neural network repeats the same process.

```text
Input

↓

Prediction

↓

Calculate Loss

↓

Update Weights

↓

New Prediction

↓

Calculate Loss

↓

Update Weights

↓

Repeat
```

This happens millions of times.

---

# Example: Dog Classifier

Suppose we have an image.

Correct label:

```text
Dog
```

The model predicts probabilities:

```text
Dog = 0.40

Cat = 0.60
```

Wrong.

Loss becomes high.

After learning:

```text
Dog = 0.97

Cat = 0.03
```

Loss becomes much smaller.

---

# Why Probability?

Notice the model didn't simply output

```text
Dog
```

Instead it outputs

```text
Dog : 97%

Cat : 3%
```

Because probabilities contain much more information.

---

# Different Problems Need Different Loss Functions

Think about sports.

Football and cricket have different scoring systems.

Similarly,

Regression problems

↓

One loss

Classification

↓

Another loss

Segmentation

↓

Another loss

---

# Regression Loss

Suppose we predict temperature.

Correct:

```text
30°C
```

Prediction:

```text
28°C
```

Difference:

```text
2°C
```

The most common regression loss is

## Mean Squared Error (MSE)

Formula:

[
Loss = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y_i})^2
]

Let's understand why.

---

# Why Square the Error?

Suppose

Prediction:

```text
40
```

Truth:

```text
50
```

Difference:

```text
-10
```

Another example

Prediction:

```text
60
```

Truth:

```text
50
```

Difference:

```text
+10
```

If we simply average:

```text
-10 + 10 = 0
```

That incorrectly suggests there is no error.

Squaring fixes this:

```text
(-10)² = 100

(+10)² = 100
```

Now both contribute equally.

---

# Why Large Errors Matter More

Suppose two predictions:

Prediction A

```text
Error = 2
```

Squared:

```text
4
```

Prediction B

```text
Error = 10
```

Squared:

```text
100
```

Notice how a large mistake is penalized much more heavily.

That's desirable in many regression tasks.

---

# Classification Needs Something Different

Suppose there are three animals.

```text
Dog

Cat

Horse
```

Prediction:

```text
Dog = 0.60

Cat = 0.35

Horse = 0.05
```

Correct answer:

```text
Cat
```

The model is confidently wrong.

We need a loss that strongly penalizes this behavior.

That's why classification usually uses **Cross-Entropy Loss**.

---

# Intuition Behind Cross-Entropy

Imagine taking a multiple-choice exam.

Correct answer:

```text
B
```

Case 1

You say

```text
A

99% confident
```

Teacher:

> Very wrong.

Large penalty.

Case 2

You say

```text
B

99% confident
```

Teacher:

> Excellent.

Very small penalty.

Cross-entropy encourages:

* High confidence when correct
* Low confidence when wrong

---

# Binary vs Multi-Class Classification

Suppose you're detecting spam.

Only two classes.

```text
Spam

Not Spam
```

Use **Binary Cross-Entropy**.

Now suppose you're recognizing animals.

```text
Dog

Cat

Horse

Tiger

Elephant
```

Use **Categorical Cross-Entropy** (often implemented in PyTorch as `CrossEntropyLoss`).

---

# What About Segmentation?

This is where your SAM project becomes relevant.

Instead of predicting

```text
Dog
```

SAM predicts

```text
Pixel 1

Pixel 2

Pixel 3

...

Pixel 1,048,576
```

Each pixel is classified as belonging to an object or not.

That means the loss is computed over the entire predicted mask.

Modern segmentation models often combine several losses, such as:

* Cross-Entropy Loss
* Dice Loss
* IoU-based Loss
* Focal Loss (for difficult or imbalanced examples)

These combinations help produce accurate object boundaries.

---

# Example

Ground Truth Mask

```text
██████

██████

......
```

Predicted Mask

```text
████..

████..

......
```

Some pixels are wrong.

The loss measures how different these masks are.

The smaller the difference,

↓

Better segmentation.

---

# Where Does LoRA Fit?

Suppose you fine-tune SAM using LoRA.

The pipeline becomes:

```text
Image

↓

SAM

↓

Predicted Mask

↓

Loss

↓

Update ONLY LoRA Parameters

↓

Better Mask
```

Notice something important.

The loss doesn't care whether you're training:

* the whole model,
* only LoRA layers,
* or just a decoder.

It simply measures prediction quality.

The optimizer decides which trainable parameters to update.

---

# PyTorch Example

Regression

```python
import torch
import torch.nn as nn

prediction = torch.tensor([28.0])
target = torch.tensor([30.0])

loss_fn = nn.MSELoss()

loss = loss_fn(prediction, target)

print(loss)
```

---

Classification

```python
import torch
import torch.nn as nn

logits = torch.tensor([[2.1, 0.5, 1.2]])
target = torch.tensor([0])

loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(logits, target)

print(loss)
```

Notice something important.

We passed **logits**, not probabilities.

PyTorch applies **Softmax internally** for numerical stability.

This is a common interview question.

---

# The Big Picture

Everything you've learned so far fits together:

```text
Image

↓

Tensor

↓

Neural Network

↓

Prediction

↓

Loss Function

↓

Optimizer

↓

Updated Weights

↓

Better Prediction
```

The optimizer (which we'll study next) uses the loss to determine **how** to update the weights.

---

# Why This Matters for SAM

When training or fine-tuning SAM:

```python
loss = criterion(predicted_masks, ground_truth_masks)

loss.backward()

optimizer.step()
```

These three lines are the heart of training.

By the end of this course, you'll understand exactly what every one of them is doing internally.

---

# Key Takeaways

Remember these ideas:

1. Loss measures how wrong a prediction is.
2. Lower loss means better predictions.
3. MSE is commonly used for regression.
4. Cross-Entropy is commonly used for classification.
5. Segmentation models often combine multiple losses.
6. During LoRA fine-tuning, the loss is computed the same way—it simply updates fewer trainable parameters.

---

# Mini Assignment

Without looking back, answer these:

1. Why can't a neural network learn without a loss function?
2. Why do we square errors in MSE?
3. Why isn't MSE typically used for classification?
4. Why does `CrossEntropyLoss` expect logits instead of probabilities in PyTorch?
5. When fine-tuning SAM with LoRA, which parameters are updated after the loss is computed?

---

# Next Lesson (Lesson 5)

Now comes one of the biggest breakthroughs in machine learning:

> **How does the network actually change its weights to reduce the loss?**

We'll learn:

* What is a gradient?
* Why derivatives matter
* Intuition behind calculus in deep learning
* Gradient Descent
* Learning Rate
* Why models sometimes fail to converge
* How PyTorch's `loss.backward()` computes gradients automatically

After Lesson 5, you'll understand what happens behind the scenes every time you call:

```python
loss.backward()
```

—which is one of the most fundamental operations in training models like **SAM**, **ViT**, and **LoRA**.
