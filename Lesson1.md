# Lesson 1 — How Does a Computer "See" the World?

**Course:** Transformers from Scratch → Vision Transformers → SAM → LoRA

---

## What you'll learn today

By the end of this lesson, you'll understand:

* ✅ How computers store information
* ✅ Why everything becomes numbers
* ✅ What tensors really are
* ✅ Why images become matrices
* ✅ Why text becomes vectors
* ✅ Why SAM can only process numbers
* ✅ Why Transformers are "just" mathematical functions on tensors

This lesson is the foundation for everything that follows.

---

# Imagine You're Teaching a Child

Suppose you show a child this picture:

🐶 Dog

The child immediately says:

> "Dog."

Easy.

Now imagine showing the same picture to a computer.

What does the computer see?

**Nothing.**

It doesn't see ears.

It doesn't see a tail.

It doesn't know what "dog" means.

It only understands **numbers**.

This is the first principle of machine learning:

> **Computers never understand images, text, or sound directly. They only process numbers.**

---

# Step 1 — Everything Becomes Numbers

Let's look at different kinds of data.

### Image

```
🐶 Dog
```

Computer sees something like

```
[
 [255, 120, 40],
 [210, 180, 60],
 ...
]
```

These are pixel values.

---

### Text

You read

```
I love AI
```

Computer sees

```
[1023, 85, 9001]
```

These are token IDs.

---

### Audio

You hear

```
Hello
```

Computer sees

```
[0.13, 0.26, -0.71, ...]
```

These are sound amplitudes sampled over time.

---

### Video

A video is simply

```
Image
↓

Image
↓

Image
↓

Image
↓

...
```

Each frame is an image represented by numbers.

---

# Everything Is a Tensor

This is one of the most important ideas in deep learning.

A **tensor** is simply a container for numbers arranged in one or more dimensions.

Examples:

### Scalar (0D)

```
5
```

One number.

---

### Vector (1D)

```
[5 3 8]
```

A list of numbers.

---

### Matrix (2D)

```
[
 [1 2 3]
 [4 5 6]
]
```

A grid of numbers.

---

### 3D Tensor

Think of stacking multiple matrices.

```
Layer 1

[
 [1 2]
 [3 4]
]

Layer 2

[
 [5 6]
 [7 8]
]
```

---

### 4D Tensor

Imagine stacking many 3D tensors.

This is exactly how batches of images are stored during training.

---

# Where Does SAM Fit?

Suppose your input image is

```
1024 × 1024
```

A color image has **3 channels**:

* Red
* Green
* Blue

So SAM receives a tensor shaped like

```
3 × 1024 × 1024
```

If you train with a batch of 8 images:

```
8 × 3 × 1024 × 1024
```

This is often written as:

```
(Batch, Channels, Height, Width)
```

or

```
(B, C, H, W)
```

If you ever print the shape of an image tensor in PyTorch, you'll see something like:

```python
torch.Size([8, 3, 1024, 1024])
```

That's not just metadata—it's telling you exactly how the numbers are organized.

---

# Why Images Become Matrices

Consider a tiny grayscale image:

```
⬜⬛⬜
⬛⬜⬛
⬜⬛⬜
```

The computer stores brightness values.

```
[
 [255,   0, 255],
 [  0, 255,   0],
 [255,   0, 255]
]
```

Each number corresponds to one pixel.

---

# Color Images

Each pixel has three values:

```
Red
Green
Blue
```

For example:

```
Pixel

R = 200

G = 100

B = 50
```

So a color image is actually three matrices stacked together.

```
Red Matrix

Green Matrix

Blue Matrix
```

This forms a 3D tensor.

---

# What About Text?

Sentence:

```
The cat sleeps
```

The tokenizer converts it into IDs:

```
[125, 402, 89]
```

The embedding layer then transforms each ID into a vector.

Instead of

```
125
```

it becomes something like

```
[0.14, -0.82, 1.21, ...]
```

Maybe 768 numbers.

This is called an **embedding**.

Transformers don't operate on token IDs—they operate on these embedding vectors.

---

# Why Embeddings?

Think about three words:

```
King

Queen

Apple
```

Token IDs might be:

```
King  = 100

Queen = 101

Apple = 500
```

Notice that the IDs themselves don't express meaning.

To the computer:

```
100

101

500
```

There is no indication that "King" and "Queen" are semantically related.

An embedding converts each word into a vector where similar words occupy nearby regions in a high-dimensional space.

Conceptually:

```
King   → ●

Queen  → ●

Apple                    ○
```

Now the model can learn relationships because similar meanings correspond to similar vectors.

---

# Why Does Everything Become Vectors?

Because neural networks perform operations like:

```
Addition

Subtraction

Dot Product

Matrix Multiplication
```

These operations require numerical inputs.

Words, images, and audio are all transformed into vectors or tensors before any learning happens.

---

# The Big Picture

Imagine a pipeline:

```
Real World
      │
      ▼
 Image / Text / Audio
      │
      ▼
 Convert to Numbers
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

Every deep learning model—from CNNs to Transformers to SAM—starts with this exact idea.

---

# How This Relates to SAM

Suppose you're fine-tuning SAM with LoRA.

When you call something like:

```python
masks = sam(image)
```

What actually happens?

```
Image
   │
   ▼
Tensor
   │
   ▼
Patch Embedding
   │
   ▼
Vision Transformer
   │
   ▼
Prompt Encoder
   │
   ▼
Mask Decoder
   │
   ▼
Segmentation Mask
```

Every stage manipulates tensors.

There is no hidden magic—just increasingly sophisticated mathematical transformations.

---

# Key Takeaways

By the end of this lesson, you should remember these five ideas:

1. Computers only understand numbers.
2. Images, text, and audio are converted into tensors.
3. A tensor is a multi-dimensional array of numbers.
4. Neural networks operate entirely on tensors.
5. SAM and Transformers never "see" images—they only process numerical representations.

---

# Mini Exercise

Try answering these without looking back:

1. Why can't a computer understand a JPEG image directly?
2. What is the difference between a vector and a matrix?
3. If an RGB image has shape `(3, 512, 512)`, what do the three dimensions represent?
4. Why are token IDs not enough for a Transformer?
5. What is an embedding, in your own words?

---

# What's Next? (Lesson 2)

In the next lesson, we'll answer a deceptively simple question:

> **How does a neural network learn from numbers?**

We'll build the **first artificial neuron from scratch**, understand weights, biases, activation functions, and forward propagation—laying the groundwork for everything that eventually becomes the Transformer. From that point onward, every component of SAM and LoRA will build on concepts you've already mastered.
