🧠 Transformer Course

📚 Module 8 — Reading the Original Research Paper

🚀 Lesson 27 — Residual Connections: Why Deep Transformers Don't Collapse

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ Why Residual Connections are necessary.

✅ What the Vanishing Gradient Problem is.

✅ Why we add the input back to the output.

✅ How gradients flow through Residual Connections.

✅ How Residual Connections are implemented in PyTorch, ViT, and SAM.




---

📍 Course Progress

LayerNorm Mathematics        ✅

↓

⭐ Residual Connections       ← Today

↓

Complete Encoder Block

↓

SAM Source Code

↓

Fine-Tuning SAM


---

🧠 Think Before Reading

Imagine you're climbing a mountain.

There are two ways to reach the top.

Path 1

Start

↓

Checkpoint 1

↓

Checkpoint 2

↓

Checkpoint 3

↓

Top

If one checkpoint is blocked...

❌ You can't continue.


---

Path 2

Now imagine there is also a shortcut.

Start

↓

Checkpoint 1

↓

Checkpoint 2

↓

Checkpoint 3

↓

Top

╲_______________________╱
      Shortcut

Even if one checkpoint becomes difficult,

you still have another path.

👉 A Residual Connection is like that shortcut.


---

🤔 The Problem Without Residual Connections

Suppose we have a deep network.

Layer 1

↓

Layer 2

↓

Layer 3

↓

Layer 4

↓

Layer 5

↓

...

Each layer changes the input a little.

As the network becomes deeper,

two major problems appear:

📉 Vanishing gradients

📈 Exploding gradients


Today we'll focus on the first one.


---

🌊 What Is the Vanishing Gradient Problem?

Remember training.

Forward Pass

↓

Compute Loss

↓

Backward Pass

During backpropagation,

the gradient travels backward.

Loss

↑

Layer 10

↑

Layer 9

↑

Layer 8

↑

...

↑

Layer 1

As it moves backward,

the gradient can become smaller and smaller.

Eventually,

Gradient ≈ 0

If the gradient is almost zero,

the weights stop learning.


---

🎨 Water Pipe Analogy

Imagine water flowing through a very long pipe.

🚰

↓

Pipe

↓

Pipe

↓

Pipe

↓

Pipe

↓

Small Drip 💧

The farther the water travels,

the weaker it becomes.

Gradients behave similarly in very deep networks.


---

💡 The Solution

Instead of only computing

Output

=

Attention(Input)

we compute

Output

=

Input

+

Attention(Input)

This simple addition is called a Residual Connection (or Skip Connection).


---

🎯 Why Add the Input?

Imagine the Attention layer doesn't learn anything useful yet.

Without a Residual Connection:

Input

↓

Attention

↓

Poor Output

The original information is lost.


---

With a Residual Connection:

Input

↓

Attention

↓

+

Original Input

↓

Output

Even if Attention is imperfect,

the original information is preserved.


---

🧠 Mathematical Example

Suppose

Input:

[1, 2, 3]

Attention Output:

[0.5, 0.2, -0.1]

Residual:

[1,2,3]

+

[0.5,0.2,-0.1]

=

[1.5,2.2,2.9]

Notice:

The model learns a modification to the input instead of replacing it completely.


---

🎨 What Is the Model Really Learning?

Without Residual:

Learn Entire Output

With Residual:

Learn

Small Improvement

to Existing Output

This is much easier.

Think about editing a document.

Would you rather:

Rewrite the whole document?


or

Make a few corrections?


The second option is usually easier.


---

🌉 Why Does This Help Gradients?

This is the biggest advantage.

Backward pass without Residual:

Loss

↓

Layer

↓

Layer

↓

Layer

↓

Input

The gradient has only one path.


---

With Residual:

Loss

↓

Layer

↓

+

↓

Input

The gradient now has an additional route.

This makes it easier for information and gradients to flow through deep networks.


---

📦 Tensor Shapes

Suppose

Batch = 2

Tokens = 5

Embedding = 768

Input:

(2,5,768)

Attention Output:

(2,5,768)

Residual Addition:

Input

+

Attention Output

Output:

(2,5,768)

🟢 Rule

Residual addition requires both tensors to have exactly the same shape.


---

🐍 PyTorch Implementation

The code is surprisingly simple:

x = x + attention_output

Or, in a complete Transformer block:

x = x + self.attention(x)

x = x + self.ffn(x)

Those two + operations are the residual connections.


---

🖼️ How ViT Uses Residual Connections

Vision Transformer processes image patches.

Patch Features

↓

LayerNorm

↓

Attention

↓

Residual

↓

LayerNorm

↓

FFN

↓

Residual

Every Transformer block uses two Residual Connections.


---

🚀 How SAM Uses Residual Connections

SAM's Image Encoder is based on ViT.

Each block follows:

Input

↓

LayerNorm

↓

Attention

↓

+

Original Input

↓

LayerNorm

↓

FFN

↓

+

Previous Output

Exactly the same pattern.

This consistency is one reason Transformers can scale to dozens or even hundreds of layers.


---

⚠️ A Common Misunderstanding

Many beginners think:

> "Residual Connections improve accuracy by adding more information."



Not exactly.

Their primary purpose is to make optimization easier by improving the flow of information and gradients through deep networks.

Higher accuracy is often a consequence of better optimization, not the direct goal.


---

🛠️ Debug Like an ML Engineer

❌ Common Error

Trying to add tensors with different shapes.

Example:

Input

(2,5,768)

Attention Output

(2,5,512)

Adding them:

x + attention_output

❌ This will fail because the shapes don't match.

Always check:

Input Shape

=

Output Shape

before performing a Residual Connection.


---

🧠 Interactive Challenge

Suppose

Input:

[10, 20]

Attention Output:

[2, -3]

Question:

What is the Residual Output?

Pause...

👇

Answer:

[10,20]

+

[2,-3]

=

[12,17]


---

🧩 Quick Quiz

Question 1

Why do we use Residual Connections?

A. To reduce embedding size

B. To help gradients and preserve information

C. To replace LayerNorm


<details>
<summary>✅ Answer</summary>B

</details>
---

Question 2

What is the basic Residual formula?

<details>
<summary>✅ Answer</summary>Output

=

Input

+

Layer(Input)

</details>
---

Question 3

Must the two tensors have the same shape before addition?

<details>
<summary>✅ Answer</summary>Yes. Element-wise addition requires matching shapes (or compatible broadcasting, which is not what is used here).

</details>
---

📌 Key Takeaways

> ✅ Residual Connections add the original input back to the layer output.



> ✅ They help preserve information across many layers.



> ✅ They improve gradient flow during backpropagation, making deep networks easier to train.



> ✅ The input and output must have the same shape for the addition.



> ✅ Every Transformer block in ViT and SAM uses Residual Connections.




---

🗺️ Complete Modern Transformer Block (Pre-LayerNorm)

Input

↓

LayerNorm

↓

Multi-Head Attention

↓

Output Projection (Wₒ)

↓

Residual Connection

↓

LayerNorm

↓

Feed Forward Network

↓

Residual Connection

↓

Output


---

🎉 Major Milestone

Congratulations!

You now understand the mathematics behind every major component of a modern Transformer Encoder block:

✅ Q, K, V projections

✅ Scaled Dot-Product Attention

✅ Softmax

✅ Multi-Head Attention

✅ Output Projection (Wₒ)

✅ Layer Normalization

✅ Residual Connections

✅ Feed Forward Networks


You're now very close to reading real Transformer implementations.


---

🚀 Next Lesson — Building a Complete Transformer Encoder Block in PyTorch (Line by Line)

Starting in the next lesson, we'll stop looking at isolated components and begin assembling them into a complete Transformer block.

We'll cover:

🧩 Writing the full TransformerBlock class.

🔍 Explaining every line of PyTorch code.

📐 Tracking tensor shapes through the forward pass.

🖼️ Mapping each line directly to Vision Transformer (ViT) and SAM source code.


> Milestone: After the next lesson, you'll be able to open a real Transformer implementation and understand it almost line by line—a crucial step toward reading and modifying the SAM Image Encoder.
