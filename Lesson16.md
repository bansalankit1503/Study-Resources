🧠 Transformer Course

📚 Module 4 — Building a Complete Transformer Block

🚀 Lesson 16 — Feed Forward Network (FFN / MLP)

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ What the Feed Forward Network (FFN) is.

✅ Why Attention alone is not enough.

✅ Why the embedding dimension is expanded and compressed.

✅ Why GELU is used.

✅ How FFN works in SAM, ViT, GPT, and BERT.




---

📍 Course Progress

Multi-Head Attention       ✅

↓

Residual Connection        ✅

↓

LayerNorm                 ✅

↓

⭐ Feed Forward Network    ← Today

↓

Residual Connection

↓

LayerNorm

↓

Complete Transformer Block


---

🧠 Think Before Reading

Imagine you're reading a book.

Step 1:

You collect information from many pages.

Chapter 1

Chapter 2

Chapter 3

This is similar to Attention.

But after collecting the information...

Do you instantly understand everything?

No.

Your brain thinks, analyzes, and forms new ideas.

That thinking process is exactly what the Feed Forward Network (FFN) does.


---

🤔 Why Isn't Attention Enough?

Let's see what Attention actually does.

Token A

↓

Looks at Token B

↓

Looks at Token C

↓

Looks at Token D

Attention is excellent at sharing information between tokens.

But it does not perform deep computation on each token.

That is the FFN's job.


---

💡 A Simple Analogy

Imagine a classroom.

Step 1: Discussion (Attention)

Students discuss with each other.

Everyone exchanges ideas.


---

Step 2: Individual Thinking (FFN)

Each student now sits quietly and thinks.

Student 1 → Think

Student 2 → Think

Student 3 → Think

No communication happens here.

Each student processes their own understanding.

This is exactly how FFN works.


---

🎨 Transformer Flow

Input

↓

Attention

↓

Everyone Shares Information

↓

FFN

↓

Everyone Thinks Individually


---

📦 FFN Architecture

A Feed Forward Network is surprisingly simple.

Linear

↓

GELU

↓

Linear

That's all.


---

📐 Why Two Linear Layers?

Suppose

Embedding = 768

Instead of

768

↓

768

Transformers do

768

↓

3072

↓

768

Notice

The embedding becomes 4 times larger.


---

🤔 Why Expand?

Imagine you have a small office.

🏠 Small Room

Five people need to brainstorm.

Very little space.

Now imagine moving everyone into a huge conference room.

🏢 Large Conference Room

Now there is much more space for ideas.

The FFN does the same thing mathematically.

It temporarily expands the feature space so the model can learn richer representations.


---

🎨 Visualization

Before

□□□□□□□□□□□□□□□□
768 Features

Expand

□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
3072 Features

Compress

□□□□□□□□□□□□□□□□
768 Features


---

🧠 Why Compress Again?

Suppose every Transformer block produced larger embeddings.

768

↓

3072

↓

12288

↓

49152

The model would quickly become impossible to run.

Instead,

the FFN expands temporarily, performs computation, and then compresses back to the original size.


---

📦 Tensor Shapes

Input

(2,5,768)

First Linear Layer

(2,5,3072)

GELU

(2,5,3072)

Second Linear Layer

(2,5,768)

Notice

The output shape is the same as the input shape.

This is necessary because a residual connection comes next.


---

🎯 Why GELU?

Earlier in the course, we learned about activation functions.

Why don't Transformers use ReLU?

Because GELU is smoother.

ReLU

Negative Numbers

↓

Become 0


---

GELU

Negative Numbers

↓

Reduced Smoothly

This smoother behavior generally improves Transformer training.

That's why almost every modern Transformer uses GELU (or a similar smooth activation such as SwiGLU in some newer models).


---

💻 PyTorch Implementation

import torch
import torch.nn as nn

ffn = nn.Sequential(
    nn.Linear(768, 3072),
    nn.GELU(),
    nn.Linear(3072, 768)
)

x = torch.randn(2, 5, 768)

y = ffn(x)

print(y.shape)

Output

torch.Size([2, 5, 768])


---

🧠 What Happens Internally?

Suppose one token is

[2, 5, 8]

After the first Linear layer

[1.2, 4.8, 7.3, 2.9, 6.1, ...]

After GELU

[1.1, 4.8, 7.2, 2.8, 6.0, ...]

After the second Linear layer

[3.5, 4.2, 6.1]

The token has learned a more expressive representation.


---

🎨 Does the FFN Mix Tokens?

This is one of the most important concepts.

Suppose we have

Token 1

Token 2

Token 3

Attention

Token 1 ←→ Token 2

Token 2 ←→ Token 3

Token 3 ←→ Token 1

Information flows between tokens.


---

FFN

Token 1 → FFN

Token 2 → FFN

Token 3 → FFN

Every token is processed independently.

The same FFN weights are applied to every token.


---

> 💡 Key Insight

Attention mixes information between tokens.

FFN transforms information within each token.




Both are essential.


---

🚀 How SAM Uses FFN

SAM's Vision Transformer first lets image patches communicate.

Patch 1

↓

Attention

↓

Understands Other Patches

Then

Patch 1

↓

FFN

↓

Improves Its Own Representation

Every image patch goes through the same FFN.

This greatly improves the quality of the learned features before they are sent to the next Transformer block.


---

🛠️ Debug Like an ML Engineer

❌ Common Mistake

Using the wrong dimensions.

nn.Linear(768, 1024)

followed by

nn.Linear(3072, 768)

This will fail because the second layer expects an input of size 3072, but receives 1024.

Always ensure:

First Linear Output Size

=

Second Linear Input Size


---

🧩 Interactive Check

Suppose

Embedding = 512

Question:

If the FFN expansion ratio is 4×,

what is the hidden dimension?

Pause...

👇

Answer

512 × 4

=

2048


---

🧩 Quick Quiz

Question 1

Which component allows tokens to communicate?

A. FFN

B. Attention

C. LayerNorm


<details>
<summary>✅ Answer</summary>B. Attention

</details>
---

Question 2

Which component processes each token independently?

A. Multi-Head Attention

B. FFN

C. Softmax


<details>
<summary>✅ Answer</summary>B. FFN

</details>
---

Question 3

If the embedding dimension is 1024 and the FFN uses a 4× expansion, what are the dimensions of the two linear layers?

<details>
<summary>✅ Answer</summary>1024

↓

4096

↓

1024

</details>
---

🏗️ The Complete Transformer Block

Now you've learned every major component.

Input
                  │
                  ▼
          Multi-Head Attention
                  │
                  ▼
           ➕ Residual Add
                  │
                  ▼
             LayerNorm
                  │
                  ▼
         Feed Forward Network
                  │
                  ▼
           ➕ Residual Add
                  │
                  ▼
             LayerNorm
                  │
                  ▼
                Output


---

📌 Key Takeaways

> ✅ Attention lets tokens exchange information.



> ✅ FFN processes each token independently.



> ✅ The FFN consists of Linear → GELU → Linear.



> ✅ It temporarily expands the embedding (e.g., 768 → 3072 → 768).



> ✅ The output shape matches the input shape so residual connections can be applied.



> ✅ Every Transformer model, including SAM, ViT, GPT, and BERT, contains an FFN after the attention layer.




---

🚀 Next Lesson — Building a Complete Transformer Block (End-to-End)

In the next lesson, we'll assemble everything you've learned into one complete Transformer block.

You'll learn:

🧩 The exact execution order inside a Transformer block.

📐 Tensor shapes at every stage.

🐍 A PyTorch implementation of a complete block.

🖼️ How this maps directly to the Vision Transformer used in SAM.

📄 How to recognize each part when reading the "Attention Is All You Need" paper and the SAM source code.


> Milestone: After the next lesson, you'll be able to read a real Transformer implementation and identify every major component with confidence.
