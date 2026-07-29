🧠 Transformer Course

📚 Module 4 — Building a Complete Transformer Block

🚀 Lesson 15 — Layer Normalization (LayerNorm)

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ What Layer Normalization is.

✅ Why Transformers use LayerNorm instead of BatchNorm.

✅ What happens inside LayerNorm mathematically.

✅ Why every Transformer block has LayerNorm.

✅ How SAM, GPT, BERT, and BERT all depend on it.




---

📍 Course Progress

Multi-Head Attention        ✅

↓

Residual Connection        ✅

↓

⭐ Layer Normalization      ← Today

↓

Feed Forward Network

↓

Residual Connection

↓

Output


---

🧠 Think Before Reading

Imagine you're preparing for an exam.

You studied:

Math = 95

Physics = 50

Chemistry = 80

Another student has

Math = 40

Physics = 38

Chemistry = 42

Can we compare these scores directly?

Maybe not.

Different exams can have different difficulty levels.

Instead, we often normalize the scores so they're on a comparable scale.

LayerNorm does exactly that—but for neural network features.


---

🤔 The Problem

Suppose after Attention we get this vector:

[1000, 2, -50, 700, 5]

Notice something.

One feature is 1000, another is 2.

Huge differences like this make training unstable.


---

⚠️ Why Is This Bad?

Imagine a classroom discussion.

Student A speaks at volume 100.

Student B speaks at volume 2.

Student C speaks at volume 5.

Can the teacher hear everyone equally?

No.

One student dominates.

Neural networks face the same issue.

Some features become extremely large, while others become tiny.


---

💡 The Solution

LayerNorm rescales the features so they have a similar distribution.

Before

[1000, 2, -50, 700, 5]

After LayerNorm

[1.52, -0.64, -0.79, 0.98, -0.61]

Notice:

The values are now much closer together.

Large numbers no longer dominate.

Training becomes much more stable.



---

🎨 Visual

Before LayerNorm

Feature Values

1000 ████████████████████████████

700  ████████████████████

5    █

2    █

-50

After LayerNorm

Feature Values

1.52 ██

0.98 ██

-0.61 ██

-0.64 ██

-0.79 ██


---

🧠 What Does LayerNorm Actually Normalize?

Suppose one token has

Embedding

[4, 8, 10, 2]

LayerNorm looks only at these four numbers.

It computes:

1️⃣ Mean

2️⃣ Standard Deviation

3️⃣ Normalize every feature

The result has:

Mean ≈ 0

Standard Deviation ≈ 1



---

📦 Important Difference

Suppose we have

Shape

(Batch, Tokens, Embedding)

(2,5,768)

LayerNorm works across

Embedding Dimension

Not across

Batch

Tokens



---

Visualization

Batch 1

Token 1

[768 Features]

↓

Normalize These Features

-------------------------

Batch 1

Token 2

[768 Features]

↓

Normalize These Features

-------------------------

Batch 2

Token 1

[768 Features]

↓

Normalize These Features

Every token is normalized independently.


---

🤔 Why Not BatchNorm?

This is one of the biggest interview questions.

Let's compare them.

BatchNorm	LayerNorm

Uses Batch Statistics	Uses Feature Statistics
Depends on Batch Size	Independent of Batch Size
Great for CNNs	Perfect for Transformers
Difficult with Batch Size = 1	Works perfectly with Batch Size = 1



---

🎯 Example

Suppose your batch contains only

1 Image

BatchNorm struggles because there isn't enough data to estimate batch statistics.

LayerNorm says

> "No problem. I'll normalize each sample independently."



That's why Transformers use LayerNorm.


---

🎨 Transformer Flow

Input

↓

Multi-Head Attention

↓

Residual Add

↓

⭐ LayerNorm

↓

Feed Forward Network

↓

Residual Add

↓

LayerNorm

Notice

Every Transformer block usually contains two LayerNorm operations.


---

📦 Tensor Shapes

Input

(2,5,768)

After LayerNorm

(2,5,768)

🟢 Important

LayerNorm never changes tensor shape.

Only the values change.


---

💻 PyTorch Example

import torch
import torch.nn as nn

layer_norm = nn.LayerNorm(768)

x = torch.randn(2,5,768)

y = layer_norm(x)

print(y.shape)

Output

torch.Size([2,5,768])


---

🧠 Interactive Check

Suppose

Input

(4,100,512)

Question

What will be the output shape after

nn.LayerNorm(512)

Pause...

👇

Answer

(4,100,512)


---

🚀 How SAM Uses LayerNorm

SAM's Vision Transformer processes thousands of image patches.

Each patch has an embedding.

Example

Patch 1

[768 Features]

↓

LayerNorm

↓

Normalized Features

Then

Patch 2

↓

LayerNorm

↓

Normalized Features

Every patch is normalized independently.

This keeps the training stable even for very deep Vision Transformers.


---

🎯 Where Does LayerNorm Happen?

A simplified Transformer block looks like this:

x = x + attention(x)

x = layer_norm(x)

x = x + mlp(x)

x = layer_norm(x)

Notice

LayerNorm is applied after every Residual Connection in this simplified example.

(We'll discuss another variant called Pre-LayerNorm later, because modern models like GPT-2, GPT-3, LLaMA, and SAM often use a slightly different ordering.)


---

⚠️ Common Mistakes

❌ Mistake 1

Thinking LayerNorm changes tensor shape.

Wrong.

Only the values change.


---

❌ Mistake 2

Thinking LayerNorm is the same as BatchNorm.

Wrong.

They normalize across different dimensions.


---

❌ Mistake 3

Thinking LayerNorm is optional.

Wrong.

Removing it often makes Transformer training unstable or much harder.


---

🛠️ Debug Like an ML Engineer

❌ Error

nn.LayerNorm(512)

Input

(2,5,768)

Result

RuntimeError:
Given normalized_shape=[512],
expected input with last dimension = 512,
but got 768

✅ Why?

The last dimension of the input must match the normalized_shape.

Correct:

nn.LayerNorm(768)


---

🧩 Quick Quiz

Question 1

LayerNorm normalizes across:

A. Batch

B. Features

C. Tokens


<details>
<summary>✅ Answer</summary>B. Features (Embedding Dimension)

</details>
---

Question 2

Does LayerNorm change tensor shape?

<details>
<summary>✅ Answer</summary>No.

Only the values change.

</details>
---

Question 3

Why is LayerNorm preferred in Transformers?

<details>
<summary>✅ Answer</summary>Because it works independently for each token and doesn't depend on the batch size.

</details>
---

📌 Key Takeaways

> ✅ LayerNorm stabilizes training by normalizing feature values.



> ✅ It operates independently for each token.



> ✅ It does not change tensor shape.



> ✅ It is different from BatchNorm.



> ✅ Every modern Transformer—including SAM, ViT, GPT, BERT, and LLaMA—relies on LayerNorm.




---

🏗️ Transformer Block So Far

Input

↓

Multi-Head Attention

↓

➕ Residual

↓

✅ LayerNorm

↓

❓ Feed Forward Network (MLP)

↓

➕ Residual

↓

✅ LayerNorm

↓

Output


---

🚀 Next Lesson — Feed Forward Network (MLP)

We've finished the Attention part of the Transformer.

Now comes a question many beginners ask:

> If Attention already mixes information between tokens, why do we still need another neural network afterward?



In the next lesson, you'll learn:

🧠 What the Feed Forward Network (MLP) is.

📈 Why it expands the embedding dimension (e.g., 768 → 3072 → 768).

⚡ Why GELU is used instead of ReLU.

🔄 Why every token passes through the same MLP.

🚀 How the MLP significantly increases the model's expressive power in SAM, ViT, GPT, and BERT.


> Preview: By the end of the next lesson, you'll understand every component inside a Transformer block. After that, we'll assemble all the pieces into a complete Transformer and start connecting them directly to the original "Attention Is All You Need" paper and the SAM source code.
