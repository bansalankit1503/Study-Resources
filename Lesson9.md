Lesson 9 — How Are Query, Key, and Value Created?

> Goal: Today we'll move from intuition to actual Transformer implementation. By the end of this lesson, you'll understand where Query (Q), Key (K), and Value (V) come from, why we need three different weight matrices, and how this is implemented in SAM's Vision Transformer.




---

Recap

Last lesson, we learned:

Input Token
      │
      ▼
Query (Q)

Key (K)

Value (V)

The next question is obvious:

> Where do Q, K, and V come from?




---

Step 1 — Input Embeddings

Suppose we have a sentence:

I love AI

After tokenization and embedding, assume each word is represented by a 4-dimensional vector.

"I"      → [0.2, 0.8, 0.1, 0.4]

"love"   → [0.6, 0.3, 0.9, 0.2]

"AI"     → [0.7, 0.5, 0.4, 0.8]

These vectors are called embeddings.


---

Step 2 — Why Can't We Use Embeddings Directly?

A common question is:

> Why not compare these embeddings directly?



Because one vector cannot perform three different jobs well.

Imagine one employee in a company.

Can the same person be:

CEO

Accountant

Security Guard


at the same time?

Probably not.

Different jobs require different skills.

Similarly,

an embedding needs three specialized versions.


---

Three Different Roles

From one embedding, we create:

Embedding
     │
     ├──► Query
     │
     ├──► Key
     │
     └──► Value

Each one has a different purpose.

Vector	Job

Query	What am I looking for?
Key	What information do I have?
Value	What information should I share?



---

Step 3 — Linear Layers

Remember from Lesson 2:

A linear layer computes:

Output = (Input × Weight) + Bias

Transformers use three different linear layers.

Input Embedding
      │
      ├────────────► Linear Layer Q ─────► Query
      │
      ├────────────► Linear Layer K ─────► Key
      │
      └────────────► Linear Layer V ─────► Value

Notice something important.

The input is identical.

Only the weights are different.


---

Example

Suppose the embedding is

[2, 5, 1]

After passing through three different linear layers, we might get

Query

[4, 8, 2]


Key

[1, 6, 3]


Value

[9, 2, 5]

The numbers are different because each linear layer has learned different weights.


---

Why Different Weights?

Imagine interviewing candidates.

The HR team looks at:

Communication

Personality


The Engineering team looks at:

Coding

Problem-solving


The Finance team looks at:

Budget awareness


Same candidate.

Different evaluation criteria.

Similarly,

Q, K, and V look at different aspects of the same embedding.


---

Tensor Shapes

Suppose

Batch Size = 2

Sequence Length = 5

Embedding Dimension = 8


Input shape:

(2, 5, 8)

Meaning:

2 Sentences

Each has 5 words

Each word has an 8-dimensional embedding

After the Q layer:

Q Shape

(2, 5, 8)

After the K layer:

K Shape

(2, 5, 8)

After the V layer:

V Shape

(2, 5, 8)

Notice:

The shape usually stays the same.

Only the values change.


---

Visual Representation

Input Embeddings
                    (2,5,8)
                       │
      ┌────────────────┼────────────────┐
      │                │                │
      ▼                ▼                ▼
   Linear Q         Linear K        Linear V
      │                │                │
      ▼                ▼                ▼
      Q                K                V
   (2,5,8)          (2,5,8)          (2,5,8)


---

PyTorch Implementation

This is almost exactly what happens inside a Transformer.

import torch
import torch.nn as nn

embed_dim = 768

q_layer = nn.Linear(embed_dim, embed_dim)
k_layer = nn.Linear(embed_dim, embed_dim)
v_layer = nn.Linear(embed_dim, embed_dim)

x = torch.randn(2, 5, embed_dim)

Q = q_layer(x)
K = k_layer(x)
V = v_layer(x)

print(Q.shape)
print(K.shape)
print(V.shape)

Output:

torch.Size([2, 5, 768])

torch.Size([2, 5, 768])

torch.Size([2, 5, 768])


---

How Does SAM Do This?

Inside the Vision Transformer,

each image patch becomes an embedding.

Suppose an image is divided into 196 patches.

Image

↓

196 Patches

↓

Patch Embeddings

↓

Linear Q

Linear K

Linear V

↓

Attention

Every image patch gets its own Q, K, and V vectors.


---

Real Tensor Shapes in SAM

A typical SAM ViT-B model might have:

Batch Size = 1

Number of Patches = 4096

Embedding Dimension = 768

Input:

(1, 4096, 768)

After Q:

(1, 4096, 768)

After K:

(1, 4096, 768)

After V:

(1, 4096, 768)

These are the tensors used in the next Attention calculation.


---

Where Is This in the Code?

A simplified Transformer block looks like this:

self.q_proj = nn.Linear(dim, dim)

self.k_proj = nn.Linear(dim, dim)

self.v_proj = nn.Linear(dim, dim)

Then during the forward pass:

Q = self.q_proj(x)

K = self.k_proj(x)

V = self.v_proj(x)

This is one of the first things you'll find when reading a Vision Transformer implementation.


---

Common Mistakes

Mistake 1

Thinking Q, K, and V come from different inputs.

❌ Wrong.

They all come from the same input embedding.


---

Mistake 2

Thinking the three linear layers share weights.

❌ Wrong.

Each has its own independently learned weight matrix.


---

Mistake 3

Thinking Q, K, and V are the final output.

❌ Wrong.

They are intermediate representations used to compute Attention.


---

How This Connects to LoRA

This is one of the most important connections for your project.

When people say:

> "Apply LoRA to the Query and Value layers"



they mean adding trainable low-rank adapters to these linear projections.

Conceptually:

Original Query Linear Layer
        │
        ├── Frozen Original Weights
        │
        └── LoRA Adapter (Trainable)

Original Value Linear Layer
        │
        ├── Frozen Original Weights
        │
        └── LoRA Adapter (Trainable)

This is why understanding these projection layers is essential before diving into LoRA.


---

Interview Questions

Q1. Why are three different linear layers used?

Answer: They transform the same embedding into three specialized representations: Query for searching, Key for matching, and Value for carrying information.


---

Q2. Do Q, K, and V have the same shape?

Answer: Usually yes. They have the same tensor shape, but different values because they are produced by different learned weight matrices.


---

Q3. Why is LoRA commonly applied to Q and V projections?

Answer: These projection layers have a large influence on how Attention behaves. Adapting them often provides strong fine-tuning performance while keeping the number of trainable parameters small.


---

Mini Assignment

1. Why can't we use the embedding directly instead of creating Q, K, and V?


2. Why do the three projection layers use different weights?


3. If the input tensor shape is (4, 128, 512), what will be the shapes of Q, K, and V?


4. Why are Q, K, and V considered intermediate representations rather than final outputs?


5. In your own words, explain why LoRA is often inserted into the Query and Value projection layers.




---

Next Lesson (Lesson 10)

Now we'll finally answer the question everyone associates with Transformers:

> How does a Query decide which Keys are important?



You'll learn:

Why we compute Query × Key

What an attention score really means

Why a dot product measures similarity

Tensor shapes for every operation

How the Attention Score Matrix is built

How this works for both text and image patches in Vision Transformers and SAM


This is the lesson where you'll see the core computation that powers every Transformer.
