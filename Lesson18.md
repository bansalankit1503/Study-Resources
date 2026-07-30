🧠 Transformer Course

📚 Module 6 — Positional Information

🚀 Lesson 18 — Positional Encoding (How Transformers Understand Order)

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ Why Transformers cannot understand order by themselves.

✅ Why positional information is necessary.

✅ The difference between Positional Encoding and Positional Embedding.

✅ How Vision Transformers and SAM encode image positions.

✅ Why positional information is one of the most important parts of a Transformer.




---

📍 Course Progress

Complete Transformer Block     ✅

↓

⭐ Positional Encoding         ← Today

↓

Stack Multiple Blocks

↓

Transformer Encoder

↓

Vision Transformer (ViT)

↓

Segment Anything Model (SAM)


---

🧠 Think Before Reading

Suppose I give you these words:

Apple

ate

John

Can you understand the sentence?

Not really.

Now look at this:

John ate Apple

Now it makes sense.

Now another order:

Apple ate John

Same words.

Different order.

Completely different meaning.


---

🤔 The Problem

A Transformer sees the input like this:

Embedding 1

Embedding 2

Embedding 3

Without extra information,

the Transformer only knows

Three vectors exist.

It does not know

Which one came first?

Which one came second?

Which one came last?


---

🎨 Imagine Shuffling Cards

Suppose your cards are

A

K

Q

J

Shuffle them.

Q

A

J

K

The cards are still the same.

Only their positions changed.

Without remembering positions,

you can't know the original order.

The Transformer has exactly the same problem.


---

🚀 Why Doesn't Attention Know Position?

Remember Attention.

Q × Kᵀ

↓

Softmax

↓

Attention × V

Notice something.

Where did we ever use

Position 1

Position 2

Position 3

Nowhere.

Attention only compares vectors.

It has no built-in understanding of order.


---

💡 The Solution

We give every token a position vector.

Instead of

Word Embedding

we use

Final Input

=

Word Embedding

+

Position Encoding

This is one of the most important equations in Transformers.


---

🎨 Visualization

Suppose

Word

Cat

Embedding

[2, 8, 5, 1]

Position Encoding

[0.1, 0.4, -0.2, 0.6]

Final Input

[2.1, 8.4, 4.8, 1.6]

Now

the model knows both

what the word is

where the word appears



---

📦 Complete Pipeline

Words

↓

Word Embedding

↓

Position Encoding

↓

Add Together

↓

Transformer Block


---

🧠 Why Addition?

Many beginners ask

> Why not concatenate?



Good question.

Suppose

Embedding

768

Position

768

Concatenation becomes

1536

Now every Transformer layer would have to process twice as many features.

Instead,

we simply add them.

768

+

768

↓

768

The tensor size remains unchanged.


---

📦 Tensor Shapes

Suppose

Batch = 2

Tokens = 5

Embedding = 768

Word Embedding

(2,5,768)

Position Encoding

(5,768)

When added, PyTorch automatically broadcasts the position encoding across the batch.

Final Input

(2,5,768)


---

🌊 The Original Paper's Positional Encoding

The original paper introduced a fixed sinusoidal positional encoding.

Instead of learning positions,

they are computed using sine and cosine waves.

Visualization:

Position

0 → ~~~

1 → ~~~~~

2 → ~~~~~~~

3 → ~~~~~~~~~

Different embedding dimensions use waves with different frequencies.


---

🤔 Why Sine and Cosine?

The intuition is:

Nearby positions have similar encodings.

Far-away positions have different encodings.

The pattern naturally extends to sequences longer than those seen during training.


The original paper chose this because it provides a smooth, deterministic way to represent position.


---

🎯 Learned Positional Embeddings

Modern Transformers often use a different approach.

Instead of fixed values,

they learn position vectors during training.

Example

Position 1

↓

[0.2,1.4,-0.3,...]

Position 2

↓

[-1.2,0.8,0.9,...]

Position 3

↓

[0.6,-0.4,2.1,...]

These vectors are updated by gradient descent just like other model parameters.


---

🆚 Positional Encoding vs Positional Embedding

Positional Encoding	Positional Embedding

Fixed	Learned
No training	Learned during training
Uses sine & cosine	Uses trainable vectors
Original Transformer	Common in modern models


Both solve the same problem.

They just use different methods.


---

🖼️ What About Images?

Images are 2D, not 1D.

Example

□□□□□□□□

□□□□□□□□

□□□□□□□□

□□□□□□□□

Suppose we divide the image into patches.

P1  P2  P3  P4

P5  P6  P7  P8

P9 P10 P11 P12

Every patch has

Row position

Column position


Unlike words,

image patches have both horizontal and vertical locations.


---

🚀 How Vision Transformers Handle This

Each image patch gets

Patch Embedding

+

2D Position Embedding

Now the Transformer knows

Which patch is in the top-left

Which patch is in the center

Which patch is in the bottom-right


Without positional information,

the image would just look like a bag of patches.


---

🎨 SAM Example

Suppose SAM receives this image.

🐶 🌳

🌸 ⚽

It splits it into patches.

P1 P2

P3 P4

Now each patch receives

Patch Embedding

+

Position Information

So SAM understands that

Dog

↓

Top Left

Tree

↓

Top Right

Flower

↓

Bottom Left

Ball

↓

Bottom Right

Without position embeddings,

SAM wouldn't know where objects are located.

For a segmentation model, that would be a huge problem.


---

💻 PyTorch Example (Concept)

import torch
import torch.nn as nn

embedding = torch.randn(2, 5, 768)

position = nn.Parameter(torch.randn(5, 768))

x = embedding + position

print(x.shape)

Output

torch.Size([2,5,768])

Notice the shape doesn't change.


---

🛠️ Debug Like an ML Engineer

❌ Common Error

embedding.shape = (2, 5, 768)

position.shape = (6, 768)

Trying to add them causes an error because the number of positions doesn't match the number of tokens.

Always verify:

Number of Tokens

=

Number of Position Vectors


---

🧩 Quick Quiz

Question 1

Why do Transformers need positional information?

A. To reduce memory usage

B. Because Attention alone doesn't know token order

C. To speed up training


<details>
<summary>✅ Answer</summary>B

</details>
---

Question 2

What is the basic equation?

<details>
<summary>✅ Answer</summary>Final Input

=

Embedding

+

Position Encoding

</details>
---

Question 3

Which type of positional information is commonly used in modern Vision Transformers?

A. Random vectors

B. Learned positional embeddings (often 2D for images)

C. No positional information


<details>
<summary>✅ Answer</summary>B

</details>
---

📌 Key Takeaways

> ✅ Attention has no built-in understanding of order.



> ✅ Positional information tells the model where each token or patch is located.



> ✅ The original Transformer used sinusoidal positional encodings.



> ✅ Many modern models use learned positional embeddings.



> ✅ Vision Transformers and SAM use positional information for image patches, often in two dimensions.




---

🏗️ Transformer Pipeline So Far

Input

↓

Embedding

↓

➕ Positional Information

↓

Transformer Block 1

↓

Transformer Block 2

↓

Transformer Block 3

↓

...

↓

Final Features


---

🚀 Next Lesson — The Transformer Encoder (Complete Architecture)

You've now learned every major building block.

In the next lesson, we'll assemble them into the complete Transformer Encoder from the original "Attention Is All You Need" paper.

You'll learn:

🏛️ What an Encoder really is.

📚 Why multiple Transformer blocks are stacked.

🔄 How information becomes richer after each layer.

🖼️ Why Vision Transformers and SAM use only the Encoder.

🤖 Why models like BERT are encoder-only, GPT is decoder-only, and the original Transformer uses both an Encoder and a Decoder.


> Milestone: After the next lesson, you'll understand the complete architecture well enough to start reading the original Transformer paper and recognize how SAM's image encoder is built from these concepts.
