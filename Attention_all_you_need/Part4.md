🧠 Transformer Course

📚 Module 8 — Reading the Original Research Paper

🚀 Lesson 25 — The Final Linear Projection (Wₒ): The Missing Piece of Multi-Head Attention

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ Why concatenating attention heads is not enough.

✅ What the Output Projection Matrix (Wₒ) does.

✅ Why every Transformer has one final Linear layer after Multi-Head Attention.

✅ How this maps directly to PyTorch, ViT, and SAM.




---

📍 Course Progress

Scaled Dot-Product Attention     ✅

↓

Multi-Head Attention             ✅

↓

⭐ Output Projection (Wₒ)         ← Today

↓

Complete Encoder Block Math

↓

Vision Transformer (ViT)

↓

SAM Source Code


---

🧠 Think Before Reading

Imagine 12 experts analyze the same problem.

Each expert writes a report.

Now you have 12 separate reports.

Can you directly send all 12 reports to the CEO?

Probably not.

Someone needs to:

Read every report

Combine the important information

Create one final report


👉 That "someone" is Wₒ (the Output Projection Matrix).


---

🔄 Let's Review Multi-Head Attention

So far we know:

Input

↓

Create Q, K, V

↓

Split into Heads

↓

Attention in Each Head

↓

Concatenate

Many beginners think the story ends here.

It doesn't.

There is one more step.


---

📦 What Happens After Concatenation?

Suppose:

12 Heads

↓

64 Features Each

After concatenation:

64

+

64

+

64

...

↓

768 Features

Now we have one large vector again.

But...

those features came from 12 independent heads.

They haven't been mixed together yet.


---

🎨 Visual Example

Imagine four heads.

Head 1

[Dog Features]

Head 2

[Color Features]

Head 3

[Shape Features]

Head 4

[Background Features]

Concatenation gives:

[Dog | Color | Shape | Background]

These features are simply placed side by side.

No interaction has happened between them yet.


---

🤔 Why Isn't Concatenation Enough?

Concatenation only joins vectors.

It does not learn how to combine information.

Think of it like this.

Imagine four students submit four different project reports.

Stapling the reports together...

📄 + 📄 + 📄 + 📄

doesn't produce one coherent report.

Someone must read them and combine the ideas.


---

💡 Enter Wₒ

The Transformer applies one more Linear layer.

Concatenated Heads

↓

Linear Layer (Wₒ)

↓

Final Output

This layer learns how to combine information from all heads.


---

🧮 Mathematical View

Suppose

Concatenated Output

Shape

(Batch, Tokens, 768)

Then

Wₒ

Shape

(768, 768)

Output

(Batch, Tokens, 768)

Notice:

The shape stays the same.

But the values change because the linear layer mixes information across all features.


---

🎨 Visualization

Without Wₒ

Head 1

↓

Head 2

↓

Head 3

↓

Head 4

↓

Just Joined Together


---

With Wₒ

Head 1

↘

Head 2

→ Mix Together

Head 3

↗

Head 4

↓

Final Representation


---

🧠 What Does Wₒ Learn?

During training,

Wₒ learns questions like:

Which head is most useful?

Which features should be emphasized?

Which features should be reduced?

How should information from different heads be combined?


No one programs these rules manually.

The model learns them through gradient descent.


---

📦 Tensor Shapes

Suppose

Batch = 2

Tokens = 5

Heads = 12

Head Dimension = 64

After attention:

(2,12,5,64)

Concatenate:

(2,5,768)

Apply Wₒ:

(2,5,768)

The shape doesn't change.

Only the representation becomes richer.


---

💻 PyTorch Example

A simplified implementation looks like this:

import torch.nn as nn

output_projection = nn.Linear(768, 768)

output = output_projection(concatenated_heads)

This is exactly what happens after concatenating all attention heads.


---

🔍 Where Is Wₒ in PyTorch?

When you use:

nn.MultiheadAttention(...)

PyTorch internally performs:

Q Projection

↓

K Projection

↓

V Projection

↓

Multi-Head Attention

↓

Concatenate Heads

↓

Output Projection (Wₒ)

So Wₒ is built into the module—you don't usually need to write it yourself.


---

🖼️ How ViT Uses Wₒ

A Vision Transformer computes attention between image patches.

Patch Attention

↓

Head 1

Head 2

...

Head 12

↓

Concatenate

↓

Wₒ

↓

Updated Patch Features

The output projection helps combine the information learned by different heads.


---

🚀 How SAM Uses Wₒ

SAM's Image Encoder is built from Vision Transformer blocks.

Every attention block follows this pattern:

Input Patches

↓

Q, K, V

↓

Multi-Head Attention

↓

Concatenate

↓

Wₒ

↓

Residual Connection

↓

LayerNorm

↓

Feed Forward Network

So yes—

every attention block in SAM also contains an output projection.


---

🛠️ Debug Like an ML Engineer

❌ Common Mistake

Many beginners think:

Concatenate

↓

Done

Wrong.

The complete attention pipeline is:

Attention

↓

Concatenate

↓

Linear (Wₒ)

↓

Residual

↓

LayerNorm

Skipping Wₒ changes the architecture.


---

🧠 Interactive Challenge

Suppose:

Embedding = 512

Heads = 8

Question:

What is the shape of Wₒ?

Pause...

👇

Answer:

(512, 512)

Why?

Because the concatenated output has 512 features, and we want the final output to also have 512 features while allowing the model to learn a new combination of those features.


---

🧩 Quick Quiz

Question 1

What does Wₒ do?

A. Creates Queries

B. Mixes information from all attention heads

C. Computes Softmax


<details>
<summary>✅ Answer</summary>B

</details>
---

Question 2

After concatenation, why do we still need another Linear layer?

<details>
<summary>✅ Answer</summary>Because concatenation only places the head outputs side by side. The output projection learns how to combine information across all heads.

</details>
---

Question 3

Does Wₒ change the embedding dimension?

<details>
<summary>✅ Answer</summary>Usually no. It typically maps from embed_dim back to embed_dim (for example, 768 → 768).

</details>
---

📌 Key Takeaways

> ✅ Concatenation alone does not combine information across heads.



> ✅ Wₒ is a trainable Linear layer applied after concatenation.



> ✅ Wₒ learns how to mix information from different attention heads.



> ✅ PyTorch's nn.MultiheadAttention includes this output projection internally.



> ✅ ViT and SAM use the same design.




---

🗺️ Complete Multi-Head Attention Pipeline

Input

↓

Linear Projections (Q, K, V)

↓

Split into Heads

↓

Scaled Dot-Product Attention

↓

Concatenate Heads

↓

Output Projection (Wₒ)

↓

Residual Connection

↓

LayerNorm

↓

Feed Forward Network

↓

Output


---

🎯 Checkpoint: You Now Understand Every Operation Inside Multi-Head Attention

You can now explain:

✅ Q, K, and V projections

✅ Dot-product attention

✅ Why we divide by √dₖ

✅ Softmax

✅ Multi-Head Attention

✅ Concatenation

✅ Output Projection (Wₒ)


This is a major milestone because you've covered the complete computation performed by the attention module.


---

🚀 Next Lesson — Layer Normalization: The Complete Mathematics

Earlier, we learned why LayerNorm is useful.

In the next lesson, we'll study its mathematics in depth:

🧮 How mean and variance are computed.

📐 Why normalization stabilizes training.

🔧 The trainable parameters γ (gamma) and β (beta).

🐍 The exact PyTorch implementation.

🖼️ How LayerNorm is used in ViT and SAM.


> Milestone: After the next lesson, you'll understand every mathematical operation inside a Transformer Encoder block, bringing you one step closer to confidently reading the SAM source code.
