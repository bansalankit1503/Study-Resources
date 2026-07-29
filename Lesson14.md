🧠 Transformer Course

📚 Module 4 — Building a Complete Transformer Block

🚀 Lesson 14 — Residual Connections (Skip Connections)

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ What a Residual Connection is

✅ Why deep neural networks stop learning

✅ Why simply stacking more layers can make a model worse

✅ Why every Transformer uses Skip Connections

✅ How SAM, GPT, BERT, and ViT use them




---

📍 Course Progress

Multi-Head Attention          ✅

↓

⭐ Residual Connection         ← Today

↓

Layer Normalization

↓

Feed Forward Network (MLP)

↓

Complete Transformer Block


---

🧠 Think Before Reading

Imagine you're climbing a mountain.

There are two paths.

Path 1

Start

↓

Step 1

↓

Step 2

↓

Step 3

↓

Destination


---

Path 2

Start

├──────────────► Destination

│

└──► Step1

     ↓

   Step2

     ↓

   Step3

     ↓

Destination

The second path gives you a shortcut.

Even if the long path becomes difficult, you can still reach the destination.

That's exactly what a Residual Connection does.


---

🤔 The Problem

Suppose we build a neural network.

Input

↓

Layer 1

↓

Layer 2

↓

Layer 3

↓

Output

Now imagine adding 100 more layers.

Would the model always become better?

Most people would say

> "Yes."



But surprisingly...

❌ No.


---

📊 What Researchers Observed

Number of Layers	Accuracy

10	90%
20	93%
50	94%
100	91% ❌


Wait...

Adding more layers actually made the model worse!

Why?


---

🎯 The Real Problem

Suppose Layer 50 learns something useful.

That information has to pass through

Layer 51

↓

Layer 52

↓

Layer 53

↓

...

↓

Layer 100

During this long journey,

the information can become weaker or distorted.

Similarly,

during backpropagation, gradients also have to travel through all these layers.

They can become extremely small.

This is called the Vanishing Gradient Problem.


---

🎨 Visual

Without Residual Connection

Input

↓

Layer

↓

Layer

↓

Layer

↓

Layer

↓

Layer

↓

Output

If one layer changes the information badly,

everything after it suffers.


---

💡 The Brilliant Idea

Instead of forcing information through every layer,

why not also send the original input directly to the output?

Like this.

Input

│

├──────────────────────────┐

│                          │

▼                          │

Attention                  │

│                          │

▼                          │

Output                     │

▲                          │

└──────── Add Input ◄──────┘

This is called a Residual Connection.


---

📌 The Formula

Instead of

Output = Attention(Input)

Transformers compute

Output = Input + Attention(Input)

Notice the difference.

We are not replacing the input.

We are adding to it.


---

🎨 Example

Suppose

Input

[2, 5, 3]

Attention Output

[1, -1, 4]

Residual Connection

[2,5,3]

+

[1,-1,4]

=

[3,4,7]

The original information is still preserved.


---

> 💡 Key Insight

Attention learns a refinement, not a completely new representation.




---

🤔 Why Is This Better?

Imagine you're editing a document.

Without residual learning:

Old Document

↓

Delete Everything

↓

Write Again

Very risky.

With residual learning:

Old Document

↓

Keep It

↓

Add Improvements

Much easier.

That's exactly how Transformers learn.


---

📦 Tensor Shapes

Suppose

Input

(2,5,768)

Attention Output

(2,5,768)

Addition

output = input + attention_output

Result

(2,5,768)


---

🟢 Important Rule

Residual addition is only possible if the shapes are identical.

Input	Attention Output	Can Add?

(2,5,768)	(2,5,768)	✅ Yes
(2,5,768)	(2,5,512)	❌ No



---

🎨 Complete Flow

Input

      │

      ▼

Multi-Head Attention

      │

      ▼

Attention Output

      │

      ▼

Add Original Input

      │

      ▼

Residual Output


---

💻 PyTorch Example

import torch

x = torch.randn(2, 5, 768)

attention_output = torch.randn(2, 5, 768)

output = x + attention_output

print(output.shape)

Output

torch.Size([2,5,768])

Very simple.

But this one line changed deep learning forever.


---

🚀 How SAM Uses Residual Connections

Inside every Vision Transformer block in SAM:

Image Patches

↓

Multi-Head Attention

↓

Residual Add

↓

LayerNorm

↓

MLP

↓

Residual Add

↓

Next Transformer Block

Notice something important.

There isn't just one residual connection.

There are two in every Transformer block:

1. After Multi-Head Attention


2. After the Feed Forward Network (MLP)



We'll understand the second one soon.


---

🧠 Interactive Check

Suppose

Input

[5,2,1]

Attention Output

[-1,3,4]

Pause...

Can you compute the Residual Output?

👇

Answer

[5,2,1]

+

[-1,3,4]

=

[4,5,5]


---

🎯 Why Does This Help Training?

Imagine the Attention layer learns nothing useful.

Its output becomes

[0,0,0]

Residual Output

Input

+

0

=

Input

The model can simply preserve the original information.

This makes optimization much easier.


---

⚠️ Common Mistakes

❌ Mistake 1

Thinking the residual connection skips computation.

Wrong.

The Attention layer is still fully computed.

The input is simply added back afterward.


---

❌ Mistake 2

Thinking residual means concatenation.

Wrong.

Residual uses element-wise addition, not concatenation.


---

❌ Mistake 3

Thinking residual connections are only used in Transformers.

Wrong.

They were first popularized in ResNet for computer vision and later became a core part of Transformers.


---

🛠️ Debug Like an ML Engineer

❌ Error

RuntimeError:

The size of tensor a (768)

must match tensor b (512)

✅ Why?

Because

output = x + attention_output

requires both tensors to have exactly the same shape.

Always check:

print(x.shape)
print(attention_output.shape)


---

🧩 Quick Quiz

Question 1

Residual Connection performs:

A. Multiplication

B. Addition

C. Concatenation


<details>
<summary>✅ Answer</summary>B. Addition

</details>
---

Question 2

Why do residual connections help?

<details>
<summary>✅ Answer</summary>They preserve the original information and make it easier for gradients to flow through very deep networks.

</details>
---

Question 3

Can you add tensors with shapes

(2,5,768)

+

(2,5,512)

?

<details>
<summary>✅ Answer</summary>No.

The shapes must match exactly for element-wise addition.

</details>
---

📌 Key Takeaways

> ✅ Residual Connections add the original input back to the layer output.



> ✅ Formula:

Output = Input + Layer(Input)



> ✅ They help preserve information.



> ✅ They improve gradient flow during training.



> ✅ Every Transformer block uses residual connections.



> ✅ SAM, GPT, BERT, and ViT all rely on this mechanism.




---

🏗️ Transformer Block So Far

Input

↓

Multi-Head Attention

↓

➕ Residual Connection ✅

↓

❓ Layer Normalization ← Next Lesson

↓

Feed Forward Network

↓

➕ Residual Connection

↓

Output


---

🚀 Next Lesson — Layer Normalization

You've probably noticed something interesting.

After the residual addition, every Transformer immediately applies LayerNorm.

Why?

If we're already preserving information with residual connections, why normalize it again?

In the next lesson, you'll learn:

📊 What Layer Normalization is.

⚖️ Why Transformers normalize after residual connections.

🔍 The difference between BatchNorm and LayerNorm.

🧠 Why LayerNorm works well even with a batch size of 1.

🚀 How LayerNorm stabilizes training in SAM, GPT, BERT, and ViT.


> Preview: Layer Normalization is one of the most misunderstood parts of Transformers. Once you understand it, the entire Transformer block will start fitting together like puzzle pieces.
