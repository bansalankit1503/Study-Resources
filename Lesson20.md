🧠 Transformer Course

📚 Module 7 — Complete Transformer Architecture

🚀 Lesson 20 — The Transformer Decoder (How GPT Generates Text)

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ What a Transformer Decoder is.

✅ Why GPT uses only the Decoder.

✅ What Masked Self-Attention is.

✅ Why GPT cannot "see the future."

✅ The difference between Encoder, Decoder, and Encoder–Decoder models.




---

📍 Course Progress

Transformer Encoder          ✅

↓

⭐ Transformer Decoder       ← Today

↓

Masked Self-Attention

↓

Cross Attention

↓

Encoder-Decoder Transformer

↓

Vision Transformer (ViT)

↓

Segment Anything Model (SAM)


---

🧠 Think Before Reading

Imagine you're writing a sentence.

You have written:

I love

What should come next?

Possible answers:

I love pizza

I love coding

I love football

Notice something...

You can only use the words you've already written.

You cannot look into the future and see the next word.

This is exactly how the Transformer Decoder works.


---

🤔 Why Can't GPT Look at Future Words?

Suppose the sentence is

The cat is sleeping.

During training,

GPT predicts one word at a time.

Example:

Step 1

Input

The

Predict

cat


---

Step 2

Input

The cat

Predict

is


---

Step 3

Input

The cat is

Predict

sleeping

Notice

When predicting "is",

GPT must not already know

sleeping

Otherwise, learning would be too easy and unrealistic.


---

🚫 The Problem

Regular Self-Attention allows every token to attend to every other token.

Example

The

↓

cat

↓

is

↓

sleeping

Without restrictions,

the word

The

could directly look at

sleeping

That's cheating.


---

💡 The Solution — Masked Self-Attention

We hide future words.

Suppose we have

The

cat

is

sleeping

Attention matrix

The  cat  is  sleeping

The        ✅   ❌   ❌      ❌

cat        ✅   ✅   ❌      ❌

is         ✅   ✅   ✅      ❌

sleeping   ✅   ✅   ✅      ✅

Each word can only attend to

Itself

Previous words


Never future words.


---

🎨 Visualization

Without Mask

Word 1

↔ Word 2

↔ Word 3

↔ Word 4

Everyone sees everyone.


---

With Mask

Word1

↓

Word2 ← Word1

↓

Word3 ← Word2 ← Word1

↓

Word4 ← Word3 ← Word2 ← Word1

Future information is blocked.


---

🧠 What Does the Mask Look Like?

Before Softmax,

we modify the Attention Scores.

Example

0   4   2   8

5   1   7   6

9   3   2   1

4   6   8   5

Now apply the mask.

0   -∞   -∞   -∞

5    1   -∞   -∞

9    3    2   -∞

4    6    8    5


---

🤔 Why Use Negative Infinity?

Remember Softmax.

Softmax

↓

Large Negative Number

↓

Probability ≈ 0

After Softmax,

Impossible positions

↓

0 Probability

The model completely ignores future tokens.


---

🎨 Decoder Pipeline

Input Tokens

↓

Embedding

↓

Position Embedding

↓

Masked Self-Attention

↓

Residual

↓

LayerNorm

↓

Feed Forward Network

↓

Residual

↓

LayerNorm

↓

Next Decoder Block

Notice

The only difference from the Encoder is

⭐ Masked Self-Attention.


---

🚀 How GPT Generates Text

Suppose you type

I love

Step 1

Input

I love

Predict

coding


---

Step 2

Now input becomes

I love coding

Predict

because


---

Step 3

Input

I love coding because

Predict

...

The process repeats until the sentence is complete.

This is called autoregressive generation.


---

🆚 Encoder vs Decoder

Encoder	Decoder

Sees all input tokens	Sees only previous tokens
Uses normal Self-Attention	Uses Masked Self-Attention
Builds representations	Generates output token by token
Used in BERT & ViT	Used in GPT



---

🤔 What Is Cross-Attention?

The original Transformer (for translation) has both an Encoder and a Decoder.

Example

English

↓

Encoder

↓

Context

↓

Decoder

↓

French

Inside the Decoder,

there is an extra Attention layer.

Instead of attending only to previous French words,

it also attends to the Encoder output.

This is called Cross-Attention.


---

🎨 Original Transformer Decoder

Masked Self-Attention

↓

Cross Attention

↓

Feed Forward Network

Three major components.


---

🧠 Interactive Check

Suppose we're predicting

Token 4

Question:

Can Token 4 attend to

Token 6

Pause...

👇

Answer

❌ No

Because Token 6 belongs to the future.


---

💻 PyTorch Concept

A causal (future-blocking) mask can be created like this:

import torch

length = 4

mask = torch.triu(
    torch.ones(length, length),
    diagonal=1
)

print(mask)

Output

0 1 1 1

0 0 1 1

0 0 0 1

0 0 0 0

The 1s indicate positions that should be masked before applying Softmax.


---

🚀 Does SAM Use a Decoder?

This is an excellent question.

The answer is yes, but not the same kind of Decoder as GPT.

SAM has:

Image

↓

ViT Encoder

↓

Image Features

↓

Prompt Encoder

↓

Mask Decoder

↓

Segmentation Mask

Important:

❌ SAM's Mask Decoder is not the language-generation Transformer Decoder used in GPT.

✅ It is a task-specific decoder that combines image features and prompts to predict segmentation masks.


So the word "decoder" can mean different things depending on the model and task.


---

⚠️ Common Mistakes

❌ Mistake 1

Thinking GPT uses an Encoder.

Wrong.

GPT is a Decoder-only model.


---

❌ Mistake 2

Thinking the Decoder can see future words.

Wrong.

Masked Self-Attention prevents that.


---

❌ Mistake 3

Thinking every decoder is like GPT's Decoder.

Wrong.

SAM's Mask Decoder is designed for image segmentation, not text generation.


---

🧩 Quick Quiz

Question 1

Why does GPT use Masked Self-Attention?

A. To reduce memory

B. To prevent looking at future tokens

C. To increase embedding size


<details>
<summary>✅ Answer</summary>B

</details>
---

Question 2

Which model is Decoder-only?

A. BERT

B. ViT

C. GPT


<details>
<summary>✅ Answer</summary>C

</details>
---

Question 3

Does the original Transformer Decoder contain Cross-Attention?

<details>
<summary>✅ Answer</summary>Yes.

It attends to the Encoder output in addition to using Masked Self-Attention.

</details>
---

📌 Key Takeaways

> ✅ The Decoder generates output one token at a time.



> ✅ It uses Masked Self-Attention so future tokens cannot be seen.



> ✅ GPT is a Decoder-only Transformer.



> ✅ The original Transformer uses both an Encoder and a Decoder connected by Cross-Attention.



> ✅ SAM has a Mask Decoder, but it is different from GPT's language-generation Decoder.




---

🏗️ Complete Architecture Comparison

BERT
Input
  │
Encoder
  │
Output


GPT
Input
  │
Decoder (Masked Attention)
  │
Generated Text


Original Transformer
Input
  │
Encoder
  │
Context
  │
Decoder
  │
Output


SAM
Image
  │
ViT Encoder
  │
Image Features
  │
Prompt Encoder
  │
Mask Decoder
  │
Segmentation Mask


---

🚀 Next Lesson — Encoder vs Decoder vs Encoder–Decoder (The Ultimate Comparison)

In the next lesson, we'll bring everything together with a detailed comparison.

You'll learn:

🧩 Why BERT, GPT, T5, ViT, and SAM chose different architectures.

📊 The strengths and weaknesses of each design.

🎯 Which architecture is best for classification, generation, translation, segmentation, and question answering.

🗺️ How to choose the right Transformer architecture for your own machine learning projects.


> Milestone: After the next lesson, you'll have a complete mental map of the entire Transformer family, making it much easier to understand modern AI models and research papers.
