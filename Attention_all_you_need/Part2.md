🧠 Transformer Course

📚 Module 8 — Reading the Original Research Paper

🚀 Lesson 23 — The Most Important Equation in Transformers (Scaled Dot-Product Attention)

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ The most famous equation in the Transformer paper.

✅ Why we divide by √dₖ.

✅ What dₖ actually means.

✅ What happens if we don't divide.

✅ How this equation maps directly to PyTorch and SAM.




---

📍 Course Progress

Original Paper Overview        ✅

↓

⭐ Scaled Dot-Product Attention ← Today

↓

Multi-Head Attention Math

↓

Transformer Paper (Full)

↓

Vision Transformer (ViT)

↓

SAM Source Code


---

🧠 Think Before Reading

Remember how we computed attention?

Q × Kᵀ

↓

Softmax

↓

Attention Weights

↓

× V

That was the simplified version.

The actual equation from the paper adds one more important step.


---

📜 The Original Equation

The paper defines attention as:

Attention(Q, K, V)

=

Softmax(

(Q × Kᵀ)

/

√dₖ

)

×

V

This equation powers:

✅ GPT

✅ BERT

✅ ViT

✅ SAM

✅ Almost every modern Transformer


If you understand this equation, you've understood the mathematical heart of Transformers.


---

🧩 Let's Break It Down

Instead of reading the whole equation at once, let's understand it piece by piece.

Attention(Q, K, V)

=

Softmax(

(Q × Kᵀ)

/ √dₖ

)

×

V

We'll go from left to right.


---

Step 1️⃣ — Compute Similarity

We first calculate

Q × Kᵀ

Example:

Q

(5 × 64)

@

Kᵀ

(64 × 5)

↓

Scores

(5 × 5)

Each value answers:

> "How much should one token pay attention to another token?"




---

Step 2️⃣ — What is dₖ?

This is one of the most confusing symbols for beginners.

dₖ means:

Dimension of each Key vector

Example:

Suppose

Embedding Size = 768

Heads = 12

Then

dₖ

=

768 / 12

=

64

Each attention head works with vectors of size 64.


---

🎨 Visualization

Embedding

768

↓

Split into 12 Heads

↓

64
64
64
64
64
64
64
64
64
64
64
64

Each head has

dₖ = 64


---

🤔 Why Divide by √dₖ?

This is the most important question.

Let's understand it with numbers.

Suppose

dₖ = 4

Query

[2, 3, 1, 4]

Key

[3, 2, 5, 1]

Dot product

2×3

+

3×2

+

1×5

+

4×1

=

6

+

6

+

5

+

4

=

21

Now imagine

dₖ = 512

The dot product will typically be much larger because many more numbers are being added together.


---

📈 Bigger Dimensions → Bigger Scores

As the dimension increases:

dₖ = 4

↓

Score ≈ 20

dₖ = 64

↓

Score ≈ 100

dₖ = 512

↓

Score ≈ Very Large

The exact values vary, but the trend is the same: larger vectors tend to produce larger dot products.


---

😨 Why Is That a Problem?

Remember Softmax.

Suppose the scores are

2

3

4

Softmax gives

0.09

0.24

0.67

Nice and balanced.


---

Now imagine

200

300

400

Softmax becomes approximately

0

0

1

One token gets almost all the attention.

The others get almost none.


---

⚠️ Why Is That Bad?

During training,

the gradients become very small for the ignored positions.

This makes optimization more difficult and learning less stable.


---

💡 The Solution

We scale the scores.

Instead of

Scores

we use

Scores

/

√dₖ

Suppose

dₖ = 64

Then

√64 = 8

If the score is

80

After scaling

80

/

8

=

10

The values become more moderate before Softmax.


---

🎨 Visual Comparison

Without Scaling

Scores

↓

250

400

390

↓

Softmax

↓

0

1

0


---

With Scaling

Scores

↓

31

50

48

↓

Softmax

↓

0.01

0.87

0.12

The exact probabilities depend on the scores, but scaling generally prevents Softmax from becoming excessively "peaked."


---

🧠 Intuition

Think of a classroom.

Without scaling,

one student answers every question.

Alice

100%

Everyone Else

0%

With scaling,

more students can contribute.

Alice

55%

Bob

25%

Charlie

20%

The model can consider multiple relevant tokens instead of focusing almost entirely on one.


---

🖼️ How SAM Uses This

SAM's Vision Transformer computes attention between image patches.

Patch 1

↓

Q

Patch 2

↓

K

Then

Q × Kᵀ

↓

÷ √dₖ

↓

Softmax

↓

Attention

↓

× V

Exactly the same equation is used.

The only difference is that the "tokens" are image patches instead of words.


---

💻 PyTorch Implementation

This equation maps almost directly to code.

import math
import torch

scores = torch.matmul(Q, K.transpose(-2, -1))

scores = scores / math.sqrt(d_k)

weights = torch.softmax(scores, dim=-1)

output = torch.matmul(weights, V)

Every line corresponds to one part of the mathematical equation.


---

🛠️ Debug Like an ML Engineer

❌ Common Mistake

Many beginners write:

scores = Q @ K.transpose(-2, -1)

weights = torch.softmax(scores, dim=-1)

and forget

scores = scores / math.sqrt(d_k)

The code will still run, but for larger key dimensions the attention distribution can become overly sharp, making training less stable.


---

🧠 Interactive Challenge

Suppose

Embedding = 1024

Heads = 16

Question:

What is

dₖ

Pause...

👇

Answer

1024 / 16

=

64

Now,

what is

√dₖ

Answer

√64

=

8

So every attention score is divided by 8 before Softmax.


---

📌 Key Takeaways

> ✅ dₖ is the dimension of each Key (and Query) vector within a single attention head.



> ✅ Larger vector dimensions tend to produce larger dot products.



> ✅ Dividing by √dₖ keeps the scores in a range where Softmax behaves more smoothly.



> ✅ This improves training stability.



> ✅ This equation is used in GPT, BERT, ViT, and SAM.




---

🗺️ Complete Attention Pipeline

Input

↓

Q, K, V

↓

Q × Kᵀ

↓

÷ √dₖ

↓

Softmax

↓

Attention Weights

↓

× V

↓

Output


---

🚀 Next Lesson — Why Multi-Head Attention Is More Powerful Than Single-Head Attention

You already know what Multi-Head Attention is.

In the next lesson, we'll go deeper into why it works so well.

We'll answer questions like:

🧠 Why not just use one very large attention head?

🎯 What different heads learn in practice?

👀 How different heads focus on different relationships.

🖼️ How multiple attention heads help Vision Transformers and SAM understand complex images.


> Milestone: After the next lesson, you'll understand why simply increasing the embedding size cannot replace Multi-Head Attention, which is a key design principle in modern Transformer architectures.
