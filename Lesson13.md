🧠 Transformer Course

📚 Module 3 — Attention Mechanism

🚀 Lesson 12 — Multiplying Attention Weights with Values (V)

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ Why we multiply Attention Weights × Values

✅ Why we don't multiply with Queries or Keys

✅ What the final output of Attention actually represents

✅ Tensor shapes at every step

✅ How this happens inside SAM, ViT, GPT, and BERT




---

📍 Course Progress

Embeddings                 ✅

↓

Q, K, V                   ✅

↓

Q × Kᵀ                    ✅

↓

Softmax                   ✅

↓

⭐ Attention Weights × V   ← Today

↓

Final Attention Output

↓

Multi-Head Attention


---

🧠 Think Before Reading

Imagine you're searching for books in a library.

You already did this:

🔍 Found relevant books

⭐ Gave each book a relevance score


Now the question is...

> Do you take the scores home?



❌ No.

You take the books.

The scores only help you decide which books matter most.

This is exactly what happens in Attention.


---

🔄 Recap

After Softmax we have:

Attention Weights

[0.10, 0.20, 0.70]

These numbers mean

Give

10% importance to Word 1

20% importance to Word 2

70% importance to Word 3

But...

There is no actual information yet.

The information is stored in Values (V).


---

🎁 What is Value (V)?

Remember our analogy.

Component	Meaning

Query	What am I searching for?
Key	How do I match?
⭐ Value	The actual information


Imagine Google Search.

Search Query

↓

Find Relevant Websites

↓

Open the Websites

↓

Read the Information

The website content is the Value.


---

🎨 Example

Suppose we have three Values.

V1

[1, 2]

V2

[5, 4]

V3

[9, 8]

Attention Weights

0.10

0.20

0.70

Question:

How much of each Value should contribute?


---

🧮 Step-by-Step Calculation

Multiply each Value by its weight.

0.10 × [1,2]

=

[0.1,0.2]


---

0.20 × [5,4]

=

[1.0,0.8]


---

0.70 × [9,8]

=

[6.3,5.6]


---

Now add them together.

[0.1,0.2]

+

[1.0,0.8]

+

[6.3,5.6]

=

[7.4,6.6]


---

🎯 Final Output

The final Attention output becomes

[7.4,6.6]

Notice something amazing.

It is not equal to

V1

V2

V3


Instead,

it's a new representation built from all three Values.


---

> 💡 Key Insight

Attention does not copy information.

It mixes information intelligently.




---

🎨 Visual Pipeline

Attention Weights

0.10

0.20

0.70

        │

        ▼

Multiply

        │

        ▼

Values

V1

V2

V3

        │

        ▼

Weighted Sum

        │

        ▼

New Representation


---

🤔 Why Don't We Multiply by Q or K?

Excellent question.

Remember the roles.

Component	Job

Query	Ask the question
Key	Find relevant information
Value	Carry information


Imagine asking a librarian.

Query

↓

Find Books

↓

Read Books

You don't take home

Your question


or

The library catalogue.


You take home

📚 The books.

That's why we multiply by Values.


---

📦 Tensor Shapes

Suppose

Batch = 2

Tokens = 5

Embedding = 8

Attention Weights

(2,5,5)

Values

(2,5,8)

Now perform

Output = Attention @ V

Shape calculation

(2,5,5)

@

(2,5,8)

↓

(2,5,8)


---

🎨 Shape Visualization

Attention

(B,T,T)

↓

(2,5,5)

        @

Value

(B,T,E)

↓

(2,5,8)

──────────────

Output

(B,T,E)

↓

(2,5,8)


---

🧠 Interactive Check

Question:

If

Attention

(1,100,100)

Value

(1,100,64)

What is the output shape?

Pause...

Think...

👇

Answer

(1,100,64)


---

💻 PyTorch Implementation

import torch
import torch.nn.functional as F

Q = torch.randn(2,5,8)
K = torch.randn(2,5,8)
V = torch.randn(2,5,8)

scores = Q @ K.transpose(-2,-1)

weights = F.softmax(scores, dim=-1)

output = weights @ V

print(output.shape)

Output

torch.Size([2,5,8])

Congratulations 🎉

You have now implemented the core computation of Attention.


---

🚀 How SAM Uses This

Imagine an image.

🐶 Dog

It is split into patches.

P1

P2

P3

...

Pn

Each patch creates

Query

Key

Value


After Softmax,

Patch 15 may decide

Patch 22

80%

Patch 31

15%

Patch 8

5%

Now its output becomes

0.80 × Value22

+

0.15 × Value31

+

0.05 × Value8

Notice

Patch 15 now contains information from other patches.

This is why Vision Transformers understand global relationships.


---

🎨 Complete Attention Pipeline

Input Embeddings

        │

        ▼

Linear Layers

        │

        ▼

Query   Key   Value

        │

        ▼

Q × Kᵀ

        │

        ▼

Attention Scores

        │

        ▼

Softmax

        │

        ▼

Attention Weights

        │

        ▼

Attention Weights × Value

        │

        ▼

✨ Final Attention Output


---

⚠️ Common Mistakes

❌ Mistake 1

Thinking the output is one of the original Values.

Wrong.

It is a weighted combination of many Values.


---

❌ Mistake 2

Thinking Softmax changes the Values.

Wrong.

Softmax only changes the Attention Scores.

Values remain unchanged until the multiplication step.


---

❌ Mistake 3

Thinking Queries and Keys are discarded.

Wrong.

They are essential for computing the Attention Weights.

Without them, the model wouldn't know which Values are important.


---

🧩 Quick Quiz

Question 1

What stores the actual information?

A. Query

B. Key

C. Value


<details>
<summary>✅ Answer</summary>C. Value

</details>
---

Question 2

Which operation produces the final Attention output?

A. Q × Kᵀ

B. Softmax

C. Attention Weights × Value


<details>
<summary>✅ Answer</summary>C

</details>
---

Question 3

If

Attention

(3,20,20)

Value

(3,20,64)

Output shape?

<details>
<summary>✅ Answer</summary>(3,20,64)

</details>
---

🛠️ Debug Like an ML Engineer

❌ Error

RuntimeError:
mat1 and mat2 shapes cannot be multiplied

✅ What to check?

1. Is K.transpose(-2, -1) applied before Q @ K?


2. Does the last dimension of Attention match the token dimension of V?


3. Print tensor shapes before every matrix multiplication:



print(Q.shape)
print(K.shape)
print(V.shape)
print(weights.shape)

This simple habit saves hours of debugging.


---

📌 Key Takeaways

> ✅ Q asks questions.



> ✅ K decides relevance.



> ✅ Softmax converts scores into probabilities.



> ✅ V contains the actual information.



> ✅ Attention Output = Attention Weights × Value.



> ✅ The output is a new context-aware representation, not a copy of any single Value.




---

🚀 Next Lesson — Multi-Head Attention

So far, we've used one Attention mechanism.

But the paper is called Multi-Head Attention for a reason.

In the next lesson, you'll learn:

🧠 Why one Attention head isn't enough.

👀 How different heads learn different relationships simultaneously.

📦 How tensors are split into multiple heads.

🔄 Why we later concatenate the heads.

🚀 How Multi-Head Attention is implemented in SAM, ViT, GPT, and BERT.


> Preview: Multi-Head Attention is where Transformers become dramatically more powerful than a single Attention mechanism. It's one of the defining innovations of the "Attention Is All You Need" paper.
