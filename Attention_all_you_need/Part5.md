🧠 Transformer Course

📚 Module 8 — Reading the Original Research Paper

🚀 Lesson 26 — Layer Normalization: The Complete Mathematics

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ What Layer Normalization actually computes.

✅ Why normalization stabilizes training.

✅ What Mean, Variance, Standard Deviation, γ (Gamma) and β (Beta) are.

✅ How nn.LayerNorm() works internally.

✅ How LayerNorm is used in ViT and SAM.




---

📍 Course Progress

Multi-Head Attention         ✅

↓

Output Projection (Wₒ)       ✅

↓

⭐ LayerNorm Mathematics      ← Today

↓

Feed Forward Network Math

↓

Vision Transformer (ViT)

↓

SAM Source Code


---

🧠 Think Before Reading

Imagine you are a teacher.

You have marks of 5 students:

20

40

60

80

100

Another class has:

520

540

560

580

600

Both classes have the same spread.

But their numbers are on completely different scales.

If we compare them directly, it isn't fair.

So we normalize them.

Transformers do something very similar.


---

🤔 Why Do We Need LayerNorm?

As data passes through many Transformer layers:

Input

↓

Attention

↓

FFN

↓

Attention

↓

FFN

↓

Attention

The values inside tensors may become:

Very Large

or

Very Small

This makes training harder.

LayerNorm keeps the values in a more stable range.


---

🎯 What Does LayerNorm Normalize?

Suppose one token has the embedding:

[2, 4, 6, 8]

Notice:

This token has 4 features.

LayerNorm computes statistics across these features.

Unlike BatchNorm, it does not compute statistics across the batch.


---

Step 1️⃣ — Compute the Mean

Mean means average.

Formula (Markdown style):

Mean = (Sum of all features) / (Number of features)

Example:

[2, 4, 6, 8]

Mean:

(2 + 4 + 6 + 8) / 4

=

20 / 4

=

5

So

Mean = 5


---

Step 2️⃣ — Subtract the Mean

Subtract 5 from every value.

2 - 5 = -3

4 - 5 = -1

6 - 5 =  1

8 - 5 =  3

Result:

[-3, -1, 1, 3]

Now the average is 0.


---

Step 3️⃣ — Compute the Variance

Variance measures

> How spread out the values are.



Formula:

Variance

=

Average of

(Value - Mean)²

Square each value:

(-3)² = 9

(-1)² = 1

1² = 1

3² = 9

Average:

(9 + 1 + 1 + 9)

/

4

=

20 / 4

=

5

So

Variance = 5


---

Step 4️⃣ — Compute Standard Deviation

Standard Deviation is simply:

√Variance

Here:

√5

≈

2.236


---

Step 5️⃣ — Normalize

Formula:

Normalized Value

=

(Value - Mean)

/

Standard Deviation

Example:

First feature:

(2 - 5)

/

2.236

≈

-1.34

Second feature:

(4 - 5)

/

2.236

≈

-0.45

Continue for all features.

Final normalized vector:

[-1.34, -0.45, 0.45, 1.34]

Now:

Mean ≈ 0

Standard deviation ≈ 1



---

🤔 Why Add a Tiny Number (ε)?

What if all features are the same?

Example:

[5, 5, 5, 5]

Mean:

5

Variance:

0

Standard deviation:

0

Now we'd divide by zero!

To avoid this, LayerNorm uses:

(Value - Mean)

/

√(Variance + ε)

where ε (epsilon) is a very small number like 1e-5.

This prevents division by zero and improves numerical stability.


---

🎯 Why Do We Need γ (Gamma) and β (Beta)?

After normalization,

the data always has:

Mean = 0

Standard Deviation = 1

But maybe the model doesn't always want that exact distribution.

So LayerNorm learns two parameters:

Output

=

Gamma × Normalized

+

Beta

Where:

Gamma (γ) controls the scale.

Beta (β) controls the shift.


These are trainable parameters updated during learning.


---

🎨 Visualization

Without Gamma & Beta

[-1.2

-0.4

0.3

1.3]

After Gamma

[-2.4

-0.8

0.6

2.6]

After Beta

[-1.4

0.2

1.6

3.6]

The model learns the best scale and offset automatically.


---

📦 Tensor Shapes

Suppose:

Batch = 2

Tokens = 5

Embedding = 768

Input:

(2, 5, 768)

LayerNorm computes statistics over the last dimension (768 features) for each token independently.

Output:

(2, 5, 768)

The shape does not change.


---

🐍 PyTorch Implementation

Using PyTorch is simple:

import torch.nn as nn

layer_norm = nn.LayerNorm(768)

output = layer_norm(x)

Internally, PyTorch performs:

Compute Mean

↓

Compute Variance

↓

Normalize

↓

Apply Gamma

↓

Apply Beta


---

🖼️ How ViT Uses LayerNorm

Each image patch has an embedding.

Example:

Patch 1

↓

768 Features

LayerNorm normalizes those 768 features before or after attention (depending on the architecture).

Every patch is normalized independently.


---

🚀 How SAM Uses LayerNorm

SAM's Vision Transformer uses Pre-LayerNorm.

Remember Lesson 17?

We briefly mentioned:

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

This is exactly the pattern used in many modern Transformer models, including SAM.


---

🛠️ Debug Like an ML Engineer

❌ Common Mistake

Many beginners think:

> LayerNorm uses information from the entire batch.



Wrong.

For an input:

(8, 196, 768)

LayerNorm computes statistics over:

768 Features

for each token independently.

It does not average across the 8 images in the batch.


---

🧠 Interactive Challenge

Suppose a token embedding is:

[10, 20, 30]

Question:

What is the mean?

Pause...

👇

Answer:

(10 + 20 + 30)

/

3

=

20

Next:

Subtract the mean.

[-10, 0, 10]

You're already doing LayerNorm by hand!


---

🧩 Quick Quiz

Question 1

LayerNorm computes statistics across:

A. The entire dataset

B. The batch dimension

C. The feature dimension of each token


<details>
<summary>✅ Answer</summary>C

</details>
---

Question 2

Why do we add ε (epsilon)?

<details>
<summary>✅ Answer</summary>To avoid division by zero and improve numerical stability.

</details>
---

Question 3

What do Gamma and Beta do?

<details>
<summary>✅ Answer</summary>Gamma scales the normalized output, and Beta shifts it. Both are trainable.

</details>
---

📌 Key Takeaways

> ✅ LayerNorm normalizes each token across its feature dimension.



> ✅ It computes Mean → Variance → Standard Deviation → Normalization.



> ✅ ε prevents division by zero.



> ✅ γ (Gamma) and β (Beta) are learned parameters.



> ✅ ViT and SAM use LayerNorm extensively, typically in a Pre-LayerNorm architecture.




---

🗺️ Complete Transformer Block (Almost Complete!)

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

🚀 Next Lesson — Residual Connections: Why Deep Transformers Don't Collapse

We introduced residual connections earlier, but now we'll study them mathematically.

You'll learn:

➕ Why we add the input back to the output.

🧠 How residuals improve gradient flow.

📉 What the vanishing gradient problem is.

🐍 How residual connections are implemented in PyTorch.

🖼️ Why every Transformer block in ViT and SAM uses residual connections.


> Milestone: After the next lesson, you'll fully understand every mathematical component of a Transformer Encoder block and will be ready to begin reading actual Transformer and SAM source code line by line.
