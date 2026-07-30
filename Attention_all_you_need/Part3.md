🧠 Transformer Course

📚 Module 8 — Reading the Original Research Paper

🚀 Lesson 24 — Why Multi-Head Attention Works (And Why One Head Is Not Enough)

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ Why Multi-Head Attention is better than Single-Head Attention.

✅ What different attention heads learn.

✅ Why increasing embedding size cannot replace multiple heads.

✅ How Multi-Head Attention helps Vision Transformers and SAM.




---

📍 Course Progress

Scaled Dot-Product Attention   ✅

↓

⭐ Multi-Head Attention (Deep Dive)

↓

Encoder Layer Details

↓

Vision Transformer (ViT)

↓

SAM Source Code


---

🧠 Think Before Reading

Imagine you're watching a football match.

Would one camera be enough?

Probably not.

Professional broadcasts use many cameras:

📷 Camera 1 → Wide view

📷 Camera 2 → Goalkeeper

📷 Camera 3 → Striker

📷 Camera 4 → Crowd

📷 Camera 5 → Coach

Each camera focuses on something different.

When combined, you get a much better understanding of the match.

👉 Multi-Head Attention works exactly like this.


---

🤔 What Happens with One Attention Head?

Suppose we have the sentence:

The little brown dog chased the ball.

With one attention head, the model creates a single attention map.

dog

↓

ball

↓

chased

It has to learn all relationships using one attention pattern.

That is a difficult job.


---

🎯 The Limitation

One attention head may focus strongly on one relationship.

For example:

dog

↓

ball

But then it may pay less attention to:

little

↓

dog

or

brown

↓

dog

One head has limited capacity to represent many different types of relationships at the same time.


---

💡 Multi-Head Attention

Instead of one attention head,

we create many independent heads.

Input

↓

Head 1

Head 2

Head 3

Head 4

↓

Combine Results

↓

Output

Each head has its own:

Query matrix (Wq)

Key matrix (Wk)

Value matrix (Wv)


So each head learns something different.


---

🧠 What Can Different Heads Learn?

Let's use the sentence:

The little brown dog chased the ball.


---

🎯 Head 1 — Subject

chased

↓

dog

This head learns:

> Who performed the action?




---

🎯 Head 2 — Object

chased

↓

ball

This head learns:

> What was affected?




---

🎯 Head 3 — Description

dog

↓

little

↓

brown

This head learns descriptive words.


---

🎯 Head 4 — Long-Distance Relationship

The

↓

ball

This head might learn broader sentence structure.

> ⚠️ Important:

These are illustrative examples. During training, no one explicitly assigns jobs to the heads. The model learns useful attention patterns automatically.




---

🎨 Visualizing Multiple Heads

Imagine four transparent attention maps.

Head 1

dog → chased

Head 2

ball → chased

Head 3

brown → dog

Head 4

little → dog

After computation,

all these perspectives are combined.


---

❓ Why Not Just Use One Bigger Head?

A common beginner question:

> "Instead of 12 heads × 64 dimensions, why not use 1 head × 768 dimensions?"



Excellent question.

Let's compare.


---

Option 1

1 Head

768 Dimensions

One attention map.

Everything

↓

One Perspective


---

Option 2

12 Heads

64 Dimensions Each

Twelve different attention maps.

Perspective 1

Perspective 2

Perspective 3

...

Perspective 12

The model can specialize.


---

🎨 Camera Analogy

Imagine inspecting a car.

One engineer checks:

🚗 Engine

Another checks:

🛞 Tires

Another checks:

💡 Lights

Another checks:

🪑 Interior

Would you replace all four engineers with one engineer trying to inspect everything at once?

Probably not.

Specialization leads to better coverage.


---

📦 Tensor Shapes

Suppose:

Batch = 2

Tokens = 5

Embedding = 768

Heads = 12

Input

(2, 5, 768)

Split into heads

(2, 12, 5, 64)

Each head performs attention independently.

Output of each head

(2, 12, 5, 64)

Concatenate

(2, 5, 768)

The final output shape matches the input embedding size.


---

🖼️ How ViT Uses Multiple Heads

Suppose an image contains:

🐶

🌳

🚗

🏠

Different heads might focus on different visual patterns.


---

Head 1

Edges


---

Head 2

Textures


---

Head 3

Object Shapes


---

Head 4

Relationships Between Objects

Again, these are conceptual examples rather than fixed assignments.


---

🚀 How SAM Benefits

SAM must understand:

Object boundaries

Relationships between image patches

Global context

Fine details


Different attention heads can naturally learn different aspects of the image.

This is one reason why Vision Transformers perform so well for segmentation.


---

💻 PyTorch Concept

# Input
x = (batch, tokens, embed_dim)

# Split into heads
Q = Q.view(batch, tokens, heads, head_dim)

# Compute attention independently
head1 = attention(...)
head2 = attention(...)
...

# Concatenate
output = torch.cat([...], dim=-1)

In real implementations, these operations are vectorized rather than computed head by head in a Python loop.


---

🛠️ Debug Like an ML Engineer

❌ Common Mistake

Many beginners think:

12 Heads

↓

12 Times More Parameters

Not necessarily.

Example:

1 Head × 768

vs

12 Heads × 64

Both still operate over a total embedding size of 768. The parameter count stays in a similar range because the heads collectively partition the embedding space rather than multiplying it.


---

🧠 Interactive Challenge

Suppose

Embedding = 1024

Heads = 16

Question:

What is the size of each head?

Pause...

👇

Answer

1024 / 16

=

64

Each head processes 64-dimensional Query, Key, and Value vectors.


---

🧩 Quick Quiz

Question 1

Why is Multi-Head Attention better than Single-Head Attention?

A. It reduces memory to zero.

B. Different heads can learn different attention patterns.

C. It removes the need for Softmax.


<details>
<summary>✅ Answer</summary>B

</details>
---

Question 2

If the embedding size is 768 and there are 12 heads, what is dₖ?

<details>
<summary>✅ Answer</summary>768 / 12 = 64

</details>
---

Question 3

After concatenating all heads, what is the output shape?

<details>
<summary>✅ Answer</summary>The embedding dimension returns to its original size (for example, 768).

</details>
---

📌 Key Takeaways

> ✅ Multi-Head Attention lets the model learn multiple attention patterns in parallel.



> ✅ Each head has its own Query, Key, and Value projection matrices.



> ✅ Multiple heads provide diverse perspectives instead of a single attention map.



> ✅ Vision Transformers and SAM rely on Multi-Head Attention to capture different visual relationships.




---

🗺️ Complete Multi-Head Attention Pipeline

Input

↓

Linear Projections

↓

Q, K, V

↓

Split into Heads

↓

Head 1 Attention

Head 2 Attention

...

Head N Attention

↓

Concatenate

↓

Final Linear Projection

↓

Output


---

🚀 Next Lesson — The Final Linear Projection (Wₒ): The Missing Piece of Multi-Head Attention

So far, we've learned how attention heads are split and concatenated.

But there's one important step we haven't discussed yet.

After concatenating all the heads, why do we apply another Linear layer (Wₒ)?

In the next lesson, you'll learn:

🎯 What the output projection matrix Wₒ does.

🧩 Why concatenation alone isn't enough.

🐍 How nn.MultiheadAttention and modern Transformer implementations use this layer.

🖼️ Where this appears inside ViT and SAM source code.


> Milestone: After the next lesson, you'll understand every operation inside a Multi-Head Attention block, making you ready to read the implementation line by line in PyTorch and in the SAM codebase.
