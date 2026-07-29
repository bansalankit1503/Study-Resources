🧠 Transformer Course

📚 Module 5 — Assembling a Complete Transformer

🚀 Lesson 17 — Building a Complete Transformer Block (Step by Step)

> 🎯 Learning Objectives

By the end of this lesson, you'll understand:

✅ How all Transformer components fit together.

✅ The exact order of execution inside a Transformer block.

✅ Tensor shapes at every stage.

✅ How the code maps to the architecture diagram.

✅ How this relates directly to the Vision Transformer (ViT) inside SAM.




---

📍 Course Progress

Embeddings                  ✅

↓

Multi-Head Attention        ✅

↓

Residual Connection         ✅

↓

LayerNorm                  ✅

↓

Feed Forward Network       ✅

↓

⭐ Complete Transformer Block   ← Today

↓

Stack Multiple Blocks

↓

Complete Transformer Model


---

🧠 Think Before Reading

You've learned many individual parts:

Attention

Residual

LayerNorm

FFN


Imagine learning about:

Engine

Wheels

Steering

Brakes


Knowing each part is useful...

But today we're assembling the entire car.

That's exactly what we're doing with the Transformer.


---

🏗️ The Big Picture

A Transformer block looks like this:

Input
                   │
                   ▼
        Multi-Head Attention
                   │
                   ▼
            Residual Add
                   │
                   ▼
              LayerNorm
                   │
                   ▼
       Feed Forward Network
                   │
                   ▼
            Residual Add
                   │
                   ▼
              LayerNorm
                   │
                   ▼
                 Output

Everything you've learned now fits together.


---

📦 Step 1 — Input Embeddings

Suppose we have

Batch = 2

Tokens = 5

Embedding = 768

Input shape

(2,5,768)

Think of this as

2 Sentences

↓

Each sentence has 5 tokens

↓

Each token has 768 features


---

📦 Step 2 — Multi-Head Attention

The input enters Multi-Head Attention.

Internally it performs

Input

↓

Q, K, V

↓

Q × Kᵀ

↓

Softmax

↓

Attention × Value

↓

Concatenate Heads

↓

Linear Layer

Output shape

(2,5,768)

Notice

The shape does not change.

Only the values change.


---

📦 Step 3 — First Residual Connection

Now

x = x + attention_output

Visual

Input

──────────────┐

              │

Attention      │

              ▼

        Add Together

              ▼

Residual Output

Output shape

(2,5,768)


---

📦 Step 4 — LayerNorm

Now normalize

x = layer_norm(x)

Shape

Before

(2,5,768)

↓

After

(2,5,768)

Again,

the shape stays exactly the same.


---

📦 Step 5 — Feed Forward Network

Each token now passes through

Linear

↓

GELU

↓

Linear

Tensor flow

(2,5,768)

↓

(2,5,3072)

↓

(2,5,3072)

↓

(2,5,768)

Notice

Only the embedding dimension expands temporarily.


---

📦 Step 6 — Second Residual Connection

Now

x = x + ffn_output

Output shape

(2,5,768)


---

📦 Step 7 — Second LayerNorm

Finally

x = layer_norm(x)

Output

(2,5,768)

One Transformer block is complete.


---

🎨 Complete Data Flow

Input (2,5,768)

        │

        ▼

Multi-Head Attention

        │

        ▼

(2,5,768)

        │

        ▼

Residual Add

        │

        ▼

LayerNorm

        │

        ▼

Feed Forward Network

768 → 3072 → 768

        │

        ▼

Residual Add

        │

        ▼

LayerNorm

        │

        ▼

Output (2,5,768)


---

💻 PyTorch Implementation (Simplified)

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.attn = MultiHeadAttention()

        self.norm1 = nn.LayerNorm(768)

        self.ffn = nn.Sequential(
            nn.Linear(768,3072),
            nn.GELU(),
            nn.Linear(3072,768)
        )

        self.norm2 = nn.LayerNorm(768)

    def forward(self, x):

        x = x + self.attn(x)

        x = self.norm1(x)

        x = x + self.ffn(x)

        x = self.norm2(x)

        return x

Don't worry if every line isn't crystal clear yet. By the end of this course, you'll be able to write this block yourself.


---

🧠 Interactive Walkthrough

Suppose

Input Shape

(4,100,512)

Multi-Head Attention

↓

(4,100,512)

Residual

↓

(4,100,512)

LayerNorm

↓

(4,100,512)

FFN

↓

(4,100,2048)

↓

(4,100,512)

Residual

↓

(4,100,512)

LayerNorm

↓

(4,100,512)

Notice that the overall shape remains constant throughout the block.


---

🚀 How SAM Uses This

SAM uses a Vision Transformer (ViT).

Instead of words,

the input tokens are image patches.

Image

↓

Split into Patches

↓

Patch Embeddings

↓

Transformer Block

↓

Transformer Block

↓

Transformer Block

↓

...

↓

Image Features

Every block has exactly the components you've learned.

The only difference is that the "tokens" are image patches instead of words.


---

🧩 What Happens After One Block?

Is one Transformer block enough?

Usually no.

Modern models stack many identical blocks.

Example:

Model	Approximate Number of Transformer Blocks

ViT-Tiny	12
ViT-Base	12
ViT-Large	24
GPT-2 Large	36
GPT-3 (large variants)	Dozens to many more, depending on the model size


Each block gradually builds richer representations.


---

🎨 Stacking Blocks

Input Embeddings

        │

        ▼

Transformer Block 1

        │

        ▼

Transformer Block 2

        │

        ▼

Transformer Block 3

        │

        ▼

...

        │

        ▼

Transformer Block N

        │

        ▼

Final Features

Every block has the same structure, but each has its own learned parameters.


---

🛠️ Debug Like an ML Engineer

❌ Common Mistake

Thinking every Transformer block shares weights.

Wrong.

Block 1

↓

Block 2

↓

Block 3

Each block learns its own Q, K, V projections, FFN weights, and LayerNorm parameters.

They have the same architecture but different learned values.


---

🧩 Quick Quiz

Question 1

How many residual connections are inside one Transformer block?

A. 1

B. 2

C. 4


<details>
<summary>✅ Answer</summary>B. Two

</details>
---

Question 2

Does the tensor shape usually change after Multi-Head Attention?

<details>
<summary>✅ Answer</summary>No. The output shape is designed to match the input shape.

</details>
---

Question 3

What expands from 768 → 3072 → 768?

A. Multi-Head Attention

B. LayerNorm

C. Feed Forward Network


<details>
<summary>✅ Answer</summary>C. Feed Forward Network

</details>
---

📌 Key Takeaways

> ✅ A Transformer block combines Attention, Residual Connections, LayerNorm, and FFN.



> ✅ The tensor shape is preserved throughout the block.



> ✅ Two residual connections and two LayerNorm layers are used in each block.



> ✅ Many identical Transformer blocks are stacked to build powerful models.



> ✅ SAM's Vision Transformer is built by stacking these Transformer blocks on image patch embeddings.




---

🎯 Important Note (Modern Transformers)

The block shown in this lesson is the Post-LayerNorm version because it's easier to understand.

Most modern models—including SAM, ViT, GPT-2, GPT-3, and LLaMA—actually use a Pre-LayerNorm design, where LayerNorm is applied before the Attention and FFN sublayers.

We'll study the Pre-LayerNorm architecture later and compare both designs, so you'll be able to read real research papers and production code confidently.


---

🚀 Next Lesson — Positional Encoding

So far, we've built a Transformer block.

But there's a huge problem:

> How does a Transformer know the order of words or image patches?



If we shuffle the input tokens, the Attention mechanism itself has no built-in notion of position.

In the next lesson, you'll learn:

📍 Why Transformers need positional information.

🌊 Sinusoidal Positional Encoding from the original paper.

🎓 Learned Positional Embeddings used in many modern models.

🖼️ 2D positional embeddings used in Vision Transformers and SAM.

🔍 Why positional encoding is essential for understanding language and images.


> Preview: This is one of the most elegant ideas in the original "Attention Is All You Need" paper, and it's also where Transformers begin to differ significantly between NLP and computer vision.
