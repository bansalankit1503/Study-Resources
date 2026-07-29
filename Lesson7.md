Lesson 7 — Why Were Transformers Invented?

> Goal: Before learning Attention, you must understand why researchers felt the need to invent Transformers. Every major AI breakthrough starts by solving a limitation of the previous generation.




---

The Evolution of Deep Learning

Let's look at the journey.

Traditional Machine Learning
        │
        ▼
Neural Networks
        │
        ▼
CNNs (Images)
        │
        ▼
RNNs (Sequences)
        │
        ▼
LSTMs / GRUs
        │
        ▼
Attention
        │
        ▼
Transformers
        │
        ▼
ViT, GPT, BERT, SAM, CLIP

Notice something.

Transformers didn't appear suddenly.

They were invented because CNNs and RNNs had important limitations.


---

Part 1 — CNNs Are Great at Images

Suppose we have this image.

🐶 Dog

A CNN looks at the image using small filters.

Example:

Image

□□□□□□□□

□□□□□□□□

□□□□□□□□

□□□□□□□□

A small filter (3×3)

■■■
■■■
■■■

moves across the image.

Step 1

■■■□□□□□

■■■□□□□□

■■■□□□□□


Step 2

□■■■□□□□

□■■■□□□□

□■■■□□□□

This process is called Convolution.


---

What Does CNN Learn?

Early layers learn

Edges

Corners

Lines

Middle layers learn

Eyes

Nose

Wheels

Deep layers learn

Dog

Car

Human

Very impressive.

CNNs dominated computer vision for years.


---

But CNN Has a Problem

Imagine this image.

Cat                           Ball

The cat is on the far left.

The ball is on the far right.

Can the first convolution see both?

No.

A 3×3 filter only sees a tiny region.

■■■□□□□□□□□□□□□□□
■■■□□□□□□□□□□□□□□
■■■□□□□□□□□□□□□□□

It has local vision.

It cannot immediately understand distant relationships.


---

Receptive Field

Each CNN layer sees only a small region.

Example

Layer 1

3 × 3 pixels

Layer 2

5 × 5 pixels

Layer 3

7 × 7 pixels

Eventually,

after many layers,

the network sees the whole image.

But this takes time.

Information must travel layer by layer.


---

Why Is This a Problem?

Suppose we're segmenting an object in SAM.

Imagine a person's hand on one side of the image and the body on the other.

To understand they belong to the same person,

the model needs global context.

CNNs obtain this only after many layers.

Transformers obtain it immediately.

We'll see how later.


---

Part 2 — RNNs

CNNs are excellent for images.

But what about text?

Sentence:

I love machine learning.

Words arrive one after another.

RNNs process them sequentially.

"I"

↓

"love"

↓

"machine"

↓

"learning"

Each word updates the hidden state.


---

RNN Memory

Think of reading a book.

You read

Page 1

↓

Page 2

↓

Page 3

You remember previous pages while reading the next.

That's exactly how an RNN works.

It has a hidden memory.


---

Example

Sentence

The sky is blue.

Processing

"The"

↓

Hidden State

↓

"sky"

↓

Hidden State

↓

"is"

↓

Hidden State

↓

"blue"

Every word depends on the previous hidden state.


---

Sounds Perfect...

But there's a serious problem.

Imagine reading a very long book.

Page 1

↓

Page 50

↓

Page 100

↓

Page 500

Can you remember every detail from Page 1?

Probably not.

RNNs have the same issue.


---

The Forgetting Problem

Consider this sentence.

The boy who won the national science competition after preparing for five years received a gold medal.

To understand "received",

the model must remember "The boy" from the beginning.

For long sentences,

RNNs gradually forget earlier information.

This is called the long-term dependency problem.


---

Why Does This Happen?

Every step depends on the previous one.

Word 1

↓

Word 2

↓

Word 3

↓

Word 4

↓

...

↓

Word 100

Information has to travel through every intermediate word.

As the sequence grows,

important information weakens.


---

LSTM Was Invented

Researchers introduced Long Short-Term Memory (LSTM) networks.

Instead of one memory,

they added special gates.

Input

↓

Forget Gate

↓

Memory

↓

Output Gate

The gates decide

What to remember.

What to forget.

What to update.


LSTMs greatly improved sequence modeling.


---

Did LSTM Solve Everything?

No.

Imagine reading a 2,000-word document.

Even LSTM processes words one by one.

Word 1

↓

Word 2

↓

Word 3

↓

...

↓

Word 2000

This creates two problems.

Problem 1 — Slow

The model cannot process Word 100 until Word 99 is finished.

Everything is sequential.

Modern GPUs prefer doing many operations at the same time.


---

Problem 2 — Long Dependencies

Although LSTMs remember better,

very distant information is still difficult.


---

Example

Sentence

The animal didn't cross the road because it was tired.

Question

Who is "it"?

Answer

The animal

The model must connect words that are far apart.

RNNs and LSTMs struggle as the distance increases.


---

Researchers Asked a New Question

Instead of forcing information to travel through every word...

Why not allow every word to directly look at every other word?

Imagine a classroom.

Old method (RNN)

Student A

↓

Student B

↓

Student C

↓

Student D

If Student A wants to tell Student D something,

the message passes through B and C.

Slow.

Now imagine

A ───── D

A ───── C

B ───── D

C ───── A

Every student can directly talk to every other student.

Much faster.

This simple idea became Attention.


---

Why This Changed AI Forever

Attention allows every token to ask:

> Which other tokens are important for me?



Instead of remembering everything,

the model focuses only on relevant information.

That's why it's called Attention.


---

How Does This Apply to Images?

Vision Transformers divide an image into patches.

Example

Image

┌──┬──┬──┬──┐
│P1│P2│P3│P4│
├──┼──┼──┼──┤
│P5│P6│P7│P8│
├──┼──┼──┼──┤
│P9│P10│P11│P12│
└──┴──┴──┴──┘

Unlike CNNs,

Patch 1 can immediately interact with Patch 12.

No need to wait through dozens of convolution layers.

This gives the model global understanding from the beginning.


---

How SAM Uses This

SAM's Image Encoder is a Vision Transformer (ViT).

Instead of processing pixels one local region at a time,

it processes image patches using Attention.

Image

↓

Split into Patches

↓

Patch Embedding

↓

Transformer Blocks

↓

Image Features

↓

Mask Decoder

↓

Segmentation Mask

This global interaction is one reason SAM performs so well across many kinds of images.


---

Comparison

Model	Strength	Weakness

CNN	Excellent at local features	Weak global context
RNN	Handles sequences	Forgets long information
LSTM	Better memory	Still sequential and slower
Transformer	Global context + parallel processing	More computationally expensive



---

Common Mistakes

Mistake 1

Thinking Transformers replaced CNNs because CNNs were "bad."

❌ Wrong.

CNNs are still widely used, especially in efficient mobile models and hybrid architectures.


---

Mistake 2

Thinking LSTMs are obsolete.

❌ Not entirely.

They are still useful for some small datasets, embedded systems, and time-series tasks.


---

Mistake 3

Thinking Attention only works for text.

❌ Wrong.

Attention powers:

GPT (text)

BERT (text)

ViT (images)

SAM (segmentation)

CLIP (image + text)

Whisper (speech)



---

Interview Questions

Q1. Why can't CNNs easily model long-range relationships?

Answer: Convolution operates on small local regions, so information from distant parts of an image must pass through many layers before interacting.


---

Q2. Why are RNNs difficult to parallelize?

Answer: Each time step depends on the previous hidden state, so computations must happen sequentially.


---

Q3. What was the biggest idea behind Attention?

Answer: Instead of passing information step by step, allow each token (or image patch) to directly interact with every other relevant token.


---

Mini Assignment

1. Why do CNNs naturally learn local features first?


2. What is the long-term dependency problem in RNNs?


3. Why are LSTMs faster than RNNs in learning long dependencies but still slower than Transformers?


4. Why is parallel processing important for modern GPUs?


5. In SAM, what replaces the traditional CNN image encoder?




---

Next Lesson (Lesson 8)

Now we arrive at the concept that transformed AI:

> Attention



We'll build Attention completely from scratch.

You'll learn:

Why Attention was invented.

What are Query (Q), Key (K), and Value (V)?

Why these strange names were chosen.

How Attention decides what to focus on.

The intuition before any mathematics.

How this becomes the core operation inside ViT, SAM, BERT, and GPT.


This is the lesson where the Transformer truly begins.
