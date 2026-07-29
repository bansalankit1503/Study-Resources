Lesson 8 — Attention: The Biggest Idea Behind Transformers

> Goal: Understand Attention from first principles. By the end of this lesson, you'll know why Query (Q), Key (K), and Value (V) exist, even before seeing a single matrix multiplication.




---

Recap

In the last lesson, we learned:

CNNs only see local regions.

RNNs process one word at a time.

LSTMs improve memory but are still sequential.


Researchers asked:

> Can every word directly access every other word?



The answer became Attention.


---

A Real-Life Example

Imagine you are in a library with 10,000 books.

You ask the librarian:

> "I want a book about Transformers in Machine Learning."



The librarian doesn't give you every book.

Instead, they search for the most relevant ones.

This process has three parts.

Your Question
      │
      ▼
Search Shelves
      │
      ▼
Relevant Book

This is exactly how Attention works.


---

Introducing Q, K, and V

Every item has three pieces of information.

Query (Q)

Key (K)

Value (V)

These names sound confusing, but they're actually very intuitive.


---

Think of a Library

Suppose one book looks like this:

Book Title:
Machine Learning with Transformers

Library Tag:
Machine Learning

Book Content:
500 pages explaining Transformers

Now compare these to Attention.

Library Search = Query (Q)

Library Tag = Key (K)

Book Content = Value (V)

Meaning:

Query asks: "What am I looking for?"

Key answers: "What topic am I about?"

Value contains the actual information.



---

Another Example: YouTube Search

You search:

SAM LoRA Tutorial

This is the Query.

Each video has

Title

Description

Tags

These are like Keys.

The actual video is the Value.

Attention works the same way.

It compares your Query with every Key.

The best matches return their Values.


---

Sentence Example

Sentence:

The cat sat on the mat.

Suppose we're processing the word

cat

Question:

Which other words are important?

Maybe

The

cat

sat

mat

The word cat should pay more attention to

The

sat

mat


and less attention to unrelated words (if any existed).


---

Attention Is Like a Conversation

Imagine four people.

Alice

Bob

Charlie

David

Alice asks:

> "Who has information useful to me?"



Bob replies:

> "I know a little."



Charlie replies:

> "I know a lot."



David replies:

> "Not relevant."



Alice listens mostly to Charlie.

That is Attention.


---

Breaking It Down

Suppose we process one word.

Step 1

Generate a Query.

Word

↓

Query

Step 2

Every word already has a Key.

Word

↓

Key

Step 3

Compare Query with every Key.

Query

↓

Compare

↓

Key 1

Key 2

Key 3

Key 4

Step 4

Important words receive higher scores.

Step 5

Collect their Values.

That's the final Attention output.


---

Example

Sentence:

The dog chased the ball.

Suppose we're processing

chased

Possible attention scores:

Word	Importance

The	0.05
Dog	0.45
Chased	0.10
The	0.05
Ball	0.35


Notice:

The model focuses mostly on

Dog

Ball


because they are closely related to chased.


---

What Are Values?

Suppose

Dog

contains information like

Animal

Subject

Living Thing

The Value stores this useful representation.

After deciding that Dog is important,

Attention copies more information from its Value.


---

Complete Attention Flow

Current Word
      │
      ▼
Create Query
      │
      ▼
Compare with Every Key
      │
      ▼
Find Important Words
      │
      ▼
Collect Their Values
      │
      ▼
New Representation

Notice:

The word itself changes.

It becomes smarter because it has gathered information from related words.


---

Why Not Use Only the Words?

Good question.

Words themselves are not enough.

The word

Apple

can mean

Fruit

Company


Depending on the sentence.

Attention lets the meaning change depending on context.

Example:

Apple released a new iPhone.

Clearly,

Apple = Company.

Now

I ate an apple.

Now,

Apple = Fruit.

The same word gets a different representation because it attends to different surrounding words.


---

Images Work the Same Way

Suppose we split an image into patches.

┌──┬──┬──┐
│P1│P2│P3│
├──┼──┼──┤
│P4│P5│P6│
├──┼──┼──┤
│P7│P8│P9│
└──┴──┴──┘

Patch P5 asks:

> Which other patches are useful?



Maybe

P2

P4

P6

P8

contain parts of the same object.

Attention connects them immediately.

Unlike CNNs,

P5 doesn't need dozens of layers to communicate with P2.


---

How SAM Uses Attention

Suppose an image contains a dog.

Head                     Tail

These regions are far apart.

CNN:

Head

↓

Layer

↓

Layer

↓

Layer

↓

Tail

Information travels slowly.

Transformer:

Head ───────── Tail

Direct connection.

That's why Vision Transformers capture global context so effectively.


---

Where Do Q, K, and V Come From?

This is the next logical question.

The input embedding is transformed into three different vectors.

Input Embedding
        │
        ├──► Query
        │
        ├──► Key
        │
        └──► Value

These are created using three different linear layers.

We'll learn exactly how in the next lesson.


---

Common Mistakes

Mistake 1

Thinking Query, Key, and Value are different words.

❌ Wrong.

They are three different representations of the same input.


---

Mistake 2

Thinking Attention copies information from only one word.

❌ Wrong.

It combines information from all words, giving more weight to the important ones.


---

Mistake 3

Thinking Attention is only for text.

❌ Wrong.

Attention is used for:

Text (GPT, BERT)

Images (ViT, SAM)

Speech (Whisper)

Multimodal models (CLIP, Florence)



---

How This Connects to SAM

Inside every Vision Transformer block in SAM:

Image

↓

Patch Embedding

↓

Query

Key

Value

↓

Attention

↓

Better Image Features

↓

Next Transformer Block

Every Transformer block repeats this process, gradually building richer image representations before the mask decoder predicts the final segmentation.


---

Interview Questions

Q1. Why do we need Query, Key, and Value?

Answer: They separate the roles of searching (Query), matching (Key), and retrieving information (Value), allowing each token or patch to selectively gather relevant context.


---

Q2. What does Attention actually compute?

Answer: It determines how much each token (or image patch) should use information from every other token (or patch).


---

Q3. Why is Attention better than sequential processing?

Answer: Because every token can directly interact with every other token in parallel, making it better at capturing long-range relationships.


---

Mini Assignment

1. Explain Query, Key, and Value using the library analogy.


2. Why can the word "Apple" have different meanings in different sentences?


3. Why does Attention improve long-range understanding?


4. In a Vision Transformer, what corresponds to "words"?


5. Why are three different representations (Q, K, V) created from the same input?




---

Next Lesson (Lesson 9)

Now we move from intuition to implementation.

You'll learn:

How Query, Key, and Value are actually created.

Why we use three different weight matrices.

Tensor shapes at every step.

The first matrix multiplication inside a Transformer.

How this is implemented in PyTorch.

Where these layers appear in the SAM Vision Transformer source code.


This is where we'll begin reading Transformer code like an ML engineer.
