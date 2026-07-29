🧠 Transformer Course

📚 Module 3 — Attention Mechanism

Lesson 11: Softmax — Turning Scores into Decisions

> 🎯 Goal

By the end of this lesson, you'll understand:

Why Query × Key scores cannot be used directly.

What Softmax actually does.

Why every row of the Attention Matrix must sum to 1.

How Attention Scores become Attention Weights.

Why Softmax is one of the most important operations in every Transformer.





---

📍 Where We Are

Lesson 8 ✅  What are Q, K, V?

        ↓

Lesson 9 ✅  How Q, K, V are created

        ↓

Lesson 10 ✅ Query × Key → Attention Scores

        ↓

⭐ Lesson 11 → Softmax


---

🤔 Think Before Reading

Imagine you're choosing one restaurant.

Google Maps shows:

Restaurant A : 98

Restaurant B : 45

Restaurant C : 210

Restaurant D : 75

Question:

👉 Can these numbers directly tell us the probability that you'll choose each restaurant?

No.

They're just scores.

We need something like

Restaurant A : 20%

Restaurant B : 5%

Restaurant C : 60%

Restaurant D : 15%

Notice two things:

Every value is between 0 and 1

Total = 100%


This is exactly what Softmax does.


---

🎯 Attention Scores from Last Lesson

Suppose we're processing the word

"Love"

The Attention Scores are

Word	Score

I	2
Love	5
AI	8


Question:

Does this mean

AI is 8% important?


❌ No.

Does it mean

Love is 5% important?


❌ No.

These are relative scores, not probabilities.


---

⚠️ Why Raw Scores Are a Problem

Imagine these scores.

2

5

8

Now imagine another sentence.

120

350

900

Both sentences may represent the same relative importance, but the numbers are on completely different scales.

We need a consistent representation.


---

💡 Softmax Solves This

Softmax converts any list of numbers into probabilities.

Example

Before Softmax

2

5

8

After Softmax

0.002

0.047

0.951

Now

0.002
+
0.047
+
0.951
=
1.000

Perfect.

Now we have probabilities.


---

🎨 Visual Flow

Attention Scores

2

5

8

      │

      ▼

   Softmax

      │

      ▼

0.002

0.047

0.951


---

🤔 Why Does the Largest Score Become Much Bigger?

Think about a classroom.

Marks:

Student	Marks

A	20
B	50
C	95


Who should receive the scholarship?

Obviously

Student C

Softmax intentionally gives much more importance to the highest score.


---

📦 Tensor Shapes

Suppose

Batch Size = 2

Sequence Length = 5

From Lesson 10

Attention Scores

Shape

(2, 5, 5)

After Softmax

Shape

(2, 5, 5)

🟢 Important

Softmax does not change the shape.

It only changes the values.


---

🎨 Example Matrix

Before Softmax

	K1	K2	K3

Q1	2	5	8
Q2	1	7	3
Q3	6	2	4



---

After Softmax

	K1	K2	K3

Q1	0.002	0.047	0.951
Q2	0.002	0.980	0.018
Q3	0.866	0.016	0.118



---

> 💡 Observation

Every row now sums to 1.

Each row represents how one Query distributes its attention across all Keys.




---

🧠 Interactive Check

Look at this row.

0.10

0.20

0.70

Pause for 5 seconds.

Ask yourself:

> Which word is receiving the most attention?



✅ Answer:

The third word.

Because it has the highest probability.


---

🚀 Now What Happens?

We have probabilities.

But we still haven't used the Values (V).

Remember

Query

↓

Compare with Keys

↓

Softmax

↓

???

What happens next?

The probabilities are used as weights.

Imagine

Attention

0.10

0.20

0.70

Values

V1

V2

V3

The final output becomes

(0.10 × V1)

+

(0.20 × V2)

+

(0.70 × V3)

Notice something.

The most important Value contributes the most.


---

🎨 Complete Attention Pipeline So Far

Input Embedding

      │

      ▼

Create Q, K, V

      │

      ▼

Q × K

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

Multiply with Values (V)

      │

      ▼

Final Attention Output


---

💻 PyTorch Example

import torch
import torch.nn.functional as F

scores = torch.tensor([[2.0, 5.0, 8.0]])

weights = F.softmax(scores, dim=-1)

print(weights)

Output (approximately)

tensor([[0.0024, 0.0473, 0.9503]])

Notice

print(weights.sum())

Output

1.0


---

🚀 How SAM Uses This

SAM divides an image into thousands of patches.

Imagine

Patch 101

asks

> "Which other patches help me understand this object?"



Attention Scores

Patch 15 → 2

Patch 80 → 8

Patch 230 → 12

Softmax converts these into

Patch 15 → 0.01

Patch 80 → 0.20

Patch 230 → 0.79

Now the representation of Patch 101 is influenced mostly by Patch 230.

This is how SAM connects distant image regions, such as a person's face and hands, without needing many convolution layers.


---

⚠️ Common Mistakes

❌ Mistake 1

Thinking Softmax chooses only one word.

Wrong.

It gives every word a probability.

Some are just much larger than others.


---

❌ Mistake 2

Thinking Softmax changes tensor shape.

Wrong.

Only the values change.


---

❌ Mistake 3

Thinking Softmax is unique to Transformers.

Wrong.

It's also widely used in classification models to convert logits into probabilities.


---

🧩 Quick Quiz

Question 1

Scores

3

9

1

Which word gets the highest attention after Softmax?

<details>
<summary>✅ Answer</summary>The second word, because it has the highest score.

</details>
---

Question 2

After Softmax, each row should sum to:

A. 0

B. 1

C. Number of words


<details>
<summary>✅ Answer</summary>B. 1

</details>
---

Question 3

Does Softmax change tensor shape?

<details>
<summary>✅ Answer</summary>No. It only changes the values.

</details>
---

📌 Key Takeaways

> ✅ Query × Key produces Attention Scores.



> ✅ These scores are not probabilities.



> ✅ Softmax converts scores into Attention Weights.



> ✅ Every row after Softmax sums to 1.



> ✅ These weights determine how much each Value (V) contributes to the final representation.




---

🏗️ Transformer Pipeline Progress

Embeddings               ✅

↓

Q, K, V                 ✅

↓

Q × K                   ✅

↓

Softmax                 ✅

↓

Attention Weights       ✅

↓

❓ Multiply by Values    ← Next Lesson


---

🚀 Next Lesson — The Final Attention Equation

In the next lesson, we'll complete the Attention mechanism.

You'll learn:

How Attention Weights combine with Values (V).

Why this creates a richer representation of each word or image patch.

The complete tensor shape flow from input to output.

The full Attention block as implemented in PyTorch.

How this exact process appears inside SAM's Vision Transformer.


> 💡 One suggestion for the course: from Lesson 12 onward, I can also include a small "Debug Like an ML Engineer" section in every lesson. It will show common tensor-shape bugs, PyTorch mistakes, and real debugging techniques that you'll encounter when working with models like SAM, ViT, and LoRA. This is something most tutorials don't teach, but it's extremely valuable in practice.
