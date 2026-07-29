Lesson 6 — Backpropagation: How Does the Error Reach Every Weight?

> Goal: Understand exactly what loss.backward() does internally and why it is the heart of deep learning.




---

Why Are We Learning This?

Suppose you're training SAM with LoRA.

Your training loop looks like this:

prediction = model(image)

loss = criterion(prediction, target)

loss.backward()

optimizer.step()

Most people know this code.

Very few know what actually happens inside loss.backward().

After this lesson, you'll know exactly what happens.


---

Recap

So far we know this pipeline:

Image
   │
   ▼
Model
   │
   ▼
Prediction
   │
   ▼
Loss

Now ask yourself:

> The model made a mistake.



How does every weight know how much it should change?

That's exactly what Backpropagation solves.


---

Imagine a Factory

Suppose a factory builds a car.

Steel
  │
  ▼
Car Body
  │
  ▼
Paint
  │
  ▼
Engine
  │
  ▼
Finished Car

Customer says:

> "The car is bad."



Can we immediately blame the engine?

No.

We inspect every stage.

Maybe:

Steel quality was poor.

Body shape was wrong.

Paint was defective.

Engine wasn't installed correctly.


The error travels backward through every stage.

This is Backpropagation.


---

Forward Pass

Let's build the smallest neural network.

We have

Input (x)

Weight (w)

Bias (b)

Suppose

Input = 4

Weight = 2

Bias = 1

The neuron calculates

Output = (Input × Weight) + Bias

Output = (4 × 2) + 1

Output = 9

This is called the Forward Pass.

Information moves from input to output.

Input
   │
   ▼
Multiply by Weight
   │
   ▼
Add Bias
   │
   ▼
Prediction


---

Step 2 — Calculate Loss

Suppose

Prediction = 9

Actual Answer = 12

Difference

Loss = 3

The model is wrong.

Now comes the important question.


---

Which Weight Caused This Error?

Suppose we change the weight from

2

↓

2.1

Will the loss

Increase?

Decrease?


We need a way to measure this.

This measurement is called the Gradient.


---

What is a Gradient?

Gradient simply means

> How much does the loss change if I slightly change a weight?



Think of it like a car steering wheel.

Turn a little.

Observe what happens.

If the car moves in the correct direction,

keep turning.

Otherwise,

turn the opposite way.

That's exactly what gradients tell the optimizer.


---

Computational Graph

Instead of solving everything at once,

deep learning breaks the computation into tiny operations.

Our neuron becomes

Input
   │
   ▼
Multiply
   │
   ▼
Add
   │
   ▼
Prediction
   │
   ▼
Loss

Every box performs one simple operation.


---

Forward Pass Example

Suppose

Input = 5

Weight = 3

Bias = 1

Step 1

Multiply

5 × 3 = 15

Step 2

Add Bias

15 + 1 = 16

Prediction

16

Suppose

Target = 20

Loss

4


---

Backward Pass

Now we walk backwards.

Instead of

Input

↓

Multiply

↓

Add

↓

Loss

we go

Loss

↑

Add

↑

Multiply

↑

Input

Notice something.

The error moves backward.

That's why it is called

Backpropagation


---

Why Not Compute Everything Directly?

Imagine a Transformer with

1 Billion Parameters

Would you manually calculate gradients?

Impossible.

Instead,

PyTorch remembers every operation performed during the forward pass.

Later,

loss.backward()

walks backward through those operations,

computing gradients automatically.

This system is called

Autograd.


---

What Does PyTorch Remember?

Suppose you write

x = layer1(image)
x = gelu(x)
x = layer2(x)

loss = criterion(x, target)

PyTorch secretly creates

Image
   │
   ▼
Layer 1
   │
   ▼
GELU
   │
   ▼
Layer 2
   │
   ▼
Loss

Every operation is stored.

Nothing is forgotten.


---

Then loss.backward() Starts

It walks backwards.

Loss

↑

Layer 2

↑

GELU

↑

Layer 1

↑

Image

At every layer,

it computes gradients.


---

What is Stored?

Each trainable parameter has

parameter.grad

Example

print(model.weight.grad)

might print

tensor([[0.42, -0.18]])

This means

Changing this weight changes the loss by approximately

0.42

or

-0.18

depending on the weight.


---

What Happens Next?

Notice

loss.backward()

does NOT

change the weights.

It only computes gradients.

Weights still have the same values.


---

Weight Update

The optimizer now updates the weights.

optimizer.step()

Think of it like

Current Weight

↓

Current Weight - (Learning Rate × Gradient)

↓

New Weight

Every trainable parameter is updated like this.


---

Complete Training Loop

Now you understand the complete picture.

Image
   │
   ▼
Forward Pass
   │
   ▼
Prediction
   │
   ▼
Loss
   │
   ▼
Backward Pass
   │
   ▼
Gradients
   │
   ▼
Optimizer
   │
   ▼
Updated Weights

This process repeats thousands of times.

Eventually,

the model becomes better.


---

How Does This Work in SAM?

Suppose you fine-tune SAM.

Input Image
      │
      ▼
Image Encoder (ViT)
      │
      ▼
Prompt Encoder
      │
      ▼
Mask Decoder
      │
      ▼
Predicted Mask
      │
      ▼
Loss

Now Backpropagation begins.

Loss

↑

Mask Decoder

↑

Prompt Encoder

↑

Image Encoder

Gradients flow backward.


---

What Changes During LoRA?

Normal Training

Loss

↑

Update Every Layer

LoRA Training

Loss

↑

Gradient flows through all layers

↓

Only LoRA layers are trainable

↓

Only LoRA weights are updated

For example

ViT Block

Query Linear
      │
      ├── Original Weight (Frozen)
      │
      └── LoRA Weight (Trainable)

During training

Gradient

↓

Original Weight ❌ Not Updated

↓

LoRA Weight ✅ Updated

This is why LoRA requires much less GPU memory.


---

PyTorch Example

import torch
import torch.nn as nn

model = nn.Linear(2, 1)

x = torch.tensor([[2.0, 3.0]])
target = torch.tensor([[8.0]])

prediction = model(x)

loss = nn.MSELoss()(prediction, target)

loss.backward()

print("Weight Gradient")
print(model.weight.grad)

print("Bias Gradient")
print(model.bias.grad)

Then update

optimizer.step()


---

Common Mistakes

Mistake 1

Thinking

loss.backward()

updates weights.

❌ Wrong

It only computes gradients.


---

Mistake 2

Forgetting

optimizer.zero_grad()

PyTorch accumulates gradients.

Always do

optimizer.zero_grad()

loss.backward()

optimizer.step()


---

Mistake 3

Confusing Gradient with Loss

Loss

How wrong is the prediction?

Gradient

How should each weight change?

Very different concepts.


---

Interview Questions

Q1. What does loss.backward() do?

Answer: Computes gradients for every trainable parameter using backpropagation and stores them in parameter.grad.


---

Q2. Does loss.backward() update weights?

Answer: No.

Only computes gradients.

Weight updates happen in

optimizer.step()


---

Q3. Why is Backpropagation efficient?

Because it reuses intermediate computations while moving backward through the computational graph, instead of recomputing everything from scratch for each parameter.


---

Mini Assignment

Answer these before the next lesson.

Question 1

What is the difference between

loss.backward()

and

optimizer.step()


---

Question 2

Why does PyTorch create a computational graph?


---

Question 3

During LoRA training,

why do frozen weights receive no updates?


---

Question 4

What is stored inside

parameter.grad


---

Question 5

Explain Backpropagation to someone who has never studied Machine Learning.


---

Next Lesson (Lesson 7)

Now we begin the journey toward Transformers.

We'll answer one of the biggest questions in AI history:

> If neural networks were already working, why did researchers invent Transformers?



We'll study:

Why CNNs struggle with global context.

Why RNNs forget long sequences.

Why LSTMs were only a partial solution.

The historical problems that led to the invention of Attention.

How these limitations motivated the architecture used today in ViT, SAM, GPT, and nearly every modern foundation model.


From Lesson 7 onward, we're entering the world of Transformer architecture—the part that connects directly to your work with SAM + LoRA.
