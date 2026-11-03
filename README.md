# PicoGPT
To learn deeply about LLMs I went to none other than [Karpathy](), his [youtube series]() on basic LLM architecture, and of course, his [NanoGPT repository](https://github.com/karpathy/nanogpt). This is an excellent way to learn as Karpathy is clear, goes through simple steps, and the repository provides an easy playground to mess with parameters and model structure - one can see which changes affect and how, like different numbers of attention heads, layers, context length, etc. My own writeups, mostly for my understanding, can be found [elsewhere](https://dmckinnon.github.io/LLMs-overview/) on my [site](https://dmckinnon.github.io/Tokenisation-in-LLMs/). The rest of this writeup assumes a considerable familiarity with Language Model architecture, like Attention Heads, embedding vectors, tokens, etc.

I'd also recently finished my [Machine Learning Business Card](https://dmckinnon.github.io/ML-Business-card/), and was struck by the thought: _could I make a NanoGPT model that ran on one of these cards?_

There's so many considerations involved in this:
- what does such a model even do? Is it too small to do anything?
- can you get a meaningful model that small? The voice recognition card has 4MB flash and ~200Kb RAM
- The base shakespeare-predictor model that NanoGPT creates, completing phrases from the works of Shakespeare based on character-tokens in a prompt, is <todo> mb in size, with X parameters. 
- how fast will this run? My lil chip runs at 125MHz. The NanoGPT example runs on my laptop in functionally-immediate speed, but my laptop has a discrete GPU.

So, where to start? 

## Contents
- [Model structure, size, and training](#model-structure-size-and-training)
- [Porting from tflite to C++](#porting-from-tflite-to-c)
- [Can this run at all](#can-this-run-at-all)
- [Optimisations]()
- [Conclusion]()

## Model structure, size, and training
First, can we even create a model that's really small and does something useful. I set myself a size budget of 100Kb - as in the model file (all model parameters) could not be larger than this - and twiddled with the base model. But what should we train it on? 100Kb model, predicting Shakespeare? That's boring, that's not much Shakespeare and also who cares. Some other pithy quotes? Amusing, but no. 

What about ... me? What if we trained the model to regurgitate facts about me? Going back to the business card idea, wouldn't it be great if my business card could just ... speak my resume to people and answer queries about that? 

I loved this idea, so I wrote up my CV, writ small, to use as the training set. 

### The base model
If we just run the prompts Karpathy has given us, to predict Shakespeare, then we get a model with:
- vocab size 65
- dataset length 1 million
- 6 layers in the model
- 6 attention heads
- 384 embedding dimensions
- ~ 10 million parameters
- ~40Mb in storage as float

This results in a model of size Y

### My model
The training data is the following:

```
David studied at UNSW Univeristy of New South Wales from 2012 to 2016\
What did he study?
He studied engineering and mathematics, focusing in robotics and computer vision 
David worked at Microsoft from 2016-2023
What did he work on?
At Microsoft he worked on HoloLens, using feature tracking.
At Microsoft he experimented with constellation tracking and non-inertial tracking
David work at startup Tetra in 2023
At Tetra he worked on ML graph passes, op implementations, and model zoo
David worked at Qualcomm in 2024
At Qualcomm he worked on ML graph passes and op implementations
```

dataset stats:
- length of dataset in characters: 582
- all the unique characters:  ,-.012346?ADHLMNQSTUWabcdefghiklmnoprstuvwxyz
- vocab size: 47 unique tokens

Couple notes:
- Why the questions? So that this could be legitimately queried, like a prompt could be those precise questions
- This is succinct because the model is 100kb at most, like I really have to squish it down
- Could we have more? Not sure, let's start here. Worst case is we overfit to this, and if this happens, I'll train on a bigger dataset

I played around with parameters. How many heads? How many embedding dimensions? How many layers? There are so many experiments to run, and I definitely did not exhaustively find the best set of parameters. But that's not my goal - I just want _something that works_. I found that the following worked:

- 1 attention head per layer
- 2 layers
- 32 embedding width
- context length 31, using a sliding window context
- 27 232 parameters
- 107Kb in storage as float

By "worked" here I mean:
- it produced reasonable predictions when prompted. Sentences like "<todo>"
- it "fits" within the memory budget I have: 100Kb (ok it's 107Kb but this is close enough, I'm going to work with it)

This is a good start! Maybe I want to return to this step for a better model, but this easily fits within the flash we have - 4Mb, the matrices for individual steps fit within the RAM budget, and the results look good. So before I stress too much over getting the perfect model, let's attempt implementation. 



## Porting from tflite to C++
When running the model in NanoGPT's sample.py, there are two parts. One is the actual forward pass of the transformer, which produces a vector of logits, and the other is a loop that runs softmax over these logits, finds the probability of each vocab entry (token), samples from a distribution to get the next token, and then runs the next forward pass over the new token set. 

The first part, the actual forward pass of the transformer, is the model inference and it's here we use the model weights. The rest is just math. So to extract the model - the set of all weights, and structure of tensors and operations - in sample.py I use the following python code, where `model` is the `torch.nn` object whose `forward` function is the transformer pass inside the `mode.py` `generate` function:

```
traced = torch.jit.trace(model, x)
torch.jit.save(traced, "traced_model.pt")
```

This produces a pytorch file we can look at in [Netron](https://netron.app/). 

[Image]

Cool. We going to manually copy the weights from this? Well, no.\
Pytorch files are not particularly well optimised, having many redundant layers like transposes that are later cancelled by other transposes, or cast ops to int, or wahtever:

image

. Qualcomm has this marvellous platform called [Qualcomm AI Hub](https://aihub.qualcomm.com/) that can optimise a model for a device. (exmaple? Picture)\
While the device I'm optimising for isn't available - they have a selection of Qualcomm chips, funnily enough - this doesn't matter. All I'm really hoping to do is to strip the unnecessary layers from the model. We run it through their platform

image

image

and now we have a tflite file!
stats, netron, show this is better. 

Looking at this tflite in netron, we can break the neural network down into its parts and consider the code for each part. 

### Layernorm
- the [layernorm]() op. Straightforward elementwise mathematical operation. (TFLite does not natively have a Layernorm op, and so breaks this out into constituent operations)


### Q/K/V
- Q/K/V matrix multiplications with the embedding vectors. Matrix $\times$ vector, simple

### Scaled Self-Attention
- Scaled self-attention
[image]
We'll come back to this


### Feed-Forward Network
- Feed Forward Network [image]. I'll come back to this

### Residual Skip Connection

### Gather
This is simply indexing into an array. "Here's some indices for vocab elements, get the vocab elements they refer to".

- Residual skip-connection. This is a fancy way of saying "Add the result from this layer to whatever was at the start of the layer". So, vector addition. Easy.


### Putting it all together
Now all we need to do is connect those parts ... 

<code>

and that;s the whole model! 

## Can this run at all
Is it actually that simple? Well, let's see if this runs at all. While for my first ML-on-business-card I used the [RP2040](), this does software emulation of floating point hardware - that is, floating point math is actually done with 32-bit integers emulating what a dedicated floating point logic gate circuit would do. This will work, but be slow. Coincidentally, the next generation, the [RP2350]() chip had just come out, and this came with a FPU coprocessor - floating point hardware! So floating point arithmetic could be performed in a single clock cycle. A dev board was acquired, a compiler was run (and run and run and run and eventually the code compiled), a chip was flashed, aaaaaand ...

<results, speed>
10 tokens per second!! For a basic implementation, that's a lot more than I was expecting


So, what do we do from here? Is this it? It runs, produces sentences, yay done move on?\
Absolutely not! Now we need to see just how fast we can get this model running. The main metric I'll use is tokens per second on prompt processing and decode processing, but as decode is a fixed 31-length sliding window, it doesn't make much of a difference. 

There are a number of optimisations I have in mind: multithreading, unrolling, removing intermediates, RAM residency, quantisation. The next section will go through everything I tried, including failures. 

## Optimisations

TODO get most of this from the chat history at work

### Cache-coherent matmul
Both attention and Feed-Forward Networks rely on matrix multiplication. My thought was that to optimise for cache-coherency, as matrices tend to be stored linearly, we could write out one row of output at a time - keeping the whole row in cache - and therefore keep one row of matrix $A$ in $C = A\times B$ resident in cache as well, whilst iterating through the columns of B. That second iteration would have cache misses, but no more so than a regular iteration keeping one column of B, iterating through rows of A and rows of output. 

A nice idea in theory; it turns out, from the [datasheet](https://support.arm.com/documentation/100230/latest/) that the ARM Cortex M33, that the RP2350 is based on, has no CPU cache, and therefore ... this is irrelevant. Oh well. A nice idea.
 

### SRAM Residency


### Unrolling

### Removing Intermediates/Fusing

### KV Cache

### Ignoring unused output

### Multithreading

### Quantisation

## Circuit design and fabrication

I used PCBWay for this

## Conclusion

