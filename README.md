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
- Can this run at all
- Optimisations
- Results

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
- X final parameters (TODO need to get this working)

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

- 2 attention heads
- 2 layers
- context length 32, using a sliding window context

By "worked" here I mean:
- it produced reasonable predictions when prompted. Sentences like "<todo>"
- it fit within the memory budget I have: X kb

This is a good start! Maybe I want to return to this step for a better model, but this easily fits within the flash we have - 4Mb, the matrices for individual steps fit within the RAM budget, and the results look good. So before I stress too much over getting the perfect model, let's attempt implementation. 



Shakespeare stats: length of dataset in characters: 1,115,394
all the unique characters: 
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
vocab size: 65
train has 1,003,854 tokens
val has 111,540 tokens

##### baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
 10.65M params


My data:

David studied at UNSW Univeristy of New South Wales from 2012 to 2016
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

My data stats: 
length of dataset in characters: 582
all the unique characters: 
 ,-.012346?ADHLMNQSTUWabcdefghiklmnoprstuvwxyz
vocab size: 47
train has 523 tokens
val has 59 tokens


model 27K params
Get proper answer from work thing

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
Pytorch files are not particularly well optimised, having many redundant layers like transposes that are later cancelled by other transposes. Qualcomm has this marvellous platform called [Qualcomm AI Hub](https://aihub.qualcomm.com/) that can optimise a model for a device. (exmaple? Picture)\
While the device I'm optimising for isn't available - they have a selection of Qualcomm chips, funnily enough - this doesn't matter. All I'm really hoping to do is to strip the unnecessary layers from the model. We run it through their platform

image

image

and now we have a tflite file!
stats, netron, show this is better. 



