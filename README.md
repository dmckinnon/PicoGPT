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

and thats the whole model! 

## Can this run at all

Is it actually that simple? Well, let's see if this runs at all. While for my first ML-on-business-card I used the [RP2040](), this does software emulation of floating point hardware - that is, floating point math is actually done with 32-bit integers emulating what a dedicated floating point logic gate circuit would do. This will work, but be slow. Coincidentally, the next generation, the [RP2350]() chip had just come out, and this came with a FPU coprocessor - floating point hardware! So floating point arithmetic could be performed in a single clock cycle. A dev board was acquired, a compiler was run (and run and run and run and eventually the code compiled), a chip was flashed, aaaaaand ...

<results, speed>
10 tokens per second!! For a basic implementation, that's a lot more than I was expecting


So, what do we do from here? Is this it? It runs, produces sentences, yay done move on?\
Absolutely not! Now we need to see just how fast we can get this model running. The main metric I'll use is tokens per second on prompt processing and decode processing, but as decode is a fixed 31-length sliding window, it doesn't make much of a difference. 

There are a number of optimisations I have in mind: multithreading, unrolling, removing intermediates, RAM residency, quantisation. The next section will go through everything I tried, including failures. 

## Optimisations

There are several benchmarks I'm using here. For some things like matrix multiplication, we can easily creawte a standalone benchmark. I'm also giving the model a small prompt and generating 100 output tokens based on this prompt, and we'll see how that is affected each time. 

All optimisations were also checked for accuracy, both on numerical benchmarks (computing matrices etc) and also checking token-generation output. Logits changed slightly, but never enough to change probabilities. Only optimisations that maintain this accuracy are considered. 

TODO get most of this from the chat history at work

### Cache-coherent matmul
Both attention and Feed-Forward Networks rely on matrix multiplication. My thought was that to optimise for cache-coherency, as matrices tend to be stored linearly, we could write out one row of output at a time - keeping the whole row in cache - and therefore keep one row of matrix $A$ in $C = A\times B$ resident in cache as well, whilst iterating through the columns of B. That second iteration would have cache misses, but no more so than a regular iteration keeping one column of B, iterating through rows of A and rows of output. 

A nice idea in theory; it turns out, from the [datasheet](https://support.arm.com/documentation/100230/latest/) that the ARM Cortex M33, that the RP2350 is based on, has no CPU cache, and therefore ... this is irrelevant. Oh well. A nice idea.
 

### SRAM Residency
The RP2350 fetches weights from flash and places them into SRAM to do the matrix calculations. There is a 16 KiB cache for this, buuuuut ... what if we avoided that fetch? Each fetch from flash has a delay due to the cache (could have a miss) and the QSPI memory interface. It's not large, but we have 500Kb of SRAM and a 107Kb model. We can fit this entirely within RAM once upon startup - incurring a one-time cost - but then reuse weights with abandon, processing as many new tokens as we see fit without any extra fetch overhead. 

Here's the code:

Here's the timing, with and without:

A considerable improvement! An expected one, but nonetheless rewarding to see. 

### Unrolling
This is essentially pipelining loops. Consider a standard matmul loop:

```
for (B matrix rows)
    for (A matrix cols)
        for (C matrix position)
            C[i,j] += a*b
```

In the inner-most loop, we do one multiplication, one addition, one store, and then check loop conditions, have a potential branch, and set up the loop again. There's almost as much overhead as there is loop calculation!

What if we were to do more per loop? Consider:

```
for (B matrix rows)
    for (4 A matrix cols)
        for (4 C matrix positions)
            C[i_1,j_1] += a_1*b_1
            C[i_1,j_1] += a_1*b_1
            C[i_1,j_1] += a_1*b_1
            C[i_1,j_1] += a_1*b_1
```

We've essentially batched this matmul, doing 4 at a time. Obviously this requires the size to be divisible by 4, but we know the size a priori so we can make this work. Now we do 4 multiplications and 4 additions per loop overhead! The multiplications in the Floating Point coprocessor take a couple of cycles, but there are multiple floating-point registers we can use so these multiplies can be pipelined so we hide the delay on those registers behind the next multiply:

```
Example pipeline of 2

Before:
cycle 1: R0 = R1*R2
cycle 2: R0 locked behind second cycle  of multiply, nothing happens
cycle 3: R4 += R0
cycle 4: loop stuff

After:
cycle 1: R0 = R1*R2
cycle 2: R0 locked, R5 = R6*R7
cycle 3: R4 += R0, R5 locked
cycle 4: R8 += R5
cycle 5: loop stuff
```

In this 'after', there's no cycle where we are just waiting - we're always doing something. This may seem small, but recall it's the inner loop of three nested loops so this happens a lot. And that's just one matmul, of three matmuls per layer, for two layers, per token. The time difference should add up. 

Does it? Let's take a look at the results:

<results>

As expected, we get gainz. That's enormous though! 

What can be unrolled?\
Any loop, really. Matrix multiplications, elementwise operations like tensor addition or GELU, parts of the softmax function. Unroll everything!




### Fused FMA
Can we make the inner loop of matmuls even faster? Until now, this inner loop of the matrix multiplication has been using two instructions: floating point multiply, and floating point add. \
But there's an important feature in this floating point coprocessor: [Fused Multiply-Accumulate](https://pip-assets.raspberrypi.com/categories/1214-rp2350/documents/RP-008373-DS-3-rp2350-datasheet.pdf)! (Section 3.7.2.4)

This does exactly as it sounds, performing a multiply and accumulate (+=) all in one instruction. This should make our inner loop have half the number of instructions! That's twice as fast! It's a relatively simple change, just a compiler flag:

TODO

And this will replace floating point multiplies followed by accumulates with the single instruction. 

Check results:

Oh.\
This is slower.\
Hmm. \
Why is this slower?

I dug a bit but not enough to properly figure it out. It was faster than the single-multiple-accumulate per loop, but not faster than unrolling. My theory is that it uses more registers and slowed down the unrolled version with register swaps? No evidence to support this. I could dig into the assembly, but we got such a massive win from unrolling and there's more to try that if all else fails I'll come back, but otherwise ... meh. Chalk this one up as a loss. 



### Removing Intermediates/Fusing
The concept behind this idea is that we are constantly performing an operation, like adding two tensors, and then writing the result out to a third tensor in memory. The next operation will read this back from memory, perform something else on it, like some elementwise operation, and then write it back out again. \
There's a lot of inefficiency here. What if we fused the two? That is, what if we didn't have the intermediate tensor?\
For example, after FFN we have the residual skip-layer, where we add the tensor from the start of the layer back into the result from the layer. We could complete the matmul, write that out, and then loop through the tensor again and element-wise add. \
Or\
We could, while performing the matmul, add the residual to the layer-input tensor at the output of the matmul, and thereby avoid a whole separate iteration through the tensor?

let's see this in code:

```
Before:
for (rows of B)
    for (cols of A)
        for (elems in dot product)
            C[i] += a*b

for (elems in tensor)
    C[i] += input[i]


After:
for (rows of B)
    for (cols of A)
        for (elems in dot product) 
            C[i] += a*b
            C[i] += input[i]
```
We still have to do all the operations, but what this saves is all the loop overhead - not just in an unrolling fashion, but ALL the loop overhead of a second loop. \
We can do this for the two residual skip-connections, and also the GELU operation in the middle of the FFN. 

Let's check these separately:

<Skip-connections>

<GELU>

So the skip connection fusing works well. Why doesn't GELU? <something about the full erff??>

Oh well. We can still unroll the GELU and get an already-proven win there. 


### KV Cache


### Ignoring unused output
In the base implementation, we compute a whole vector of output logits, softmax to get the probabilities per index, and then choose a token per index. But previous indices have already been computed! We only need the last token. 

Simple: don't softmax the whole vector, just the last entry, and then only compute for this token. Removes unneeded compute. 

Let's check the speedup we get:


Saves a few hundred milliseconds over the 100 token generation. Not terrible, not great - much like reading 3.6 roentgens. But it causes no harm, and it is objectively faster, so we'll keep it. 

### Multithreading
Another seemingly obvious win. Until now, all algorithms have run on a single core - which we needed to clearly know what the speedup was, so this one has to come last. 

The Cortex M33 has two cores, each with a floating point coprocessor. There are no CPU caches, as mentioned previously, and both cores just read and write directly to SRAM. All we need to do is split the loops we intend to multi-thread correctly so no memory contention occurs, and this should just fall into place. We'll multithread the matrix multiplication loops - unrolled and with fused intermediates, so this means multithreading several implementations - and try the elementwise GELU loop. The softmax ... is finicky enough and doesn't take enouhg time for this to be worth it. I profiled the whole thing, and matmul dominates. Softmax was a mere 5% of time, to the ~60% of time spent on matmul. 

How should we split up the matrix multiplication? GELU loop is easy - first half to one core, second half to other core. Matrices, though - the cores need to share one matrix entirely. \
The way that sounds best on paper is, for $C = A\times B$, to split $A$ between the two cores, share $B$, and then also split the output $C$. The split would occur vertically through the matrix:

```

do this in latex

```

Because matrices are laid out linearly, row-major, this means that we give one linear array to each core:

```
A = [a_00 .... a0n a10 ... a1n ... a_m/20...am/2n] 
```

simple split, and we don't have to stress about cache coherency because no caches. The output matrix C is split similarly. \
There's some nuance I'm brushing over here with setting up the second core to run, passing jobs to it, etc, but we don't need to go into that. The code is simple enough, and the point here is the theory of splitting and multithreading.

Let's take a look at the results:

A clear win. We didn't really expect anything else - doing half the work simultaneously with the other half of the work, with minimal overhead.

### Quantisation
"I thought multithreading was the last optimisation"\
It is. I didn't end up doing quantisation, but it's here because it's a well known one, and a major one to consider. 

Quantisation is where instead of using floating point values for all inputs, outputs, and computation, we get the range of values we expect and squash all values in between into a range of 256 - so 8 bits. Now we are computing on 8 bits, and storing on 8 bits. The model is smaller, we can compute faster (while the FPU does handle arithmetic in a small number of clock cycles, the DSP engine in the Cortex M33 allows us to do arithmetic on 4 separate bytes simultaneously, giving us a speedup of nearly 4 within things like matrix multiplication). 

However, this comes with a significant accuracy hit. How bad?\
I did not quantify. After I saw the results we get from the above optimisations chained together, I ... didn't feel like quantisation added anything. Yes, I could potentially get another speedup, but this was already more than fast enough and from experience this could take an accuracy hit of up to 30%? Maybe worse?\
Large models quantise far more viciously, down to 2 or 1 bit, but they have enough parameters to overcome this brutally tight quantisation. I could therefore make a larger model and quantise it harder ... and this is a subject of later discussion.

So to wrap this up, I was happy with the speed and quality of floating point, and did not see a need. The original motivation for this project was indeed to use the DSP acceleration of int8 arithmetic, but ... now I want to try other things. See the conclusion for more details. 

For now, I'm considering optimisation done.

## Circuit design and fabrication
I'm using the same basic concept for the board as with my previous

I used PCBWay for this

## Conclusion

