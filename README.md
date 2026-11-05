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
- [Build and serial usage](#build-and-serial-usage)
- [Model structure, size, and training](#model-structure-size-and-training)
- [Porting from tflite to C++](#porting-from-tflite-to-c)
- [Can this run at all](#can-this-run-at-all)
- [Optimisations](#optimisations)
- [Circuit design and fabrication](#circuit-design-and-fabrication)
- [Conclusion](#conclusion)
- [Usage](#usage)

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


```
for (int i = 0; i < SEQ_LEN; ++i) {
    // Compute mean
    float mean = 0.0f;
    for (int d = 0; d < DIM; ++d) {
        mean += input[i][d];
    }
    mean /= DIM;

    // Compute variance
    float var = 0.0f;
    for (int d = 0; d < DIM; ++d) {
        float val = input[i][d];
        float diff = val - mean;
        var += diff * diff;
    }
    var /= (float)DIM;

    float inv_std = 1.0f / sqrtf(var + LN_EPS);
    // Normalize
    for (int d = 0; d < DIM; ++d) {
        float val = input[i][d];
        float norm = (val - mean) * inv_std; /* gamma=1, beta=0 */
        output[i][d] = norm;
    }
}
```


### Q/K/V
- Q/K/V matrix multiplications with the embedding vectors. Matrix $\times$ vector, simple

```

```

### Scaled Self-Attention
- Scaled self-attention
[image]

```
for (int i = 0; i < SEQ_LEN; ++i) {
    // Compute scores against all keys
    float scores[SEQ_LEN];
    float max_score = -INFINITY;
    for (int j = 0; j < SEQ_LEN; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < DIM; ++d) {
            float q_f = Q[i][d];
            float k_f = K[j][d];
            dot += q_f * k_f;
        }
        dot *= scale;
        scores[j] = dot;
        if (dot > max_score) max_score = dot;
    }

    // Subtract max for numerical stability and do softmax
    float exp_scores[SEQ_LEN];
    float sum_exp = 0.0f;
    for (int j = 0; j < SEQ_LEN; ++j) {
        float e = expf(scores[j] - max_score);
        exp_scores[j] = e;
        sum_exp += e;
    }

    // Compute weighted sum of V
    for (int d = 0; d < DIM; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < SEQ_LEN; ++j) {
            float weight = exp_scores[j] / sum_exp;
            float v_f = V[j][d];
            acc += weight * v_f;
        }
        context[i][d] = acc;
    }
}
```


### Feed-Forward Network
- Feed Forward Network [image]. I'll come back to this

### Residual Skip Connection

### Gather
This is simply indexing into an array. "Here's some indices for vocab elements, get the vocab elements they refer to".

- Residual skip-connection. This is a fancy way of saying "Add the result from this layer to whatever was at the start of the layer". So, vector addition. Easy.


### Putting it all together
Now all we need to do is connect those parts ... 

```
compute_qkv(input, weight_qkv, Q, K, V);

/* Step 2: Scaled dot‑product attention. */
q8_t context[SEQ_LEN][DIM];
compute_attention(Q, K, V, context);

/* Step 3: Residual add.  Add context back to input. */
q8_t resid1[SEQ_LEN][DIM];
add_residual(input, context, resid1);

/* Step 4: LayerNorm. */
q8_t ln1[SEQ_LEN][DIM];
apply_layernorm(resid1, ln1);

/* Step 5: Feed‑forward network. */
q8_t hidden[SEQ_LEN][HIDDEN_DIM];
fully_connected(ln1, weight_fc1, hidden);
relu_inplace(hidden);
q8_t mlp_out[SEQ_LEN][DIM];
fully_connected2(hidden, weight_fc2, mlp_out);

/* Step 6: Second residual add. */
q8_t resid2[SEQ_LEN][DIM];
add_residual(mlp_out, ln1, resid2);

/* Step 7: Optional LayerNorm. 
```

and thats the whole model! 

## Can this run at all

Is it actually that simple? Well, let's see if this runs at all. While for my first ML-on-business-card I used the [RP2040](), this does software emulation of floating point hardware - that is, floating point math is actually done with 32-bit integers emulating what a dedicated floating point logic gate circuit would do. This will work, but be slow. Coincidentally, the next generation, the [RP2350]() chip had just come out, and this came with a FPU coprocessor - floating point hardware! So floating point arithmetic could be performed in a single clock cycle. A dev board was acquired, a compiler was run (and run and run and run and eventually the code compiled), a chip was flashed, aaaaaand ...

<results, speed> - get this from chat running it
10 tokens per second!! For a basic implementation, that's a lot more than I was expecting


So, what do we do from here? Is this it? It runs, produces sentences, yay done move on?\
Absolutely not! Now we need to see just how fast we can get this model running. The main metric I'll use is tokens per second on prompt processing and decode processing, but as decode is a fixed 31-length sliding window, it doesn't make much of a difference. 

There are a number of optimisations I have in mind: multithreading, unrolling, removing intermediates, RAM residency, quantisation. The next section will go through everything I tried, including failures. 

## Optimisations

There are several benchmarks I'm using here. For some things like matrix multiplication, we can easily creawte a standalone benchmark. I'm also giving the model a small prompt and generating 100 output tokens based on this prompt, and we'll see how that is affected each time. 

All optimisations were also checked for accuracy, both on numerical benchmarks (computing matrices etc) and also checking token-generation output. Logits changed slightly, but never enough to change probabilities. Only optimisations that maintain this accuracy are considered. 


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
            C[i_2,j_2] += a_2*b_2
            C[i_3,j_3] += a_3*b_3
            C[i_4,j_4] += a_4*b_4
```

We've essentially batched this matmul, doing 4 at a time. Obviously this requires the size to be divisible by 4, but we know the size a priori so we can make this work. Now we do 4 multiplications and 4 additions per loop overhead! The multiplications in the Floating Point coprocessor take a couple of cycles, but there are multiple floating-point registers we can use so these multiplies can be pipelined so we hide the delay on those registers behind the next multiply:

```
Example with unrolling 2 at a time

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

In this 'after', there's no cycle where we are just waiting - we're always doing something. This may seem small, but recall it's the inner loop of three nested loops so this happens a lot. And that's just one matmul, of three matmuls per layer, for two layers, per token. The time saving should add up. 

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


Saves a few hundred milliseconds over the 100 token generation. Not terrible, not great - much like reading 3.6 roentgens on a dosimeter. But it causes no harm, and it is objectively faster, so we'll keep it. 

### Multithreading
Another seemingly obvious win. Until now, all algorithms have run on a single core - which we needed to clearly know what the speedup was, so this one has to come last. 

The Cortex M33 has two cores, each with a floating point coprocessor. There are no CPU caches, as mentioned previously, and both cores just read and write directly to SRAM. All we need to do is split the loops we intend to multi-thread correctly so no memory contention occurs, and this should just fall into place. We'll multithread the matrix multiplication loops - unrolled and with fused intermediates, so this means multithreading several implementations - and try the elementwise GELU loop. The softmax ... is finicky enough and doesn't take enouhg time for this to be worth it. I profiled the whole thing, and matmul dominates. Softmax was a mere 5% of time, to the ~60% of time spent on matmul. 

How should we split up the matrix multiplication? GELU loop is easy - first half to one core, second half to other core. Matrices, though - the cores need to share one matrix entirely. \
The way that sounds best on paper is, for $C = A\times B$, to split $A$ between the two cores, share $B$, and then also split the output $C$. The split would occur vertically through the matrix:


$$
C = AB =
\left[\begin{array}{c}
A_1 \\ \hline A_2
\end{array}\right] B
=
\left[\begin{array}{c}
A_1 B \\ \hline A_2 B
\end{array}\right]
=
\left[\begin{array}{c}
C_1 \\ \hline C_2
\end{array}\right]
$$

where $A_1, A_2 \in \mathbb{R}^{(m/2) \times k}$ and $C_1, C_2 \in \mathbb{R}^{(m/2) \times n}$. Thread $i$ computes $C_i = A_i B$.


This works because $A$ is laid out row by row in memory, so by given the first half of the rows to one core, we give it a contiguous memory block, and the second half of the rows is also one contiguous memory block going to the second core. There's no memory contention or overlap. $C$ is laid out the same way.\
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
I'm using the same basic concept for the board as with my previous [card](https://dmckinnon.github.io/ML-Business-card/) - the USB C port in the corner, thin enough to slot into a USB male connector. There's a couple of big differences:

<image>

1. This uses the RP2354, not RP2350. What's the difference? The 2354 has 2MB of flash built in, instead of needing an external flash chip. It's not much, and I have spare flash chips sitting around, but it makes circuit design and routing just that little bit simpler
2. I'm breaking out all the rest of the chip's GPIO. The intention here is that this board is reusable as a Raspberry Pi Pico 2 dev board, just a nice touch to whoever you give this away to, OR if I just happen to need another dev board and my main one is tied up in another project. Reusability is nice. 

I used [PCBWay](www.pcbway.com) for fabrication. I've talked about [using them before](), and choose them each time. It's such an easy interface and experience:

![](https://dmckinnon.github.io/assets/nfcpre/pcbway.png)

- to get a quote, just put in board dimensions and other details like colour, and approximate price is ready for you
- to get the board itself, [export and uploading all necessary files from KiCad](https://www.pcbway.com/blog/PCB_Design_Tutorial/How_to_Export_Gerber_and_Production_Files_in_KiCad.html) - there might be some clarification but they'll help you through issues
- board is at your door not long after. They even have the option of doing all the components for you (i still do this myself, but it's just a preference)

Here's the finished board! 

<image>

It's easy to flash, and I'm realy happy with Raspberry Pi for this one - holding the BOOTSEL button puts it in flashing mode and it appears as a storage device on the computer. Drag and drop the binary, it reboots, and then runs. See below for [Usage](#usage)

## Conclusion
I'm really pleased with how this turned out. I never got to the original idea that kicked it off - using the DSP to accelerate a quantised version to reasonable speeds - but this was still fun.\
The model is definitely smaller than it could be, and considerably overfit, so it comes across as too regurgitative of the training data. But the optimisations drastically speed things up, and that proves a good initial point - I could use a much bigger model, and still get reasonable speed. \
Other tech workers have seen this demonstrated and were delighted or impressed at the concept and execution, also proving the point that it's a useful exercise and gets the attention of a potential hiring manager. 

### What next
As mentioned, the model is small and overfit, and runs far faster than necessary. Given that the chip I used has 2Mb flash, what I'd like to try next is a _much_ bigger model - perhaps a whole megabyte, perhaps even more. I'd practically write my biography, so far more detail and training data, and have it learn that to answer any query deeply. Then there's problems to reckon with like paging the weights in and out of flash with one core while the other core performs matmuls, and we might need to make more serious optimisations, cos there's so much more model to process. It'll run a whole lot slower, but this would be a fun test of just how big I can go and maintain something like, say, 10 tokens per second generation speed. 

### A note about AI usage


## Usage

### Build and serial usage

### Makefile versus CMake

There are two build entry points in [PicoGPT/](PicoGPT/), targeting different
machines:

| Entry point | Purpose | Output |
|-------------|---------|--------|
| [Makefile](PicoGPT/Makefile) | Hand-written rules for native GCC, tests and extraction | Linux/host executables such as `gpt_model` |
| [CMakeLists.txt](PicoGPT/CMakeLists.txt) | Configure the Pico SDK, cross-compiler, board, libraries and memory layout | RP2350 firmware: ELF, UF2, map file |

Make executes build rules. CMake generates those rules for a backend such as
Make or Ninja. Thus the Pico build can also contain a generated Makefile,
but it is not the hand-written host Makefile. Use `cmake --build <directory>`
so CMake invokes the correct backend; plain `make` in this source directory
builds the host program, not the Pico firmware.

The [shell wrapper](PicoGPT/build_and_run.sh) now deliberately builds and runs
the native host demo. Firmware must be copied to the Pico rather than executed
as a Linux program.

Native commands (run in WSL for this checkout):

```sh
cd "$HOME/PicoGPT/PicoGPT"
make -B gpt_model
./gpt_model
# Equivalent host build/run:
bash build_and_run.sh

# Numerical tests, with the current dual-core dispatch exercised on host threads:
make parity-test PYTHON=./gpt_venv/bin/python \
  CFLAGS='-Wall -Wextra -Werror -O2 -std=c99 -ffp-contract=off -DGPT_MATMUL_UNROLL4=1 -DGPT_ATTENTION_UNROLL4=1 -DGPT_FFN_FUSE_GELU=0 -DGPT_MULTICORE=1'
```

The host demo is still a diagnostic program. The interactive USB console below
is implemented in [pico_main.c](PicoGPT/pico_main.c).

### Interactive Pico 2 firmware

Configure the SRAM-resident, dual-core, separate-GELU, unrolled implementation:

```sh
cd "$HOME/PicoGPT/PicoGPT"
cmake -S . -B build/pico-serial-2.3.1 \
  -DPICO_SDK_PATH="$HOME/.pico-sdk/sdk/2.3.1" \
  -DPICO_TOOLCHAIN_PATH="$HOME/.pico-sdk/toolchain/15_2_Rel1" \
  -DPICO_BOARD=pico2 -DCMAKE_BUILD_TYPE=Release \
  -DPICO_DEFAULT_BINARY_TYPE=copy_to_ram \
  -DPICOTOOL_FORCE_FETCH_FROM_GIT=ON \
  -DGPT_MATMUL_UNROLL4=ON -DGPT_ATTENTION_UNROLL4=ON \
  -DGPT_FFN_FUSE_GELU=OFF -DGPT_MULTICORE=ON -DGPT_PROFILE_ENABLED=OFF
cmake --build build/pico-serial-2.3.1 --parallel 4
```

SDK/compiler paths are specific to the local installation; adjust them on
another machine. The toolchain host architecture here is aarch64. The
`PICOTOOL_FORCE_FETCH_FROM_GIT` option builds the SDK's build-only picotool,
avoiding the extension-provided executable's missing `libusb` dependency in
this WSL environment. The first configuration needs network access.

After configuration, only the `cmake --build ...` command is needed to rebuild.
Use a new build directory when changing SDK/compiler versions to avoid mixing
cached paths.

The flashable image is `PicoGPT/build/pico-serial-2.3.1/PicoGPT.uf2`.
Hold BOOTSEL while connecting the Pico 2, copy the UF2 onto its RP2350 drive,
and then open the USB serial port. A 115200 baud terminal setting is conventional
(USB CDC is not clocked by that UART setting). Enable DTR, disable local echo,
and send lines terminated by CR, LF or CRLF. In WSL, Windows may own the serial
device; using a Windows terminal is fine.

On connection, the firmware shows help and `pico>`. It does **not** automatically
run benchmarks or generate text.

| Input, followed by Enter | Behavior |
|--------------------------|----------|
| `/random` | Generate 100 new characters from a single newline token with a fresh seed |
| `What did he study?` | Generate 100 new characters using that exact prompt |
| `/repeat` | Reuse the last successful prompt and seed |
| `/bench` | Run matmul benchmarks, reference check, seed-42 generation and memory reports |
| `/memory` | Print current runtime memory diagnostics |
| `/profile` | Run stage and SRAM-bus profiling, if compiled in |
| `/help` | Show commands and input limits |

The former immediate `r` and `p` shortcuts are replaced by slash commands so
ordinary prompt characters are not intercepted. Commands require Enter.
Backspace/DEL edits a line; Ctrl-C cancels the **input line**, not an inference
already running. Empty lines do nothing. Wait for `pico>` before sending the
next request; there is no concurrent command-input task during generation.

Prompts must contain **1-31 characters** from the exact, case-sensitive
[47-character vocabulary](PicoGPT/gpt_vocab.c). Unsupported characters are
rejected with their byte value and position. Non-ASCII/control input and lines
longer than 31 characters are rejected as whole lines, never silently truncated.
After such an error, press Enter or Ctrl-C to start a new line. Reconnecting
clears a partially entered line.

Every successful request produces **100 new model-sampled characters**, not
100 total characters including the prompt. The model uses the latest 31 tokens
as a rolling context. Temperature remains 0.8. `/random` and custom prompts
obtain a seed from the SDK `pico_rand` API, using RP2350 entropy sources, and
print that seed. Output is sampled from model probabilities, not from a uniform
character distribution. Different seeds may still produce the same text.
`/repeat` is reproducible with the same firmware; it survives a serial reconnect,
not a power cycle. `/bench`, errors and diagnostic commands do not overwrite
the remembered prompt/seed.

### Timing and writeup measurements

Each generation prints three separate figures:

- **Model processing:** time inside `gpt_inference` across all new tokens,
  excluding sampling, output and input entry.
- **Token sampling:** time selecting tokens from the logits, including the
  sampling softmax and RNG calls, excluding text output.
- **Total generation:** elapsed generation-loop time including model calls,
  validation, sampling, output calls, context updates and bookkeeping.
  This excludes seed acquisition, the printed header, and typing the prompt;
  it does not guarantee all buffered USB output has reached the terminal.

These are elapsed times on core 0, not sums of both cores' CPU time. Inference
includes the completion wait for parallel kernels. A failed generation
explicitly reports failure, the number of characters completed, and timings
for the attempted work. Normal runs omit benchmark and arena dumps; `/bench`
retains those measurements and the fixed seed 42 for repeatable writeup runs.

An interactive profiling image is also available at
`PicoGPT/build/pico-serial-profile-2.3.1/PicoGPT.uf2`. To rebuild it, use the
same configuration command with `-B build/pico-serial-profile-2.3.1` and
`-DGPT_PROFILE_ENABLED=ON`, then build that directory. Send `/profile` and
Enter. Profiling changes code layout and timing even when capture is inactive;
use the nonprofiling image for final performance figures.

Both images retain the 144 KiB arena (observed model peak **107,216 bytes**)
and guarded 4 KiB stacks for each core. The first interactive build reserves
**311,484 bytes of main RAM** and uses **159,264 bytes of flash**. The profiling
variant reserves **315,772 bytes of main RAM** and uses **163,344 bytes of flash**.
Stack reservations are in separate scratch banks and are additional to the
main-RAM figures. These sizes are build-specific.

All 22 host tests pass in multicore mode, including numerical comparisons for
prompted generation, line limits/editing/reconnection, random-seed replay,
on-demand diagnostics, and timing accounting. AddressSanitizer and
UndefinedBehaviorSanitizer also pass. Host serial/RNG/clock shims validate logic,
not USB hardware behavior, entropy quality, or on-board timing; those still
require testing this image on the Pico.
