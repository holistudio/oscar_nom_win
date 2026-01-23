# Development Log

## 2026-01-22

Alrighty, training loop now done. Claude suggested I add the following beyond the typical GPT training stuff I've learned:
- Add cosine annealing learning rate schedule, with a warm-up period of 10% of total steps.
- Gradient clipping to 1.0 to prevent exploding gradients

Assuming batch size of 2 for both training and validation, a single epoch takes around 2 mins, so 100 epochs will take around 3-4 hours. Not bad!

But with the small batch size I'm curious how the test performance will actually turn out.

Also I still have yet to think about and deal with the class imbalance.

To be continued...

## 2026-01-21

Good news is the transformer code is now updated and works for a random sequence of 100k token integers.

Bad / OK news is that the batch size has to be set to 2, otherwise my laptop GPU runs out of memory.

Claude recommends two ways to deal with this:

**Option A:** Process chunks in a loop with gradient checkpointing. Each chunk is encoded sequentially, gradients are reconstructed during backward pass. Slower but memory-efficient.

**Option B:** Freeze the chunk encoder. Use a pretrained encoder (like the first 6 layers of BERT), run it in `torch.no_grad()`, only train the aggregator. Much faster, much less memory, and honestly might work just as well for your task.

Coincidentally Option B is sorta where I was headed already, but I wanted to use the pretrained GPT-2 and keep it frozen.

Of course in my mind, there is an **Option C**: Keep going! Keep the batch size to 2, there are only 1000 or so training samples and 400 or so validation samples.

My `unit_train.py` also works so I'm confident I can take a real movie script from the training script and give it to the `OscarNomTransformer` model in a training loop. Time to start training this thing!

## 2026-01-20

Finally had some time to circle back to this and refresh my memory on how a transformer's encoder and decoder works. As I was re-writing a boilerplate version a couple things stood out to me that seem to be tricky to change if I want the transformer to process chunks of a movie script into a sequence of embeddings and then decode it into two logits for binary classification.

- Encoder and Decoder will need different `d_model` parameters
   - They can also have different `nhead` and `dim_ff` parameters
   - With different `d_model` parameters, Encoder and Decoder will have different positional encoders
- The same `token_emb`/`nn.Embedding` is typically applied to both `src` and `tgt`.
   - instead the `tgt_emb` embedding going into the Decoder could just be a linear layer that processes the embedding sequence?
- The big question in my mind is how the `memory` that usually gets passed from Encoder to Decoder will happen, since what I think I want is the Encoder to pass an embedding sequence.
   - The other question is what is the `tgt` in this problem anyway? In machine translation, `tgt` starts out as the target language start token and slowly grows as the Transformer translates from source language to target language. But in this binary classification problem, we're not really doing something like that. So shouldn't target just be something arbitrary/blank/random noise?

Just wanted to get these specific questions clarified before I go back to asking Claude what to do next haha.

Next step is just to give Claude this boiler plate and my rough diagram of what I want the transformer architecture to end up looking like and have Claude walk through step by step what changes have to be made. (Note: not via Claude Code, I'm looking for a customized tutorial from Claude, not a one-shot vibecoded solution...not yet at least)

## 2026-01-15

Just took a few minutes to clean up and organize the data processing and exploratory data analysis notebooks and datasets.

Time to start coding a transformer, in my mind the "chunky transformer encoder-decoder" which has to take the movie script window chunk by window chunk first before predicting probability of Oscar nomination.

...and now I'm realizing that I need to take some more time to refresh my memory on the specific pieces of the transformer and GPT-2 so that I can build up towards something that can handle a movie script chunk by chunk. 

For today I'm just going to make sure the "basic puzzle pieces" are all here:
- a basic Transformer with separate encoder and decoders
- a GPT-2 model with pre-trained weights (courtesy of Sebastian Raschka's LLMs from Scratch book)

### TODOs for later

- [ ] Get Chunky versions of transformer and GPT-2 to work. 
- [ ] Make sure these are done in separate branches
- [x] Write a simple "unit test" script: load a random movie script in the training set and give it to a model. It should at least spit out a probability with no issues.
- [ ] Exploratory data analysis of word frequencies in movie screenplays.
- [ ] Consider ways to deal with the class imbalance during training.
- [ ] Training-validation-test loops
- [ ] Use Raytune for tuning hyperparameters.

## 2026-01-14

Hello world!

This project is a revisit of this one-day hacking/pairing [project](https://github.com/minsun-ss/recurse-pairing) at the Recurse Center with other programmers. The goal is the same as before: Given a movie's screenplay text (and other metadata) predict whether that movie gets Oscar nominations or wins, using this HuggingFace [dataset](https://huggingface.co/datasets/Francis2003/Movie-O-Label).

Building on what we worked on before, I want to explore a few different approaches here. Will circle back and document them clearly after some brainstorming and chatting with Claude...

...and take some time to refresh my memory on important bits about the dataset:
- The HuggingFace dataset doesn't label which movies won an Oscar, but one of its [reference datasets](https://github.com/DLu/oscar_data) does, so the previous project made sure to clean the data in that aspect.
- There is significant class imbalance when it comes to Oscar nominations and wins: around ~19% of the movies got nominated, and ~4-5% actually won. This presents a serious challeng for a classifier model.
- Assuming a GPT-2 tokenizer (byte-wise encoding), the screenplays are on average 37k tokens long, ranging between 7k and **100k tokens**.
- During the last attempt, this became an issue because an entire script couldn't fit in the context window of a GPT all in one shot and hit my 4090 laptop GPU's memory limit.
- The dataset also contains script embeddings, which to me represents a compression of the input data to get things to fit into a smaller context window.

My basic "pie-in-the-sky" goals: 
- Don't use the dataset's script embeddings.
- Revisit how this could be done with:
    - Transformer Encoder and Decoder
    - pre-trained GPT-2 and Transformer Decoder
    - If I have time look at advanced Transformer-based models since GPT-2 (Longformer, Mamba)
- For funzies, give a whole script to an open source local LLM and:
    - Measure its classification performance out-of-the-box.
    - Figure out a way to fine tune it.

More realistically I'm thinking my goals/outcomes should be:
- Try the following without the dataset's script embeddings:
    - Transformer Encoder and Decoder
    - pre-trained GPT-2 and Transformer Decoder
- And if the performance still sucks, use the script embeddings to train the TransformerDecoder from scratch

The other "pie-in-the-sky" goals are beyond my current coding/ML knowledge since I haven't read those advanced transformer papers or learned how to code/re-tool existing LLMs...yet :D



### TODOs for later

- [x] Re-organize notebook files and datasets 
- [ ] Exploratory data analysis of word frequencies in movie screenplays.

### Regarding AI-assisted workflows

I am starting off this project with two important aspects of Claude:
- Claude Projects where all my chats share a context in terms of instructions and other files.
- Claude Code which I plan to use to help quickly get some exploratory data analysis out of the way before diving into the ML model architecture and training myself.

I am also planning to use both Gemini Code Assist and Claude Code for help with coding the ML models as I am curious to see how they approach the coding differently.

It may be valuable for others to note my Claude Project instructions here:

```
# Role and Purpose

You are a fellow ML research engineer on this project.

## Context

- Given a movie's screenplay text (and other metadata) predict whether that movie gets Oscar nominations or wins.
- This amounts to a binary classification problem.
- Dataset: https://huggingface.co/datasets/Francis2003/Movie-O-Label

## My Goals

- Experiment with various transformer, GPT, and LLM related ways to classify the movie scripts.

## General Behavior

- Be a helpful research collaborator. Balance giving advice presuming an approach that I am interested in with occasional challenges to approach an ML problem differently.
- Prioritize comments on how to improve the model's acccuracy.

## Output Preferences

- **Style:** Explain in the style of Andrej Karpathy.

## Key Constraints and Limitations

- Do NOT provide code snippets beyond what I ask for.
- Always assume I am working with a decent CUDA-enabled laptop GPU (RTX 4090 with 16GB vRAM)
- Again it's the laptop GPU, not the desktop PC GPU.
- When uncertain: Suggest ways cloud GPU computing resources may be required, with a clear detailing of the costs.
```