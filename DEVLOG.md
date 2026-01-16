# Development Log

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
- [ ] Write a simple "unit test" script: load a random movie script in the training set and give it to a model. It should at least spit out a probability with no issues.
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