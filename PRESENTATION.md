## Motivation

I struggled too. Coming from a traditional full-stack background, diving into a new paradigm without a clear ‘Hello World’ was tough. It required peeling back layers and simplifying concepts—yet it still felt challenging. What made the difference? Hands-on practice. And that’s exactly what we’re going to do today.

## Introduction

In simple terms: Feed a massive amount of text into an AI model, which learns patterns and connections between words, so it can generate meaningful text on its own.

TODO: Insert Diagram AI[ML[DL[LLMs]]]

### 📜History and Evolution

Both of below papers were **landmarks in NLP**, ultimately led to the state we are in today:  
- **["Attention Is All You Need" (2017)](https://arxiv.org/abs/1706.03762)** introduced **Transformers**, replacing RNNs/LSTMs.  
- **["Improving Language Understanding by Generative Pre-Training" (2018, GPT-1)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)** showed that **pretraining transformers** on unlabeled data enables **transfer learning for NLP**.

---

#### 🛠 Things to know

| **Feature** | **"Attention Is All You Need" (2017, Transformers)** | **"Improving Language Understanding by Generative Pre-Training" (2018, GPT-1)** |
|------------|-------------------------------------------------|------------------------------------------|
| **Main Contribution** | Introduced **Transformer architecture**, eliminating RNNs. | Introduced **unsupervised pretraining + fine-tuning** for NLP. |
| **Breakthrough** | Replaced **RNNs/LSTMs with self-attention** (faster training & better long-range dependencies). | Showed that **pretraining a language model** on raw text **reduces the need for labeled data**. |
| **Architecture** | **Full Transformer (Encoder + Decoder)** | **Decoder-only Transformer (Autoregressive)** |
| **Training Objective** | **Self-attention for parallel processing** | **Causal Language Modeling (next-word prediction)** |
| **Key Mechanism** | **Self-attention replaces recurrence** → allows **parallel training**. | **Transfer learning for NLP** (train once, fine-tune on multiple tasks). |
| **Impact on Speed** | **Massively faster** than RNNs (parallelizable). | Enabled **large-scale language modeling** by removing task-specific training. |
| **What It Replaced** | Replaced **RNNs and LSTMs**. | Replaced **task-specific NLP models** with a **single pretrained model**. |
| **What It Enabled** | **Foundation for all future Transformer-based models** (BERT, GPT, T5, ViTs). | **Foundation for GPT-2, GPT-3, ChatGPT, and other generative AI models**. |
| **Primary Use Cases** | Machine translation, NLP, speech, and vision models. | Text generation, chatbots, question answering, summarization. |
| **Limitations** | Required **huge training data** and computational power. | **Unidirectional training** (GPT-1 couldn’t see future words). |

---

#### 🔍 How These Papers Differed from Prior Work

| **Feature** | **Prior Work (RNNs, LSTMs, Word2Vec, etc.)** | **"Attention Is All You Need" (2017)** | **"GPT-1" (2018)** |
|------------|--------------------------------------|-------------------------|------------------|
| **Architecture Type** | **Recurrent (RNNs, LSTMs, CNNs for NLP)** | **Fully transformer-based (no recurrence)** | **Transformer decoder (left-to-right only)** |
| **Training Type** | **Supervised learning** (task-specific) | **Unsupervised self-attention training** | **Pretraining + fine-tuning** |
| **Parallelization** | ❌ No (RNNs are sequential) | ✅ Yes (Self-attention enables parallel processing) | ✅ Yes (Transformer-based) |
| **Context Window** | Short (RNNs forget long-range dependencies) | Large (Self-attention captures long-term dependencies) | Large (Good for generative text modeling) |
| **Scalability** | Hard to scale (due to recurrence) | Scalable (can be stacked into deeper models) | Very scalable (inspired GPT-2, GPT-3, GPT-4) |
| **Major Weakness** | **Slow training, memory issues** | **Computationally expensive** | **Lack of bidirectionality (unlike BERT)** |

---
**Together, these papers led to the rise of models like GPT-3, ChatGPT, BERT, T5, and other AI advances**.

**NOTE**: the next-word prediction pretraining task for GPT models, the system learns to predict the upcoming word in a sentence by looking at the words that have come before it. This approach helps the model understand how words and phrases typically fit together in language, forming a foundation that can be applied to various other tasks.

## Architecture

TODO: Insert Diagram
TODO: Prepare > Train > Consume

### What happens internally?

| **Stage** | **Step**                   | **What Happens?**                                           |
|-----------|----------------------------|-------------------------------------------------------------|
| Prepare   | **1. Tokenization**         | Break text into tokens and convert to numbers.              |
| Prepare   | **2. Embeddings**          | Convert numbers into **meaningful vector representations**. |
| Train     | **3. Self-Attention**      | Determine which words are important to each other.          |
| Train     | **4. Feed-Forward Layers** | Process attention information and make decisions.           |
| Train     | **5. Positional Encoding** | Ensure word order is considered.                            |
| Consume   | **6. Decoding ** | Predict the next word, one step at a time.                  |


### **1️⃣ Tokenization (Turning Words into Numbers)**
Before processing, text is broken into **tokens** (smaller pieces).  
Example:  
- Sentence: **"The cat sat on the mat."**  
- Tokens: `["The", "cat", "sat", "on", "the", "mat", "."]`  
- Each token is then converted into a **number** for processing.

---

### **2️⃣ Embeddings (Understanding Word Meaning)**
Transformers convert tokens into **vector representations (embeddings)**—turning words into **lists of numbers** that capture meaning.

---

### **3️⃣ Self-Attention (Understanding Word Relationships)**
This is the **core innovation of Transformers**. Instead of reading words in order, the model **assigns importance to different words based on context**.

#### **Example:**
🔹 Sentence: **"The animal didn’t cross the road because it was too tired."**  
🔹 Question: What does **"it"** refer to?  
- A Transformer, using **self-attention**, looks at the entire sentence and determines that **"it" refers to "animal"**.  

#### **Attention Score Example:**
| Word  | The | animal | didn’t | cross | the | road | because | it | was | tired |
|--------|----|--------|--------|--------|----|----|----|----|----|----|
| **it** | 0  | **0.9** | 0  | 0  | 0  | 0  | 0  | **1.0** | 0  | 0 |

Here, **"it" strongly attends to "animal"**, meaning the model correctly understands the reference.

---

### **4️⃣ Feed-Forward Layers (Making Decisions)**
Once self-attention determines word relationships, the model **passes the data through neural network layers** to learn patterns and make decisions.

---

### **5️⃣ Positional Encoding (Remembering Word Order)**
Transformers **don’t process text sequentially** like RNNs, so they add **positional encodings**—extra information that tells the model where each word appears in a sentence.

Example:  
- **"The cat sat on the mat."** → The model understands **"The" comes first, "mat" comes last**.

---

### **6️⃣ Decoding (Generating the Next Word)**
Once the Transformer understands a sentence, it can **predict the next word** in a sequence.

Example (GPT-style generation):  
- **Input:** "The sun is shining, and the sky is"  
- **GPT predicts:** `"blue."`  

It does this by:
- **Processing the input** using self-attention.
- **Selecting the most probable next word** based on learned patterns.
- **Repeating the process** to generate longer text.

🔹 Example: Generating a sentence  
1️⃣ Input: `"The dog ran"`  
2️⃣ GPT predicts: `"toward"`  
3️⃣ GPT appends `"toward"` to input and predicts the next word: `"the"`  
4️⃣ GPT appends `"the"` and predicts: `"park."`  

Final output: `"The dog ran toward the park."`

---
## Lets get building our MiniGPT

Goto [Hugging Face](https://huggingface.co), sign up, verify etc., 

Generate a Toke for use with Colab [Hugging Face Token](https://huggingface.co/settings/tokens) with Write

Goto [Colab Research Notebooks](https://colab.research.google.com/). Set the `HF_TOKEN` secret in Colab

Join the slack channel #llm-labs

## Next Steps

### Comparing Models

Remember we are using **characters** as tokens, the other GPT models are using a proper tokenizer as shown in [TikTokenizer](https://tiktokenizer.vercel.app/?model=cl100k_base)

| Model   | Release Date | Parameter Count     | Number of Layers               | Hidden Size    | Attention Heads  | Context Window         |
|---------|--------------|---------------------|--------------------------------|---------------|------------------|------------------------|
| MiniGPT | N/A          | ~100K – 1M¹        | ~4 (configurable)              | ~128 (configurable) | 4 (configurable)  | ~64 – 512 tokens²      |
| GPT-2   | Feb 14, 2019 | ~117M – 1.5B³      | 12 – 48 (depending on size)    | 768 – 1600     | 12 – 25           | 1024 tokens            |
| GPT-3   | Jun 11, 2020 | 175B⁴              | 96                             | 12,288         | 96               | 2048 tokens            |
| GPT-4   | Mar 14, 2023 | Not Disclosed⁵ (est. >1T) | Not Disclosed            | Not Disclosed  | Not Disclosed    | 8K or 32K tokens⁶      |

### Improve Model Performance 

Change the hyperparameters and see for yourselves. 

Post your generated text (which you think is reasonable) + your hyperparameters you used 

### Saving, Publishing and Consuming Models

* Simple demo of saving the model we generated to HuggingFace also available in [./miniGPT.ipynb](./miniGPT.ipynb)

### Look at some publicly available model weights + config information

* Use the [`runLlama.ipynb`](./runLlama.ipynb)  file, simple 3 steps to get up and running! Yay!


## References

[Visualize LLM Models] (https://bbycroft.net/llm)
