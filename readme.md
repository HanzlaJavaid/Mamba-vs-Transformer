# Transformers VS Mamba Architecture: Report

**Team Members:**
- Muhammad Hanzla Javaid
- Saransh Gupta
- Suman Lamsal

**Github link:** https://github.com/HanzlaJavaid/mamba_transformer_comparison

**Date:** 12/17/2024

## Introduction

The field of sequence modeling has evolved significantly, with the Transformer and Mamba architectures emerging as two key frameworks. While the Transformer (2017) redefined Natural Language Processing (NLP) using the self-attention mechanism, the Mamba architecture (2023) addresses Transformers' efficiency and scalability limitations, particularly for long sequences. This project conducts a comparative analysis of these architectures to assess their strengths, limitations, and performance.

## Transformer Architecture

Introduced by Vaswani et al. in "Attention Is All You Need" (2017), the Transformer architecture replaced recurrent layers with the self-attention mechanism, enabling models to process sequences efficiently and capture long-range dependencies.

### Key Features of Transformer

- Self-Attention Mechanism: Computes contextual relationships across input tokens dynamically
- Parallel Processing: Eliminates sequential bottlenecks by processing entire sequences simultaneously
- Long Range Dependency Capture: Efficiently models relationships across distant tokens
- Scalability: Scales effectively with large datasets, forming the foundation of state of the art models like BERT and GPT

### Applications of Transformers

Transformers have powered advancements in LLMs (Large language models), VLMs (Vision Language Models), Diffusion Models and even in Reinforcement Learning.

## Mamba Architecture

The Mamba architecture, introduced by Gu and Dao (2023), addresses the scalability and efficiency challenges of Transformers. Built upon Structured State Space Models (SSMs), Mamba processes sequences with linear time complexity, making it a promising alternative for long-sequence tasks.

### Key Features of Mamba

- Linear Time Complexity: Processes input sequences efficiently, scaling with input size
- Optimized Hardware Efficiency: Enhances both training and inference on modern hardware
- Selective State Updates: Maintains long-term dependencies without excessive resource overhead
- Extended Context Windows: Handles lengthy sequences, ideal for tasks like document-level NLP and time-series analysis

## Problem Description

With the growing need for efficient sequence modeling, evaluating emerging architectures is crucial. While Transformers have set benchmarks in NLP, their quadratic time complexity restricts scalability with increasing sequence lengths. The Mamba architecture offers a promising alternative with linear complexity, but its real-world performance requires further exploration.

This project aims to compare the scalability, efficiency, and predictive performance of Mamba and Transformer architectures. The central research question is:

"Can Mamba's linear time complexity provide results comparable to or better than Transformers, particularly in large datasets and long input sequences?"

### Relevance and Significance

- Computational Limitations: Transformers' quadratic complexity poses challenges for scalability in resource-limited environments
- Emergence of Mamba: Mamba's linear complexity offers improved efficiency but requires validation against Transformers
- Real-World Applications: Tasks like document classification, time-series analysis, and edge computing demand scalable solutions for extended sequences

### Core Research Objectives

- Accuracy Comparison: Compare predictive accuracy on a binary classification task (human-written vs. AI-generated texts)
- Scalability and Efficiency Analysis: Evaluate performance with increasing sequence lengths
- Strengths and Limitations: Identify scenarios where each architecture excels or struggles, offering actionable insights for practical use cases

## Description of the Data

### Exploratory Data Analysis & Sampling

The dataset used in this project was sourced from Kaggle's competition on AI text detection, specifically designed to classify whether a given text is human-written or AI-generated. It originally contains the following data distribution:

| Category | Count |
|----------|--------|
| Human written texts | 222154 |
| AI written texts | 124823 |
| Total samples | 346977 |

The texts vary significantly in length, ranging from a few words to over 6,000 words, with a skewed distribution peaking at around 2,000 words.

Since the original dataset is imbalanced but contains a rich amount of data, we decided to take a subset of this large dataset that has a balanced number of samples while still maintaining normal distribution in text lengths. To do this, we obtained the following distribution of the dataset:

| Category | Train | Test |
|----------|--------|-------|
| Human written texts | 9985 | 2497 |
| AI written texts | 9986 | 2496 |
| Total samples | 19971 | 4993 |

### Preprocessing

As with any NLP's intersection with Deep Learning, we did preprocessed the dataset with following key steps:

1. **Convert to lowercase**
   - Converts all characters in the text to lowercase to ensure case consistency

2. **Remove punctuation**
   - Removes all punctuation marks from the text using str.maketrans

3. **Remove numbers**
   - Removes numeric characters from the text by filtering out any digits

4. **Tokenize**
   - Splits the text into individual words (tokens) using word_tokenize

5. **Remove stopwords and lemmatize**
   - Filters out stopwords (common words that do not contribute much to meaning, e.g., "the," "and")
   - Applies lemmatization to each remaining word to reduce it to its base or root form

6. **Rejoin tokens into a single string**
   - Combines the processed tokens back into a single string with spaces separating words

## Methodology

We fine tuned a Transformer and MAMBA model for the downstream task of detecting if the input sequence is AI generated or human written. This is a binary classification problem. Once both models were fine tuned, we used a test dataset to get evaluations including confusion matrix and F1 scores. We also ran inference speed tests afterwards.

### Selection of Models

**Transformer:**

| Attribute | Value |
|-----------|--------|
| Model Name | Bert base uncased |
| Hugging Face Repo | https://huggingface.co/google-bert/bert-base-uncased |
| Parameters | 110 Million |
| Pre trained | Yes |

**MAMBA:**

| Attribute | Value |
|-----------|--------|
| Model Name | Mamba-370M |
| Hugging Face Repo | https://huggingface.co/state-spaces/mamba-370m |
| Parameters | 370 Million |
| Pre trained | Yes |

### Fine tuning Details

| Aspect | Details |
|--------|----------|
| Strategy | Changing LM head with a linear layer with output of two neurons along with softmax activation |
| Parameters | Full fine tuning all parameters |
| Epochs | 3 |
| Learning rate | 0.00005 |
| Batch size | 16 |
| GPU | A100 (40gb) |

## Results

### Classification Metrics

Both architectures achieved an F1 score of 0.99, indicating near-perfect classification capability for this task. The confusion matrix of both fine tuned models showed equal level of performance.

### Scalability and Efficiency

To see how the inference time increases with increasing input size, we tested both fine tuned models on a specialized sample of the original dataset. The sample had essays from both classes with increasing word counts from 500 to 5000.

Following are the observations:
- Transformer: Showed slower performance when handling long sequences due to quadratic time complexity
- Mamba: Demonstrated linear scalability, handling longer sequences with same compute time

Given that the pretrained MAMBA model is larger in number of parameters than the transformer and yet it performs fast, the efficiency of MAMBA is evident. With this inference speed, it can be deployed on edge computing devices to empower on device language models.

## Discussion

Our study focused on a binary classification problem of distinguishing between human-written and AI-generated texts, exploring the comparative analysis of both architectures.

### Impact of Model Architecture on Performance

The results highlight the remarkable performance of both Transformer and Mamba architectures. Both models achieved an impressive F1 score of 0.99, demonstrating their ability to classify human written and AI generated texts effectively.

### Scalability and Model Complexity

The most significant distinction emerges in computational efficiency. Despite having a larger parameter count (370 million vs. 110 million), the Mamba model demonstrated substantially faster inference times.

Mamba's linear time complexity presents a significant advancement in sequence modeling. The model scales efficiently with increasing input sizes, addressing a key limitation of traditional Transformer architectures.

This scalability allows for more extensive parameter training that can embed more intelligence into the model while keeping it practical to run.

## Limitations

### Integrations and Maturity

Currently, MAMBA architecture is not well integrated with Pytorch and other deep learning libraries. The reason is it being a new architecture that just got released in 2023. This makes it significantly difficult for the open source deep learning community to actively apply and test how MAMBA can perform for other modalities of data.

### Long Context Accuracy

Several papers suggest that due to MAMBA's state space limitations, it trades off accuracy and generalization to keep linear time complexity. When context size grows, MAMBA tends to lose information that is critical for advanced chatbots.

These losses can constitute of:
- Inability to remember facts from the context
- Inability to learn from in context examples

### Dependence on Nature of Data

MAMBA is as of now, known to work well with discrete modalities of data, like natural language and DNA sequence, however, researchers have noted its performance dropping for data which is continuous, like audio.

## Future Directions

The future for MAMBA seems very promising just as it was for transformers back in 2018. Following are some active areas of research:

- Multimodality in MAMBA
- Parameter scaling to trillions
- Hybrid architectures that combine Transformers and MAMBA

## References

1. V. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention Is All You Need," Advances in Neural Information Processing Systems, vol. 30, pp. 5998-6008, 2017.

2. A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv, 2023. [Online]. Available: https://arxiv.org/abs/2312.00752.

3. https://kaggle.com/competitions/llm-detect-ai-generated-text, 2023. Kaggle.

4. M. Grootendorst, "A Visual Guide to Mamba and State Space Models," Maarten Grootendorst's Newsletter, Feb. 19, 2024. [Online]. Available: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state.

5. R. Waleffe, W. Byeon, D. Riach, V. Korthikanti, T. Dao, A. Gu, A. Hatamizadeh, S. Singh, D. Narayanan, G. Kulshreshtha, V. Singh, J. Casper, J. Kautz, M. Shoeybi, and B. Catanzaro, "An Empirical Study of Mamba-based Language Models," arXiv:2406.07887 [cs.LG], Jun. 2024. Available: https://doi.org/10.48550/arXiv.2406.07887

6. Y. Zou, Y. Chen, Z. Li, L. Zhang, and H. Zhao, "Venturing into Uncharted Waters: The Navigation Compass from Transformer to Mamba," arXiv:2406.16722 [cs.CL], Jun. 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2406.16722.

7. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv:1810.04805 [cs.CL], Oct. 2018. [Online]. Available: https://doi.org/10.48550/arXiv.1810.04805.
