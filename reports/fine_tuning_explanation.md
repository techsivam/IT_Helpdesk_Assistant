# Fine-Tuning Explanation Report

This report explains the concepts, mechanisms, and hyper-parameters involved in the domain-specific fine-tuning process of the IT Helpdesk Assistant.

---

## 1. Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning (PEFT)

### Why Full Fine-Tuning is Expensive
In full fine-tuning, all the parameters of the pre-trained Large Language Model (LLM) are updated during training. This is extremely resource-intensive because:
* **Memory overhead**: The optimizer (like AdamW) needs to store optimizer states (moments, gradients, and parameters) for every single parameter. For a 7-billion parameter model, AdamW requires around 56 GB of GPU memory just for optimizer states, plus memory for weights and activations.
* **Storage cost**: Saving the model outputs require saving full model checkpoint copies (several gigabytes each time), making checkpoint storage and deployment very expensive.
* **Catastrophic forgetting**: Updating all weights might cause the model to lose general knowledge and capabilities it learned during pre-training.

### What LoRA (Low-Rank Adaptation) Does
LoRA addresses this by freezing the original pre-trained model weights and introducing trainable rank decomposition matrices into each layer of the Transformer architecture.
* For a weight update matrix \(\Delta W\), LoRA factors it into two low-rank matrices \(A\) and \(B\):
  \[\Delta W = B \times A\]
  where \(W \in \mathbb{R}^{d \times k}\), \(B \in \mathbb{R}^{d \times r}\), and \(A \in \mathbb{R}^{r \times k}\) with rank \(r \ll \min(d, k)\).
* Only the low-rank matrices \(A\) and \(B\) are updated. This reduces the number of trainable parameters by 99%+, significantly lowering VRAM requirements and training time.

### What QLoRA (Quantized Low-Rank Adaptation) Does
QLoRA takes LoRA further by quantizing the base model to high-fidelity 4-bit precision (specifically using the **NormalFloat4 (NF4)** data type) and performing adaptation over it.
* **NF4 Data Type**: An information-theoretically optimal quantization type for normally distributed weights.
* **Double Quantization (DQ)**: Quantizes the quantization constants, saving an additional 0.37 bits per parameter.
* **Paged Optimizers**: Prevents memory spikes (OOM errors) during gradient checkpoints by paging optimizer states to system RAM during active gradient steps.

### Why QLoRA is Useful on Limited GPUs
QLoRA reduces the memory footprint of a 7B model from ~14GB (in 16-bit) to ~4GB (in 4-bit NF4). This allows GenAI engineers to fine-tune 7B or 8B parameter models on a single commercial GPU (like a Tesla T4 or a laptop RTX 4060/5070 with 8-12GB VRAM) instead of expensive enterprise server GPUs.

---

## 2. Stages of Fine-Tuning

### Stage 1: Non-Instruction Fine-Tuning
* **Definition**: Pre-training style fine-tuning using raw, unstructured domain-specific text documents.
* **Goal**: Teaches the model domain terminology, jargon, writing style, and company-specific background knowledge (next-token prediction objective).
* **Format**: Pure text blocks (e.g. policies, wiki articles, logs).

### Stage 2: Instruction Fine-Tuning (Supervised Fine-Tuning / SFT)
* **Definition**: Fine-tuning the model on instruction-response pairs (Q&A format).
* **Goal**: Teaches the model how to act as a conversational assistant, understand user instructions, format outputs (Markdown, bullet points), and answer questions based on the domain knowledge learned in Stage 1.
* **Format**: `{"instruction": "...", "response": "..."}`.

### Stage 3: Direct Preference Optimization (DPO) / ORPO Alignment
* **Definition**: An optimization method that directly aligns the language model with human preferences by training it on pairs of "chosen" and "rejected" responses.
* **Goal**: Refines response style, tone, correctness, safety, and conciseness, teaching the model to avoid generic or unhelpful answers.
* **Format**: `{"prompt": "...", "chosen": "...", "rejected": "..."}`.
* **DPO vs. SFT**: While SFT teaches the model *how* to answer, DPO teaches the model *which* answer is preferred and guides its tone, style, and safety constraints. DPO optimizes a closed-form loss function directly on the relative probability of chosen vs. rejected outputs without needing a separate reward model (unlike RLHF).

---

## 3. Hyper-Parameters Used in This Project

Below are the hyper-parameters configured in the notebooks:

* **Rank (\(r\)) = 16**: The rank of the low-rank adaptation matrices. A rank of 16 provides a good balance between learning capacity and parameter efficiency.
* **Lora Alpha (\(\alpha\)) = 16**: Scaling factor for the LoRA adapters. The weights are scaled by \(\frac{\alpha}{r}\). Setting \(\alpha = r\) (ratio of 1.0) is a standard stable configuration.
* **Lora Dropout = 0**: LoRA dropout rate. Set to 0 in Unsloth for maximum training speed and efficiency.
* **Learning Rate**:
  * **SFT / Non-Instruction**: `2e-4` (Standard learning rate for adaptation stages).
  * **DPO Alignment**: `5e-6` (A much smaller learning rate is used for DPO to avoid destabilizing the SFT-trained model and preventing catastrophic policy shifts).
* **Batch Size**:
  * **Per-device batch size**: `2`
  * **Gradient Accumulation Steps**: `4`
  * **Effective Batch Size**: `2 * 4 = 8` (Saves VRAM while maintaining stable gradient updates).
* **Optimizer**: `adamw_8bit` (Reduces memory consumption of optimizer states by using 8-bit quantization).
* **Weight Decay**: `0.01` (Prevents overfitting during training).
* **LR Scheduler**: `linear` for SFT/Non-Instruction, `cosine` for DPO.
