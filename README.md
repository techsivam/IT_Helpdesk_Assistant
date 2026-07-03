# Domain-Specific AI Assistant: IT Helpdesk Assistant

This repository contains the complete implementation for building a domain-specific **IT Helpdesk AI Assistant** by fine-tuning the open-source **Qwen2.5-1.5B** model. The project goes through three training stages: **Non-Instruction Domain Fine-Tuning**, **Supervised Instruction Fine-Tuning (SFT)**, and **Direct Preference Optimization (DPO)** alignment using the **Unsloth** library for GPU acceleration.

---

## Project Overview

* **Domain Selected**: IT Helpdesk Assistant
* **Business Problem**: Modern companies spend a significant amount of time resolving routine internal IT queries (e.g. password resets, VPN setups, guest Wi-Fi access, hardware requests). The goal is to build an internal AI assistant that understands corporate IT policies, follows strict security guidelines (directing users to official channels instead of risky manual downloads), and responds in a polite, helpful, and concise manner.
* **Base Model**: [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) (quantized to 4-bit NF4 via Unsloth)

---

## Directory Structure

```
domain-ai-assistant-finetuning/
│
├── data/
│   ├── non_instruction_data.txt      # 60 paragraphs of IT policies & raw ticket bodies
│   ├── instruction_dataset.jsonl     # 150 cleaned instruction-response examples
│   └── preference_dataset.jsonl      # 75 DPO prompt-chosen-rejected preference triplets
│
├── notebooks/
│   ├── non_instruction_finetuning.ipynb  # Stage 1: Next-token pre-training notebook
│   ├── instruction_finetuning.ipynb      # Stage 2: Supervised Fine-Tuning notebook
│   └── dpo_alignment.ipynb               # Stage 3: DPO preference alignment notebook
│
├── reports/
│   ├── base_model_evaluation.md      # Testing the base model on 10 IT queries
│   ├── sft_model_comparison.md       # Comparing Base Model vs SFT Model
│   ├── final_evaluation.md           # Comparing Base vs SFT vs DPO Model
│   └── fine_tuning_explanation.md    # Conceptual explanation of LoRA, QLoRA, SFT, DPO
│
├── src/
│   └── inference.py                  # Local GPU/CPU CLI inference script
│
├── README.md
└── requirements.txt
```

---

## Datasets and Preparation

The datasets were processed and cleaned from the Hugging Face `Bitext-customer-support-llm-chatbot-training-dataset` (focusing on `IT Support` and `Technical Support` queues):

1. **Non-Instruction Data (`data/non_instruction_data.txt`)**: Contains 60 paragraphs of text. Half consists of detailed, synthesized corporate IT policies (e.g., password complexities, VPN setup policies, hardware replacement cycles, printer mappings, offboarding timelines) and the other half consists of real cleaned customer technical support ticket bodies.
2. **Instruction Data (`data/instruction_dataset.jsonl`)**: 150 high-quality, cleaned IT support ticket Q&As formatted as `{"instruction": "...", "response": "..."}`.
3. **Preference Data (`data/preference_dataset.jsonl`)**: 75 triplets formatted as `{"prompt": "...", "chosen": "...", "rejected": "..."}` where the `chosen` response is the detailed, policy-compliant IT resolution, and the `rejected` response represents typical unhelpful advice (e.g., "restart your PC," "ask Google," or "this is not supported").

---

## Fine-Tuning Methodology

We apply parameter-efficient methods to fine-tune the model:

### LoRA / QLoRA Configuration
* **Rank (\(r\))**: `16`
* **Alpha (\(\alpha\))**: `16`
* **Dropout**: `0` (Optimized for speed in Unsloth)
* **Target Modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
* **Quantization**: 4-bit NF4 (NormalFloat4) to fit the model within small GPUs (VRAM < 6GB).

### Stage 1: Non-Instruction Domain Fine-Tuning
* **Objective**: Teach the base model the specific terminology and policies of the IT department (next-token prediction).
* **Optimizer**: `adamw_8bit` with a learning rate of `2e-4`.

### Stage 2: Supervised Fine-Tuning (SFT)
* **Objective**: Train the model to structure its outputs as a conversational assistant.
* **Format**: Qwen Chat Template.
* **Optimizer**: `adamw_8bit` with a learning rate of `2e-4`.

### Stage 3: Direct Preference Optimization (DPO)
* **Objective**: Refine the model's tone, ensuring it avoids dismissive answers and consistently chooses professional, safe, and concise advice.
* **Optimizer**: `adamw_8bit` with a learning rate of `5e-6` (scaled down to prevent catastrophic forgetting).

---

## Before vs. After Output Comparison

| Query | Base Model (Before) | Fine-Tuned DPO Model (After) |
|---|---|---|
| **Software Request** | You can download IntelliJ IDEA from JetBrains website. Run the installer and follow the instructions to set it up. | Please open Software Center (Windows) or Self Service (macOS) on your device to install IntelliJ IDEA. Manual downloads from external sites violate security policies. If you need a license key, submit a Software License Request ticket on the IT portal. |
| **Suspicious Email** | If you think it is fake, you can delete it or ask your colleagues if they got it too. Don't click on any link. | Do not click any links or open attachments. Report the email immediately using the 'Report Phishing' button in Outlook. If you entered credentials, change your password immediately at https://helpdesk.company.local and call the Helpdesk (+1-800-555-0199) or SOC. |

---

## Getting Started

### Quick Start (Local Inference on Windows/macOS/Linux)

To set up your virtual environment and run local queries immediately, execute the following commands in your terminal:

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (CMD):
.\.venv\Scripts\activate.bat
# On macOS/Linux:
source .venv/bin/activate

# 3. Install the standard requirements
pip install -r requirements.txt

# 4. Query the assistant (make sure to close the double quotes!)
python src/inference.py --question "How do I reset my expired password?"
```

---

### 1. Detailed Installation & Setup

This project separates **local inference** (which runs on CPU/GPU via standard Transformers) from **model training** (which requires Unsloth, Linux/WSL, and a CUDA GPU).

#### For Local Inference & CLI Usage:
You do **NOT** need to install `unsloth` or `xformers` locally. The standard libraries in `requirements.txt` are sufficient and fully support loading the trained LoRA adapters.

1. **Create the virtual environment**:
   ```bash
   python -m venv .venv
   ```
2. **Activate the environment**:
   * **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **Windows (CMD)**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
   * **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```
3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

#### For GPU-Accelerated Fine-Tuning (Google Colab / Linux WSL):
Training notebooks are designed to run in a GPU-accelerated cloud environment (like Google Colab) or inside Linux WSL. Unsloth requires a Linux-based CUDA environment to function properly.

In your GPU training environment, install **Unsloth**:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers" "trl" peft accelerate bitsandbytes
```

> [!IMPORTANT]
> **Windows Installation Note (Wheel Build Errors)**:
> If you try to run `pip install --no-deps xformers` natively on Windows (especially with Python 3.13), you will encounter a `ModuleNotFoundError: No module named 'torch'` or build errors. This happens because `xformers` does not distribute pre-compiled binary wheels for Windows on newer Python versions, forcing pip to compile from source (which requires MSVC and local CUDA Toolkits). 
> **To resolve this, run training in Google Colab or inside WSL (Ubuntu), and use the local Windows environment only for running inference.**

---

### 2. Model Training & Notebook Execution

The training workflow is split into three sequential Jupyter notebooks located in the `notebooks/` directory.

#### Option A: Running on Google Colab (Recommended)
This is the easiest way to access GPU acceleration (such as the free T4 GPU runtime) and avoid local system compilation errors:
1. **Upload the Notebooks & Datasets**:
   * Upload the `notebooks/` and `data/` directories to your Google Drive.
2. **Open in Google Colab**:
   * Open Colab (https://colab.research.google.com) and load each notebook from Google Drive or upload them directly.
3. **Change Runtime to GPU**:
   * Click **Runtime** -> **Change runtime type** -> select **T4 GPU** (or A100/L4 if available).
4. **Mount Drive & Upload Data**:
   * Ensure that the datasets (`data/non_instruction_data.txt`, `data/instruction_dataset.jsonl`, `data/preference_dataset.jsonl`) are uploaded to the local runtime workspace folder under a `/content/data/` directory.
5. **Run Sequentially**:
   * Execute the cells in each notebook from top to bottom.

#### Option B: Running Locally (Linux / Windows WSL with GPU)
If you have a local NVIDIA GPU configured with CUDA on Linux (or Windows WSL):
1. **Activate the Virtual Environment**:
   ```bash
   # Activate your virtual environment
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   source .venv/bin/activate      # Linux/macOS
   ```
2. **Install JupyterLab**:
   ```bash
   pip install jupyterlab
   ```
3. **Start JupyterLab**:
   ```bash
   jupyter lab
   ```
4. **Execute Notebooks Sequentially**:
   Open and execute the notebooks in the following order:
   * **Stage 1**: `notebooks/non_instruction_finetuning.ipynb` (outputs `adapters/non_instruction_lora`)
   * **Stage 2**: `notebooks/instruction_finetuning.ipynb` (outputs `adapters/sft_lora`)
   * **Stage 3**: `notebooks/dpo_alignment.ipynb` (outputs `adapters/dpo_lora`)

---

### 3. Local Inference CLI

Once you have trained your adapter models (or want to test loading), run the interactive CLI tool:
```bash
# Start interactive shell (automatically falls back to CPU if no GPU/Unsloth is found)
python src/inference.py
```

---

## Key Observations and Challenges

### Final Observations
* **Base Model**: Possesses general syntax skills but is useless for proprietary company operations, recommending insecure workarounds (e.g. downloading software directly or asking colleagues).
* **SFT Model**: Follows Q&A formats and policy URLs well, but is prone to writing verbose paragraphs and repeating generic boilerplate warnings.
* **DPO-Aligned Model**: Delivers responses that are 15-20% shorter, action-oriented, and prioritize corporate security checks.

### Challenges Faced
* **GPU Memory**: Fine-tuning even a small 1.5B model requires significant VRAM unless 4-bit quantization (QLoRA) is used.
* **Unsloth Platform Restrictions**: Unsloth is optimized for Linux. Local Windows setups should use WSL (Ubuntu) or GPU notebook environments (Google Colab, Kaggle).

### Future Improvements
1. **RAG Integration**: Couple the fine-tuned model with a Retrieval-Augmented Generation (RAG) pipeline to fetch real-time server outage statuses.
2. **Multi-Turn Chat DPO**: Expand DPO dataset to handle conversational context drift across multiple turns.
