# llm_finetune

A project for fine-tuning Large Language Models (LLMs) such as Gemma using custom datasets.

## Features

- Data loading and preprocessing from Excel, CSV, or JSON
- Gemma tokenizer integration using Hugging Face Transformers
- Dataset splitting for training and evaluation
- Training with LoRA (Low-Rank Adaptation) or base models
- Logging and experiment tracking with MLflow

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd llm_finetune
   ```

2. Install dependencies (recommended: use [Poetry](https://python-poetry.org/)):
   ```bash
   poetry install
   ```
   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage


### 1. Prepare Your Dataset

- Place your data files (Excel, CSV, or JSON) in a directory.
- Use the `LoadDatasetAndSaveDataset` class in `utils/load_dataset.py` to preprocess and chunk your data into pickled files.

#### Data Format Example

Your data should have at least two columns/fields: `prompt` and `answer`.

**Excel/CSV Example:**

| prompt                  | answer                |
|-------------------------|-----------------------|
| Who is the king of ...? | The lion is the king. |
| What is 2+2?            | 2+2=4                 |

**JSON Example:**

[
  { "prompt": "Who is the king of the jungle?", "answer": "The lion is the king." },
  { "prompt": "What is 2+2?", "answer": "2+2=4" }
]

### 2. Split Dataset

- Use `utils/create_test_train_dataset.py` to split your pickled dataset into `train` and `eval` folders.

### 3. Training

- Run the training script:
  ```python
  from trainer import train

  train(
      model_name="google/gemma-3-1b-pt",
      file_paths=["path/to/your/data.xlsx"],
      load_mode="excel",  # or "csv", "json"
      local_path="path/to/output",
      batch_size=1,
      lora_method=True,  # or False for base model
      num_epochs=10,
      train_ratio=0.8,
      chunk_size=30,
      lora_rank=7,
      lora_alpha=16,
      init_lora_weights="gaussian",
      lora_dropout=0.0,
      target_modules=['q_proj', 'v_proj', 'k_proj'],
      learning_rate=1e-6,
      weight_decay=0.001,
      tracking_uri="",
  )
  ```

### 4. Customization

- Modify `models.py` to add or change model architectures.
- Adjust tokenization in `utils/tokenize.py` as needed.

## Project Structure

```
llm_finetune/
├── datasets.py
├── models.py
├── poetry.lock
├── pyproject.toml
├── README.md
├── trainer.py
└── utils/
    ├── create_test_train_dataset.py
    ├── load_dataset.py
    └── tokenize.py
```

## Requirements

- Python 3.12
- torch
- transformers
- sentencepiece
- mlflow
- pydantic
- pandas
- openpyxl
- peft


(See `pyproject.toml` for full list.)

## MLflow Tracking UI

To view experiment results and training logs, run the following command in your project directory:

```bash
mlflow ui
```

This will start the MLflow tracking server. By default, you can access the UI at [http://localhost:5000](http://localhost:5000) in your browser to explore runs, metrics, and artifacts.

## License

MIT