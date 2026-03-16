# Movie Title Generator using FLAN-T5 Base

An NLP project that generates creative and contextual movie titles using the FLAN-T5 base model, leveraging transfer learning and advanced prompt engineering for creative text generation.

## 📋 Overview

This project implements a movie title generation system using FLAN-T5 base. It takes movie plots, genres, themes, or other attributes and generates relevant, creative movie titles using state-of-the-art transformer models and fine-tuning techniques.

## 🎯 Objectives

- Generate creative movie titles from plot descriptions
- Leverage FLAN-T5 base model capabilities
- Demonstrate fine-tuning on domain data
- Handle various movie genres and themes
- Evaluate title creativity and relevance
- Support batch processing for multiple plots

## 🗂️ Project Structure

```
Movie_title_generator_using_flanT5_base/
├── Movie_title_generator_using_flanT5_base.ipynb  # Main notebook
├── README.md                                      # Project documentation
└── data/                                          # Datasets
    ├── movie_plots.csv  (optional)
    └── movie_titles.csv (optional)
```

## 🛠️ Technologies & Libraries

- **Model**: FLAN-T5 Base (Hugging Face)
- **Deep Learning**: PyTorch, Transformers
- **NLP Tools**: Tokenizers, Seq2Seq utilities
- **Data Processing**: Pandas, NumPy
- **Evaluation**: BLEU, ROUGE, BERTScore
- **Utilities**: tqdm, collections, JSON

## 📊 Key Features

- **FLAN-T5 Integration**:
  - Instruction-following capabilities
  - Multi-task learning support
  - Few-shot learning
  - Fine-tuning flexibility

- **Title Generation**:
  - Diverse title options
  - Genre-aware generation
  - Tone-specific titles
  - Multi-word sequences

- **Input Processing**:
  - Plot summaries
  - Genre information
  - Thematic elements
  - Budget/Rating constraints
  - Actor names (optional)

- **Generation Strategies**:
  - Greedy decoding
  - Beam search
  - Diversity sampling
  - Temperature control

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- GPU with 6GB+ VRAM recommended
- Internet for downloading models

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Movie_title_generator_using_flanT5_base
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download model weights:
   ```bash
   python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
   AutoTokenizer.from_pretrained('google/flan-t5-base'); \
   AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')"
   ```

4. Launch notebook:
   ```bash
   jupyter notebook Movie_title_generator_using_flanT5_base.ipynb
   ```

## 📈 Model Pipeline

```
Movie Plot + Metadata
         ↓
Prompt Engineering
         ↓
Tokenization
         ↓
FLAN-T5 Encoder/Decoder
         ↓
Title Generation
         ↓
Post-processing
         ↓
Output Ranking
```

## 🔧 Architecture Details

### Input Prompt Construction
```python
# Example prompts for different contexts
prompts = {
    'basic': "Generate a movie title: {plot}",
    'genre': "Generate a {genre} movie title: {plot}",
    'tone': "Generate a {tone} movie title for: {plot}",
    'detailed': "Based on: {plot}, Genre: {genre}, Tone: {tone}. Generate title:"
}
```

### Model Configuration
```python
MODEL_CONFIG = {
    'model_name': 'google/flan-t5-base',
    'source_max_length': 512,
    'target_max_length': 50,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 3,
    'warmup_steps': 500,
    'weight_decay': 0.01
}
```

## 💾 Training & Inference

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
import torch

# Load pretrained model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Fine-tune on movie data
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    save_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

## 📝 Usage Examples

### Single Plot Title Generation

```python
from title_generator import MovieTitleGenerator

# Initialize
generator = MovieTitleGenerator(
    model_name="google/flan-t5-base",
    device="cuda"
)

# Generate title
plot = """A young programmer discovers a hidden message in the stock market 
and uncovers a government conspiracy."""

title = generator.generate(
    prompt=f"Generate a thriller movie title: {plot}",
    num_beams=4,
    temperature=0.8
)

print(f"Generated Title: {title}")
```

### Batch Processing

```python
import pandas as pd

# Load dataset
movies_df = pd.read_csv('movie_plots.csv')

# Generate titles for all plots
generated_titles = []
for plot in movies_df['plot'].head(100):
    prompt = f"Generate a movie title: {plot}"
    title = generator.generate(prompt)
    generated_titles.append(title)

# Save results
results_df = movies_df[['plot']].assign(generated_title=generated_titles)
results_df.to_csv('generated_titles.csv', index=False)
```

### Genre-Specific Generation

```python
genres_prompts = {
    'thriller': "Generate a suspenseful thriller title: {plot}",
    'comedy': "Generate a funny comedy title: {plot}",
    'romance': "Generate a romantic movie title: {plot}",
    'action': "Generate an action-packed movie title: {plot}",
    'drama': "Generate a compelling drama title: {plot}"
}

plot = "A love story set in post-war Europe"
genre = "romance"

prompt = genres_prompts['romance'].format(plot=plot)
title = generator.generate(prompt)
print(f"Romance Title: {title}")
```

### Advanced Generation with Constraints

```python
# Generate multiple title options
def generate_title_options(plot, n_options=5):
    titles = []
    for temp in [0.5, 0.7, 0.9, 1.0, 1.2]:
        title = generator.generate(
            prompt=f"Generate a creative movie title: {plot}",
            temperature=temp,
            top_p=0.95
        )
        titles.append(title)
    
    return titles

plot = "A detective uncovers mysteries in a small town"
options = generate_title_options(plot)
for i, title in enumerate(options, 1):
    print(f"{i}. {title}")
```

## ⚙️ Hyperparameters

```python
# Generation parameters
GENERATION_CONFIG = {
    'temperature': 0.7,      # 0=deterministic, >1=creative
    'top_k': 50,            # Top-k sampling
    'top_p': 0.95,          # Nucleus sampling
    'num_beams': 4,         # Beam search width
    'max_length': 50,       # Max title length
    'early_stopping': True,
    'repetition_penalty': 2.0,
    'length_penalty': 2.0
}
```

## 📊 Evaluation

```python
from evaluate import evaluate_titles

# Evaluate against reference titles
evaluation_metrics = evaluate_titles(
    generated_titles=generated_titles,
    reference_titles=reference_titles,
    metrics=['bleu', 'rouge', 'bertscore']
)

print(evaluation_metrics)
```

## 🎯 Example Outputs

| Plot | Generated Title |
|------|-----------------|
| "A spy infiltrates a terrorist organization" | "Agent Infiltration" |
| "Two friends start a tech startup" | "The Silicon Dream" |
| "A woman searches for her missing sister" | "Sister's Secret" |
| "An astronaut gets stranded on Mars" | "Red Planet Escape" |

## 🔍 Advanced Techniques

- **Few-shot Learning**: Provide title examples
- **Prompt Templates**: Optimize for different genres
- **Ensemble Methods**: Combine multiple generations
- **Ranking**: Score and rank titles
- **Filtering**: Remove duplicates/inappropriate content

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## 📚 References

- [Scaling Instruction-Finetuned Language Models (FLAN-T5)](https://arxiv.org/abs/2210.11416)
- [Exploring the Limits of Transfer Learning (T5)](https://arxiv.org/abs/1910.10683)
- [Hugging Face Documentation](https://huggingface.co/docs/transformers/)
- [Attention Mechanisms in Transformers](https://arxiv.org/abs/1706.03762)

## 📄 License

This project is open source and available under the MIT License.

## 💡 Tips for Better Results

- Provide detailed plot descriptions
- Include genre information
- Use temperature tuning for creativity
- Ensemble multiple generations
- Post-process with ranking/filtering
- Fine-tune on domain-specific data

## ✉️ Contact

For questions or suggestions, please open an issue or contact the project maintainers.