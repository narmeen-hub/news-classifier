# News Topic Classifier using DistilBERT

## Objective
Fine-tune a transformer model to classify news headlines into World, Sports, Business, or Sci/Tech.

## Methodology
- **Dataset**: AG News (5,000 training / 1,000 test samples)
- **Model**: `distilbert-base-uncased` fine-tuned for sequence classification
- **Training**: 1 epoch, batch size 16, Hugging Face Trainer
- **Evaluation metrics**: Accuracy (89.5%), Weighted F1 (89.4%)

## Deployment
Live demo on Hugging Face Spaces:  
[https://huggingface.co/spaces/narmeenbilal/news-classifier-demo](https://huggingface.co/spaces/narmeenbilal/news-classifier-demo)

## Repository Structure
- `train_notebook.ipynb` – training code (Google Colab)
- `app.py` – Gradio app script
- `requirements.txt` – Python dependencies

## How to run locally
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run `python app.py`
4. Open the local URL
