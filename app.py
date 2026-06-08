import gradio as gr
from transformers import pipeline

# Use your model ID from the Hub
MODEL_ID = "narmeenbilal/news-classifier-distilbert"   # <-- CHANGE THIS

# Load the model and tokenizer from Hub
classifier = pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID)

def predict(text):
    result = classifier(text)[0]
    # result['label'] is like "LABEL_0", "LABEL_1" etc.
    # Map to actual category names
    label_map = {"LABEL_0": "World", "LABEL_1": "Sports", "LABEL_2": "Business", "LABEL_3": "Sci/Tech"}
    return label_map[result['label']]

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=2, placeholder="Enter a news headline...", label="Headline"),
    outputs=gr.Label(label="Predicted Category"),
    title="📰 News Topic Classifier",
    description="Fine-tuned DistilBERT on AG News dataset. Predicts: World, Sports, Business, Sci/Tech.",
    examples=[
        ["SpaceX launches new rocket"],
        ["Manchester United wins the cup"],
        ["Stock market hits record high"],
        ["New AI model beats human tests"]
    ]
)

iface.launch()
