from langsmith import Client
from langsmith.evaluation import evaluate
from sklearn.metrics import classification_report, confusion_matrix
from langsmith import Client
from dotenv import load_dotenv
from query_classification import classify_query
import os

load_dotenv()
client = Client()

dataset = [
    # --- Category 1: Topic Search ---
    {"query": "Show me recent advancements in graph neural networks.", "label": "topic_search"},
    {"query": "What’s new in multimodal models combining text and video?", "label": "topic_search"},
    {"query": "Find me papers on federated learning in healthcare.", "label": "topic_search"},
    {"query": "Tell me about current research in quantum machine learning.", "label": "topic_search"},
    {"query": "Are there new benchmarks for large language models in 2025?", "label": "topic_search"},
    {"query": "List examples of generative models used in drug discovery.", "label": "topic_search"},
    {"query": "What are the latest papers on vision-language pretraining?", "label": "topic_search"},
    {"query": "Show me recent trends in 3D scene reconstruction with deep learning.", "label": "topic_search"},
    {"query": "What’s happening in research on foundation models for robotics?", "label": "topic_search"},
    {"query": "Find new work on reinforcement learning with human feedback.", "label": "topic_search"},
    {"query": "Show me applications of diffusion models in medical imaging.", "label": "topic_search"},
    {"query": "What are current topics in AI fairness and bias mitigation?", "label": "topic_search"},
    {"query": "List recent survey papers on graph representation learning.", "label": "topic_search"},
    {"query": "What is being researched in neurosymbolic AI today?", "label": "topic_search"},
    {"query": "Show me research directions in efficient transformers.", "label": "topic_search"},
    {"query": "Give me examples of zero-shot transfer in NLP models.", "label": "topic_search"},
    {"query": "What are the most cited works in video understanding since 2023?", "label": "topic_search"},
    {"query": "Tell me about new architectures in speech-to-speech translation.", "label": "topic_search"},
    {"query": "Find studies on AI for climate modeling and weather prediction.", "label": "topic_search"},
    {"query": "Show me models applied in protein structure prediction.", "label": "topic_search"},

    # --- Category 2: Summarization _ Explanation ---
    {"query": "Summarize the contributions of the AlphaGo Zero paper.", "label": "summarization_explanation"},
    {"query": "Break down the experimental results in the CLIP paper.", "label": "summarization_explanation"},
    {"query": "Summarize the limitations discussed in the T5 paper.", "label": "summarization_explanation"},
    {"query": "Explain the methodology section of the DINO self-supervised learning paper.",
     "label": "summarization_explanation"},
    {"query": "Summarize the experiments of the DreamFusion paper.", "label": "summarization_explanation"},
    {"query": "Summarize the related work section of the PaLM paper.", "label": "summarization_explanation"},
    {"query": "Summarize the appendix of the Stable Diffusion paper.", "label": "summarization_explanation"},
    {"query": "Explain the ablation study results in the ViT (Vision Transformer) paper.",
     "label": "summarization_explanation"},
    {"query": "Summarize the evaluation section of the DeepMind’s Gato paper.", "label": "summarization_explanation"},
    {"query": "What are the main conclusions of the Whisper speech recognition paper?",
     "label": "summarization_explanation"},
    {"query": "Summarize the experiments in the SAM (Segment Anything Model) paper.",
     "label": "summarization_explanation"},
    {"query": "Summarize the results from the LLaMA paper.", "label": "summarization_explanation"},
    {"query": "Summarize the contributions of the DeepLab segmentation paper.", "label": "summarization_explanation"},
    {"query": "What are the findings in the RLHF section of the InstructGPT paper?",
     "label": "summarization_explanation"},
    {"query": "Summarize the challenges highlighted in the AlphaStar paper.", "label": "summarization_explanation"},
    {"query": "Summarize the case studies in the DeepMind WaveNet paper.", "label": "summarization_explanation"},
    {"query": "Summarize the scalability analysis in the Chinchilla paper.", "label": "summarization_explanation"},
    {"query": "Summarize the introduction and motivation of the Diffusion Models paper.",
     "label": "summarization_explanation"},
    {"query": "Summarize the evaluation metrics used in the DETR object detection paper.",
     "label": "summarization_explanation"},
    {"query": "Summarize the key experiments in the Flamingo multimodal model paper.",
     "label": "summarization_explanation"},

    # --- Category 3: Out of Scope ---
    {"query": "What time is it in Tokyo right now?", "label": "out_of_scope"},
    {"query": "Can you help me write an email to my boss?", "label": "out_of_scope"},
    {"query": "Translate “hello” into Spanish.", "label": "out_of_scope"},
    {"query": "Who won the Oscars in 2024?", "label": "out_of_scope"},
    {"query": "Can you solve this math equation: 124 + 589?", "label": "out_of_scope"},
    {"query": "What’s the capital of Brazil?", "label": "out_of_scope"},
    {"query": "Write me a short story about a dragon and a robot.", "label": "out_of_scope"},
    {"query": "Can you give me cooking tips for pasta?", "label": "out_of_scope"},
    {"query": "Who is the current president of Egypt?", "label": "out_of_scope"},
    {"query": "Recommend me a good laptop under $1000.", "label": "out_of_scope"},
    {"query": "Can you create a travel itinerary for Italy?", "label": "out_of_scope"},
    {"query": "What’s the latest Marvel movie?", "label": "out_of_scope"},
    {"query": "Tell me a joke about programmers.", "label": "out_of_scope"},
    {"query": "How can I lose weight quickly?", "label": "out_of_scope"},
    {"query": "What’s the price of Bitcoin right now?", "label": "out_of_scope"},
    {"query": "Generate HTML code for a personal website.", "label": "out_of_scope"},
    {"query": "What’s the score of the NBA game tonight?", "label": "out_of_scope"},
    {"query": "Can you recommend me good restaurants in Paris?", "label": "out_of_scope"},
    {"query": "Who sings the song “Blinding Lights”?", "label": "out_of_scope"},
    {"query": "Give me fashion advice for a wedding.", "label": "out_of_scope"},
]

y_true = [item['label'] for item in dataset]
y_pred = []

for item in dataset:
    pred = classify_query({'question': item['query']})
    y_pred.append(pred['category'])

print(set(y_pred))
print(set(y_true))
report = classification_report(y_true, y_pred, digits=3)
conf_matrix = confusion_matrix(y_true, y_pred)
print(report)
print(conf_matrix)

client.create_run(
    project_name='research-paper-testing',
    name="aggregate-metrics",
    inputs={},
    outputs={
        "classification_report": report,
        "confusion_matrix": conf_matrix
    },
    run_type="tool",
    api_key=os.getenv('LANGSMITH_API_KEY')
)
