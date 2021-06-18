"""Testing HF pipeline"""

from pprint import pprint
import sys

from transformers import pipeline

task = sys.argv[1]

if task == "sentiment":
    classifier = pipeline("sentiment-analysis")
    result = classifier("I've been waiting for a HuggingFace course my whole life.")
elif task == "classif":
    classifier = pipeline("zero-shot-classification")
    result = classifier(
        "Barack Omaba taught what to do to Google and Facebook.",
        candidate_labels=["education", "politics", "business"],
    )
elif task == "unmask":
    unmasker = pipeline("fill-mask")
    result = unmasker("The meaning of citizenship is <mask>.", top_k=2)
elif task == "ner":
    ner = pipeline("ner", grouped_entities=True)
    result = ner("Franco works for the Commission in Brussels.")
elif task == "qa":
    question_answerer = pipeline("question-answering")
    result = question_answerer(
        question="Where do I work?",
        context="Hello I'm Max, I mostly work for the European Commission in Brussels but also for the ULB as well as UNIGE in Geneva."
    )
elif task == "translate":
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    result = translator("J'ai trouvé un bon avocat, ce sera parfait pour ma salade.")
else:
    result = "Unknown task!"

pprint(result)
