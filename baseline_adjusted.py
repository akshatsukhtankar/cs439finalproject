import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay

from phrasebank_data import load_financial_phrasebank_dataset

# keep same seed
np.random.seed(42)


dataset = load_financial_phrasebank_dataset(test_size=0.2, seed=42)

train_texts = dataset["train"]["sentence"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["sentence"]
test_labels = dataset["test"]["label"]
standard_stop_words = text.ENGLISH_STOP_WORDS
###updating stop words to keep financial sentiment words impacting direction. model might perform better with more refinement of this list. 
financial_keep_list = {'up', 'down', 'above', 'below', 'over', 'under', 'no', 'not', 'off', 'on'}

custom_stop_words = list(set(standard_stop_words) - financial_keep_list)

vectorizer = TfidfVectorizer(max_features=5000, stop_words=custom_stop_words)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# logreg training
reg = LogisticRegression(max_iter=1000, random_state=42)
reg.fit(X_train, train_labels)

start_time = time.time()
preds = reg.predict(X_test)
pred_probs = reg.predict_proba(X_test)
end_time = time.time()

eval_runtime = end_time - start_time
num_samples = len(test_texts)

eval_results = {
    "eval_accuracy": accuracy_score(test_labels, preds),
    "eval_f1": f1_score(test_labels, preds, average="weighted"),
    "eval_loss": log_loss(test_labels, pred_probs), 
    "eval_runtime": eval_runtime,
    "eval_samples_per_second": num_samples / eval_runtime
}

print("Baseline Evaluation Results:")
print(json.dumps(eval_results, indent=2))

os.makedirs("./results", exist_ok=True)
with open("./results/baseline_adjusted_metrics.json", "w", encoding="utf-8") as f:
    json.dump(eval_results, f, indent=2)

# results
def predict_function(texts):
    X = vectorizer.transform(texts)
    return reg.predict_proba(X)

masker = shap.maskers.Text(tokenizer=r"\W+")
explainer = shap.Explainer(predict_function, masker)

sample_texts = test_texts[:10]
shap_values = explainer(sample_texts)

example_idx = 0
example_probs = predict_function([sample_texts[example_idx]])[0]
example_class = int(np.argmax(example_probs))

example = shap_values[example_idx]

example_values = example.values
if example_values.ndim == 2:
    example_values = example_values[:, example_class]
    
example_base = example.base_values
if isinstance(example_base, (list, np.ndarray)):
    example_base = example_base[example_class]

example_expl = shap.Explanation(
    values=example_values,
    base_values=example_base,
    data=example.data,
    feature_names=example.feature_names,
)

shap.plots.waterfall(example_expl, show=False)
plt.title("Adjusted Baseline (TF-IDF + Logistic Regression + Custom Stop Words)")
plt.savefig("./results/baseline_adjusted_shap_waterfall.png", dpi=300, bbox_inches="tight")
plt.close()

cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format="d")
plt.title("Adjusted Baseline Confusion Matrix")
plt.savefig("./results/baseline_adjusted_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("Baseline adjusted evaluation complete. Metrics and SHAP plot saved to ./results/")