import json
import os
import torch
import evaluate
import numpy as np
import shap
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
from peft import get_peft_model, LoraConfig, TaskType
from phrasebank_data import load_financial_phrasebank_dataset

set_seed(42)


dataset = load_financial_phrasebank_dataset(test_size=0.2, seed=42)

model_checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# peft
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, 
    num_labels=3
)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, 
    inference_mode=False, 
    r=16, 
    lora_alpha=32, 
    lora_dropout=0.1,
    target_modules=["query", "value"] 
)

peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()

# setup for metrics 
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def metrics(eval_pred):
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    accuracy = float(acc["accuracy"]) if acc is not None else 0.0
    f1_score = float(f1["f1"]) if f1 is not None else 0.0
    return {"accuracy": accuracy, "f1": f1_score}

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=8,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=metrics,
)


trainer.train()
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

os.makedirs("./results", exist_ok=True)
with open("./results/peft_metrics.json", "w", encoding="utf-8") as f:
    json.dump(eval_results, f, indent=2)

# results
def pred(texts):
    inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True).to(peft_model.device)
    with torch.no_grad():
        outputs = peft_model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return scores.cpu().numpy()

explainer = shap.Explainer(pred, tokenizer)
sample_texts = dataset["test"]["sentence"][:5]
shap_values = explainer(sample_texts)

example_idx = 0
example_probs = pred([sample_texts[example_idx]])[0]
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
shap_viz = shap.plots.text(example_expl, display=False)
if hasattr(shap_viz, "data"):
    shap_html = shap_viz.data
elif hasattr(shap_viz, "_repr_html_"):
    shap_html = shap_viz._repr_html_()
else:
    shap_html = str(shap_viz)

shap_html = (
    "<div style=\"font-family: Helvetica, Arial, sans-serif; font-size: 18px;"
    " font-weight: 600; margin-bottom: 12px;\">"
    "PEFT (LoRA RoBERTa)\</div>"
    + shap_html
)

with open("./results/peft_shap.html", "w", encoding="utf-8") as f:
    f.write(shap_html)

shap.plots.waterfall(example_expl, show=False)
plt.title("PEFT (LoRA RoBERTa)")
plt.savefig("./results/peft_shap_waterfall.png", dpi=300, bbox_inches="tight")
plt.close()