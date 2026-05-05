from __future__ import annotations

from typing import Dict

from datasets import DatasetDict, load_dataset
import pandas as pd


LABEL_MAP: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def load_financial_phrasebank_dataset(
    test_size: float = 0.2,
    seed: int = 42,
    config: str = "sentences_75agree",
) -> DatasetDict:
    dataset = load_dataset("financial_phrasebank", config, trust_remote_code=True)
    base_split = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    return DatasetDict({"train": base_split["train"], "test": base_split["test"]})


def load_financial_phrasebank_dataframe(
    config: str = "sentences_75agree",
) -> pd.DataFrame:
    dataset = load_dataset("financial_phrasebank", config, trust_remote_code=True)
    df = dataset["train"].to_pandas()
    df["label_text"] = df["label"].map(LABEL_MAP)
    return df
