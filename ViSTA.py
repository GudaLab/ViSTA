import os
import csv
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification
from sklearn.metrics import confusion_matrix
from collections import Counter
import pandas as pd

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

logging.basicConfig(
    stream=sys.stdout,
    format='%(message)s',
    level=logging.INFO
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    tokenizer_name: Optional[str] = field(default=None)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default="query,value")


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    kmer: int = field(default=-1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    eval_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: Optional[str] = field(default="eval_accuracy")
    greater_is_better: Optional[bool] = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)


class WeightedBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def generate_kmer_str(sequence: str, k: int) -> str:
    """
    Converts a DNA sequence into space-separated k-mers.
    
    Args:
        sequence (str): DNA sequence (e.g., "ACGTACGT")
        k (int): k-mer size (e.g., 6)
        
    Returns:
        str: space-separated k-mers (e.g., "ACGTAC CGTACG GTACGT")
    """
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


class SupervisedDataset(Dataset):
    

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str = None,
                 kmer: int = -1, data: List[Tuple[str, int]] = None):
        if data_path:
            with open(data_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                data = [(row[0], int(row[1])) for row in reader]

        texts, labels = zip(*data)

        if kmer != -1:
            logging.warning(f"Using {kmer}-mer as input...")
            texts = [generate_kmer_str(text, kmer) for text in texts]

        output = tokenizer(
            list(texts), return_tensors="pt", padding="longest",
            max_length=tokenizer.model_max_length, truncation=True
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.Tensor(labels).long()
        return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):

    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    valid_mask = labels != -100

    if predictions.shape != labels.shape:
        raise ValueError(f"Shape mismatch: predictions shape {predictions.shape}, labels shape {labels.shape}")

    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="macro"),
        "weighted_f1": sklearn.metrics.f1_score(valid_labels, valid_predictions, average="weighted"),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(valid_labels, valid_predictions),
        "precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="macro"),
        "weighted_precision": sklearn.metrics.precision_score(valid_labels, valid_predictions, average="weighted"),
        "recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="macro"),
        "weighted_recall": sklearn.metrics.recall_score(valid_labels, valid_predictions, average="weighted"),
    }    


def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.argmax(logits, dim=-1)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1) if logits.ndim > 1 else np.rint(logits).astype(int)
    return calculate_metric_with_sklearn(predictions, labels)


def reverse_complement(seq: str) -> str:
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

## This function is for balancing the data by reverse complementing DNA sequence.
def AugmentWithReverseComplements(df: pd.DataFrame, sequence_col='sequence', label_col='label', random_state=42) -> pd.DataFrame:
    """
    Augments the input DataFrame by adding reverse complement sequences for underrepresented classes.
    
    Args:
        df (pd.DataFrame): Input dataset with at least 'sequence' and 'label' columns.
        sequence_col (str): Name of the column containing sequences.
        label_col (str): Name of the column containing class labels.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Augmented dataset with balanced classes.
    """
    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()
    augmented = []

    print("\n Reverse Complement Augmentation Summary:")
    for label, count in class_counts.items():
        df_class = df[df[label_col] == label]
        augmented.append(df_class)

        if count < max_count:
            # Number of additional samples needed
            needed = max_count - count

            # Sample from current class (with replacement)
            sampled = df_class.sample(n=needed, replace=True, random_state=random_state)
            sampled_aug = sampled.copy()
            sampled_aug[sequence_col] = sampled[sequence_col].apply(reverse_complement)

            augmented.append(sampled_aug)
            print(f"  Class {label}: added {needed} reverse complement sequences.")

        else:
            print(f"  Class {label}: already has max count ({max_count}), no augmentation needed.")

    balanced_df = pd.concat(augmented).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"\n Total samples after augmentation: {len(balanced_df)}")
    return balanced_df



import torch
import torch.nn.functional as F
import numpy as np
import os

def save_logits_and_labels(trainer, dataset, dataset_type, num_classes, results_path):
    predictions_output = trainer.predict(dataset)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    logits_tensor = torch.tensor(logits)

    if logits_tensor.ndim == 1:
        # Unexpected for multi-class classification
        raise ValueError(f"Logits are 1D but you have {num_classes} classes. Check model output.")
    elif logits_tensor.ndim == 2:
        if logits_tensor.shape[1] == 1 and num_classes > 2:
            raise ValueError(f"Logits have shape (N, 1) but expected (N, {num_classes}) for multi-class classification.")
        if num_classes == 2:
            #Binary classification with logits for 1 class use sigmoid
            p1 = torch.sigmoid(logits_tensor).squeeze(1)
            p0 = 1.0 - p1
            probs = torch.stack([p0, p1], dim=1).numpy()
        else:
            # Multi-class-apply softmax
            probs = F.softmax(logits_tensor, dim=1).numpy()
    else:
        raise ValueError(f"Unexpected logits shape: {logits_tensor.shape}")

    # Save
    np.save(os.path.join(results_path, f"{dataset_type}_probs.npy"), probs)
    np.save(os.path.join(results_path, f"{dataset_type}_labels.npy"), labels)

    print(f"Saved probabilities and labels for {dataset_type} set to: {results_path}")



import pandas as pd
from collections import defaultdict
from random import seed
import random

def AugmentWithPriority1(df: pd.DataFrame, sequence_col='sequence', label_col='label',
                        random_state=42, priority_order=None) -> pd.DataFrame:
    """
    Balances classes in a dataset using augmentation methods based on user-defined priority.

    Args:
        df (pd.DataFrame): DataFrame with 'sequence' and 'label' columns.
        sequence_col (str): Name of sequence column.
        label_col (str): Name of label column.
        random_state (int): Reproducibility seed.
        priority_order (List[str]): List of operations to apply in order. Options: 'complement', 'reverse', 'reverse_complement', 'duplicate'.

    Returns:
        pd.DataFrame: Augmented and balanced dataset.
    """
    def complement(seq):
        return seq.translate(str.maketrans("ATCG", "TAGC"))

    def reverse(seq):
        return seq[::-1]

    def reverse_complement(seq):
        return reverse(complement(seq))

    # Set default order if not specified
    if priority_order is None:
        priority_order = ['complement', 'reverse', 'reverse_complement', 'duplicate']

    # Mapping method names to functions
    methods = {
        'complement': complement,
        'reverse': reverse,
        'reverse_complement': reverse_complement,
        'duplicate': lambda x: x  # identity for sampling
    }

    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()
    augmented_dfs = []

    seed(random_state)
    random.seed(random_state)
    print("\n?? Augmentation Summary (Custom Priority):")

    for label, count in class_counts.items():
        df_class = df[df[label_col] == label].copy()
        existing = set(df_class[sequence_col])
        needed = max_count - count
        augmented = []

        if needed > 0:
            for method_name in priority_order:
                if method_name not in methods:
                    raise ValueError(f"Unknown augmentation method: {method_name}")

                if method_name == 'duplicate':
                    remaining = needed - len(augmented)
                    if remaining > 0:
                        sampled = df_class.sample(n=remaining, replace=True, random_state=random_state)[sequence_col]
                        augmented.extend([(s, label) for s in sampled])
                    break  # Done

                func = methods[method_name]
                for seq in df_class[sequence_col]:
                    if len(augmented) >= needed:
                        break
                    aug_seq = func(seq)
                    if aug_seq not in existing:
                        augmented.append((aug_seq, label))
                        existing.add(aug_seq)

        new_df = pd.DataFrame(augmented, columns=[sequence_col, label_col])
        augmented_dfs.append(df_class)
        augmented_dfs.append(new_df)

        print(f"  Class {label}: original={count}, added={len(new_df)}, final={count + len(new_df)}")

    final_df = pd.concat(augmented_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"\n? Total samples after augmentation: {len(final_df)}")
    return final_df



from transformers import TrainerCallback
import os
import json

class TestEvaluationCallback(TrainerCallback):
    def __init__(self, test_dataset, output_dir, interval=100):
        self.test_dataset = test_dataset
        self.output_dir = output_dir
        self.interval = interval
        self.trainer_ref = None  # Set externally
            
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer_ref is None:
            return
    
        epoch = int(state.epoch)
        if epoch % self.interval == 0:
            metrics = self.trainer_ref.evaluate(eval_dataset=self.test_dataset)
    
            # Rename keys to remove prefix (cleaner and consistent)
            clean_metrics = {}
            for k, v in metrics.items():
                if k.startswith("eval_") or k.startswith("test_epoch"):
                    new_k = k.split("_", maxsplit=1)[-1]
                    clean_metrics[f"test_{new_k}"] = v
                else:
                    clean_metrics[f"test_{k}"] = v
            clean_metrics["epoch"] = epoch
    
            # Append to single file
            output_file = os.path.join(self.output_dir, "test_results.txt")
            os.makedirs(self.output_dir, exist_ok=True)
            with open(output_file, "a") as f:
                f.write(json.dumps(clean_metrics) + "\n")
    
            print(f"? [Epoch {epoch}] Test metrics appended to: {output_file}")




def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # Load train.csv manually
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from collections import Counter
  
    # Reload data after state reset
    train_df = pd.read_csv(os.path.join(data_args.data_path, "train.csv"))
    dev_df = pd.read_csv(os.path.join(data_args.data_path, "dev.csv"))
    test_df =pd.read_csv(os.path.join(data_args.data_path, "test.csv"))
    
    # Combine them into a single DataFrame
    combined_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
        
    # Optionally filter out sequences shorter than 1000
    MIN_SEQ_LENGTH = None  # Change this to None to disable filtering
    if MIN_SEQ_LENGTH:
        combined_df = combined_df[combined_df['sequence'].str.len() >= MIN_SEQ_LENGTH]
    
    # Non-overlapping data, Stratified split into train, val, test
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, stratify=combined_df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
    
    
    # Count class distributions
    print ("\nCount class before balancing ---")
    train_counts = Counter(train_df['label'])
    val_counts = Counter(val_df['label'])
    test_counts = Counter(test_df['label'])
    summary_df = pd.DataFrame({
        "Train": pd.Series(train_counts),
        "Validation": pd.Series(val_counts),
        "Test": pd.Series(test_counts)
    }).fillna(0).astype(int)
    print("\nTrain/Val/Test Split Class Counts:")
    print(summary_df)
    
    
    ###---AugmentWithPriority, it giving priority for 1. Complement 2. Reverse 3. Reverse Complement 4. Duplicates (if needed)
    print ("Augmentation started --")
    

    priority = ['reverse_complement', 'duplicate'] # 'complement', 'reverse', 'reverse_complement', 'duplicate'
    #priority = ['reverse_complement', 'duplicate']
    
    # final ViSTA, augment only train and val data.
    #train_df = AugmentWithPriority1(train_df, priority_order=priority)
    #val_df = AugmentWithPriority1(val_df, priority_order=priority)
    #test_df = AugmentWithPriority1(test_df, priority_order=priority) 

    print ("---- Completed ---")


    ## Saving the test data
    test_data_save_path = os.path.join(training_args.output_dir, "test_data_used.csv")
    os.makedirs(os.path.dirname(test_data_save_path), exist_ok=True)
    pd.DataFrame(test_df).to_csv(test_data_save_path, index=False)
    print(f"? Test dataset saved to: {test_data_save_path}")
    
    print ("\nCount class after balancing ---")
    # Count class distributions
    train_counts = Counter(train_df['label'])
    val_counts = Counter(val_df['label'])
    test_counts = Counter(test_df['label'])
    summary_df = pd.DataFrame({
        "Train": pd.Series(train_counts),
        "Validation": pd.Series(val_counts),
        "Test": pd.Series(test_counts)
    }).fillna(0).astype(int)
    
    print("\nTrain/Val/Test Split Class Counts:")
    print(summary_df)  
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data=train_df.values.tolist(), kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, data=val_df.values.tolist(), kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, data=test_df.values.tolist(), kmer=data_args.kmer)
    #print("Sample from test_dataset:")
    #print(test_dataset[0])  # prints first item
    
    
    ## Adding weights
    label_counts = np.bincount(train_dataset.labels)
    weights = 1.0 / label_counts
    weights = weights / weights.sum()
    class_weights = torch.tensor(weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

    #adding manual weight
    from sklearn.utils.class_weight import compute_class_weight
    unique_labels = sorted(set(train_dataset.labels))
    computed_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=train_dataset.labels)
    class_weights = torch.tensor(computed_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")



    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True
    )
    model = WeightedBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        class_weights=class_weights
    )

    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        # Patch: make active_adapters subscriptable
        if hasattr(model, "active_adapters") and callable(model.active_adapters):
            model.active_adapters = model.active_adapters()        
        
        model.print_trainable_parameters()
        
    # Init callback first
    test_callback = TestEvaluationCallback(
        test_dataset=test_dataset,
        output_dir=os.path.join(training_args.output_dir, "test_results", training_args.run_name),
        interval=20  # after 20 epoch test accuracy will be calculated.
    )
        
    # Initialize trainer
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Manually assign trainer into callback and add
    test_callback.trainer_ref = trainer
    trainer.add_callback(test_callback)    

    #trainer = transformers.Trainer(
    #    model=model,
    #    tokenizer=tokenizer,
    #    args=training_args,
    #    #preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    #    compute_metrics=compute_metrics,
    #    train_dataset=train_dataset,
    #    eval_dataset=val_dataset,
    #    data_collator=data_collator
    #)

    trainer.train()
    
    #if training_args.load_best_model_at_end:
    #    print(f"Best model checkpoint loaded from: {trainer.state.best_model_checkpoint}")
            # Save best model to a dedicated folder after training
    from peft import PeftModel
    
    if training_args.load_best_model_at_end and trainer.state.best_model_checkpoint is not None:
        print(f"Best model checkpoint loaded from: {trainer.state.best_model_checkpoint}")
        
        best_model_dest = os.path.join(training_args.output_dir, "best_model")
        
        if os.path.exists(best_model_dest):
            import shutil
            shutil.rmtree(best_model_dest)
        os.makedirs(best_model_dest, exist_ok=True)
    
        # If using LoRA, save adapter separately
        if isinstance(model, PeftModel):
            model.save_pretrained(best_model_dest)
            print("? Saved LoRA adapter weights.")
        else:
            model.save_pretrained(best_model_dest, safe_serialization=True)
    
        tokenizer.save_pretrained(best_model_dest)
        print(f"? Tokenizer and model saved to: {best_model_dest}")


    if training_args.eval_and_save_results:
        predictions_output = trainer.predict(test_dataset)
        raw_preds = predictions_output.predictions
        labels = np.array(predictions_output.label_ids).flatten()
        predictions = np.argmax(raw_preds, axis=-1) if raw_preds.ndim > 1 else raw_preds
        predictions = np.array(predictions).flatten()

        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        os.makedirs(results_path, exist_ok=True)

        results = compute_metrics((predictions, labels))
        with open(os.path.join(results_path, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=4)

        all_preds_file = os.path.join(results_path, "all_predictions.csv")
        misclassified_file = os.path.join(results_path, "misclassified_samples.csv")
        summary_file = os.path.join(results_path, "eval_summary.txt")

        misclassified_counter = Counter()
        total_counter = Counter(labels)

        with open(all_preds_file, "w") as fa, open(misclassified_file, "w") as fm:
            fa.write("Index,Predicted,True\n")
            fm.write("Index,Predicted,True\n")
            for i, (pred, true) in enumerate(zip(predictions, labels)):
                fa.write(f"{i},{int(pred)},{int(true)}\n")
                if int(pred) != int(true):
                    fm.write(f"{i},{int(pred)},{int(true)}\n")
                    misclassified_counter[int(true)] += 1

        confusion = confusion_matrix(labels, predictions)

        with open(summary_file, "w") as fsum:
            fsum.write("Class-wise Performance:\n")
            for cls in sorted(total_counter):
                total = total_counter[cls]
                misclassified = misclassified_counter.get(cls, 0)
                accuracy = 100.0 * (total - misclassified) / total
                fsum.write(f"  Class {cls}: Total = {total}, Misclassified = {misclassified}, Accuracy = {accuracy:.2f}%\n")
                print (f"  Class {cls}: Total = {total}, Misclassified = {misclassified}, Accuracy = {accuracy:.2f}%\n")
            fsum.write("\nConfusion Matrix (rows = true, cols = predicted):\n")
            print ("\n")
            for row in confusion:
                fsum.write("  " + "\t".join(map(str, row)) + "\n")
                print ("  " + "\t".join(map(str, row)) + "\n")


    # Save ROC-relevant logits and labels
    save_logits_and_labels(trainer, train_dataset, "train", train_dataset.num_labels, results_path)
    save_logits_and_labels(trainer, test_dataset, "test", train_dataset.num_labels, results_path)


    if training_args.save_model:
        trainer.save_state()
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        trainer._save(training_args.output_dir, state_dict=cpu_state_dict)


if __name__ == "__main__":
    train()
