import modal
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = modal.App("rna-transformer")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers==4.36.2",
        "tqdm",
        "huggingface_hub",
        "scikit-learn"
    )
    .add_local_dir("training_data/homo_sapiens/chromosome_20", remote_path="/root/training_data/homo_sapiens/chromosome_20")
    .add_local_file("data_tokenization.py", remote_path="/root/data_tokenization.py")
)

@app.function(
    image=image,
    gpu="L40S",
    timeout=60*60*6
)
def validate():
    import torch
    from transformers import AutoModelForTokenClassification
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    from sklearn.metrics import classification_report
    from data_tokenization import GenomicTokenDataset

    def get_model(repo_id: str, device):
        model = AutoModelForTokenClassification.from_pretrained(repo_id, trust_remote_code=True)
        
        model.to(device)

        return model

    def initialize_dataloader(data_dir: str, tokenizer_name: str) -> DataLoader:
            dataset = GenomicTokenDataset(
                data_dir=data_dir,
                tokenizer_name=tokenizer_name
            )

            #NOTE: validate a small subset of an unseen chromosome sequence
            quarter_size = len(dataset) // 4
            indices = torch.randperm(len(dataset))[:quarter_size]
            subset = Subset(dataset, indices)

            dataloader = DataLoader(subset, batch_size=128, num_workers=4, pin_memory=True)

            return dataloader

    def validate_model(model, val_dataloader, device, threshold: float = 0.5):

        print("\nStarting validation...\n")

        model.eval()

        all_preds = []
        all_true_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    logits = outputs.logits

                    # Convert logits to probabilities
                    probs = torch.softmax(logits, dim=-1)

                    # Get exon probabilities (class 1)
                    exon_probs = probs[:, :, 1]

                    # Apply threshold to get binary predictions
                    preds = (exon_probs > threshold).long()

                preds_flat = preds.view(-1)
                labels_flat = labels.view(-1)

                active_indices = (attention_mask.view(-1) == 1) & (labels_flat != -100)
                valid_preds = preds_flat[active_indices].cpu().numpy()
                valid_labels = labels_flat[active_indices].cpu().numpy()

                all_preds.extend(valid_preds)
                all_true_labels.extend(valid_labels)

        print("\n--- Model Eval Report ---\n")

        import numpy as np
        unique_labels = np.unique(all_true_labels)
        print(f"Unique labels in validation set: {unique_labels}")
        unique_preds = np.unique(all_preds)
        print(f"Unique predictions made by model: {unique_preds}\n")

        report = classification_report(
            all_true_labels,
            all_preds,
            labels=[0, 1],
            target_names=['Intron (0)', 'Exon (1)'],
            digits=4
        )

        print(report)

    def run_validation():
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        model = get_model("batmanLovesAI/exon_intron_classifier", device)

        val_dataloader = initialize_dataloader("/root/training_data/homo_sapiens/chromosome_20", "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species")

        validate_model(model, val_dataloader, device, threshold=0.7)

    run_validation()
