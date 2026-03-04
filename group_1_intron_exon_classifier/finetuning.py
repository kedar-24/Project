import modal
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.getenv("HF_READ_TOKEN") and os.getenv("HF_WRITE_TOKEN"):
    raise ValueError("API Tokens not provided!")

read_token = os.getenv("HF_READ_TOKEN")
write_token = os.getenv("HF_WRITE_TOKEN")

# NOTE: Modal provides SOTA GPUs for free
app = modal.App("rna-transformer")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers==4.36.2",
        "tqdm",
        "huggingface_hub"
    )
    .add_local_dir("training_data/homo_sapiens/chromosome_21", remote_path="/root/training_data/homo_sapiens/chromosome_21")
    .add_local_file("data_tokenization.py", remote_path="/root/data_tokenization.py")
)

@app.function(
    image=image,
    gpu="L40S",
    timeout=60*60*6
)

def train():

    import torch
    from torch.optim import AdamW
    from torch.nn import CrossEntropyLoss
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from tqdm import tqdm
    from data_tokenization import GenomicTokenDataset
    from torch.utils.data import DataLoader, Subset

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA!")
    else:
        device = torch.device("cpu")
        print("Using CPU :(")


    def initialize_dataloader(data_dir: str, tokenizer_name: str) -> DataLoader:
        dataset = GenomicTokenDataset(
            data_dir=data_dir,
            tokenizer_name=tokenizer_name
        )
        
        #NOTE: Training on entire dataset is very expensive so train sequentially.
        # quarter_size = len(dataset) // 2
        # indices = torch.randperm(len(dataset))[:quarter_size]
        # subset = Subset(dataset, indices)

        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

        return dataloader

    def initialize_model(model_name: str, device):
        
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            token=read_token, # read token
            num_labels=2,
            trust_remote_code=True
        ).to(device)

        if hasattr(torch, "compile"):
            model = torch.compile(model)

        return model

    model_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species" # Base model for fine-tuning 
    tokenizer_name = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"

    model = initialize_model(model_name, device)

    dataloader = initialize_dataloader("/root/training_data/homo_sapiens/chromosome_21", tokenizer_name)

    class_weights = torch.tensor([0.1, 0.9]).to(device) # Give more importance to Exons because they are less in number
    criterion = CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    optimizer = AdamW(model.parameters(), lr=2e-5, fused=True)

    scaler = torch.amp.GradScaler("cuda")

    epochs = 1
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):

            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits

                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, 2)
                active_labels = labels.view(-1)

                loss = criterion(
                    active_logits[active_loss],
                    active_labels[active_loss]
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss+=loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

        repo_id = "batmanLovesAI/exon_intron_classifier" # HF model repo where the model will be pushed and later will be pulled to continue training on other chromosomes

        model.push_to_hub(
            repo_id,
            token=write_token, # write token
            commit_message=f"Human v1 model trained on Human chromosome 22 for epoch {epoch+1}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", 
        trust_remote_code=True
    )

    tokenizer.push_to_hub(
        repo_id, 
        token=write_token, # write token
        commit_message="Uploading tokenizer"
    )

    print(f"Success! Model uploaded to: https://huggingface.co/{repo_id}") 

