import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class GenomicTokenDataset(Dataset):
    def __init__(self, data_dir, tokenizer_name, chunk_size=512, kmer_size=6, stride=256):

        if not os.path.exists(data_dir):
            raise ValueError(f"Data Directory {data_dir} doesn't exists. Please check the path.")

        self.file_paths = glob.glob(os.path.join(data_dir, "*.npz"))
        self.chunk_size = chunk_size
        self.stride = stride
        self.kmer_size = kmer_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

        print("Building chunk index (time consuming)...")
        self.chunk_index = []
        for file_idx, path in enumerate(self.file_paths):
            data = np.load(path)
            seq_len = len(data['labels'])

            if seq_len >= self.chunk_size:
                num_chunks = seq_len // self.chunk_size

                num_chunks = ((seq_len-self.chunk_size) // self.stride) + 1

                for i in range(num_chunks):
                    start_ps = i*self.stride
                    self.chunk_index.append((file_idx, start_ps))

                last_start = seq_len-self.chunk_size
                if len(self.chunk_index) == 0 or self.chunk_index[-1][1] != last_start:
                    self.chunk_index.append((file_idx, last_start))

        print(f"Total overlapping chunks available: {len(self.chunk_index)}")

    def __len__(self):
        return len(self.chunk_index)

    def sequence_to_kmers(self, seq):
        kmers = [seq[i:i+self.kmer_size] for i in range(len(seq)-self.kmer_size+1)]

        return "".join(kmers)

    def __getitem__(self, idx):
        file_idx, start_pos = self.chunk_index[idx]
        file_path = self.file_paths[file_idx]

        data = np.load(file_path)
        end_pos = start_pos + self.chunk_size

        raw_seq = str(data['sequence'])[start_pos:end_pos]
        chunk_labels = data['labels'][start_pos:end_pos]

        kmer_string = self.sequence_to_kmers(raw_seq)

        encoding = self.tokenizer(
            kmer_string,
            padding="max_length",
            truncation=True,
            max_length=self.chunk_size,
            return_tensors="pt"
        )

        aligned_labels = chunk_labels[:-(self.kmer_size-1)]

        pad_length = self.chunk_size - len(aligned_labels)
        if pad_length > 0:
            padded_labels = np.pad(aligned_labels, (0, pad_length), constant_values=-100)
        else:
            padded_labels = aligned_labels[:self.chunk_size]

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(padded_labels, dtype=torch.long)

        return item

if __name__ == "__main__":
    dataset = GenomicTokenDataset(
        data_dir="training_data/homo_sapiens/chromosome_20",
        tokenizer_name="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    batch = next(iter(dataloader))
    print("Input IDs shape: ", batch['input_ids'].shape)
    print("Labels shape: ", batch['labels'].shape)
