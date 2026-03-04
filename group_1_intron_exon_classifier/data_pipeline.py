import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
import os

def create_exon_intron_dataset(fasta_path: str, gff_path: str, output_dir: str, target_chrom: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading fasta file for chromosome: {target_chrom}...")

    record_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    if target_chrom not in record_dict:
        raise ValueError(f"Chromosome {target_chrom} is not found in the fasta file.")
    chrom_seq = record_dict[target_chrom].seq

    print("Loading and parsing GFF file...")

    gff_cols = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
    df = pd.read_csv(gff_path, sep='\t', comment='#', names=gff_cols)

    print(df.head())

    df = df[df['seqid'] == target_chrom]

    def get_attribute(attr_str: str, key):
        for item in attr_str.split(';'):
            if item.startswith(key+'='):
                return item.split('=')[1]
        return None

    transcripts = df[df['type'].isin(['mRNA', 'transcript'])].copy()
    exons = df[df['type'] == 'exon'].copy()

    transcripts['ID'] = transcripts['attributes'].apply(lambda x: get_attribute(x, 'ID'))
    exons['Parent'] = exons['attributes'].apply(lambda x: get_attribute(x, 'Parent'))

    print(f"Found {len(transcripts)} transcripts. Processing...")

    processed_count = 0
    for _, transcript in transcripts.iterrows():
        t_id = transcript['ID']
        t_start = transcript['start']-1
        t_end = transcript['end']
        strand = transcript['strand']

        # Raw pre-mRNA sequence
        pre_mrna_seq = chrom_seq[t_start:t_end]

        # Initialize the label masks with 0s (introns)
        seq_length = t_end - t_start
        label_mask = np.zeros(seq_length, dtype=np.int8)

        # Find child exons and update the mask
        child_exons = exons[exons['Parent'] == t_id]
        for _, exon in child_exons.iterrows():
            # Exon coordinates relative to the transcript start
            e_start_rel = (exon['start']-1) - t_start
            e_end_rel = exon['end'] - t_start

            # Set exon region to 1
            label_mask[e_start_rel:e_end_rel] = 1

        # Handle reverse strand
        if strand == '-':
            pre_mrna_seq = pre_mrna_seq.reverse_complement()
            label_mask = label_mask[::-1]

        output_file = os.path.join(output_dir, f"{t_id.replace(':', '_')}.npz")
        np.savez_compressed(
            output_file,
            sequence=str(pre_mrna_seq),
            labels=label_mask
        )

        processed_count+=1
        if processed_count % 500 == 0:
            print(f"Processed {processed_count} transcripts...")

    print(f"Done! Dataset saved to {output_dir}")

fasta_file = "data/humans/GCF_009914755.1/GCF_009914755.1_T2T-CHM13v2.0_genomic.fna"
gff_file = "data/humans/GCF_009914755.1/genomic.gff"
output_directory = "training_data/homo_sapiens/chromosome_20"
target_chromosome = "NC_060944.1" # Chromosome 20

create_exon_intron_dataset(fasta_file, gff_file, output_directory, target_chromosome)
