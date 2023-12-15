import numpy as np
import torch
from datasets import load_dataset
from more_itertools import chunked
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def perplexity(model, tokenizer, dataset, context_size, batch_size=1):
    """Calculate the perplexity of a model on a dataset.

    Adapted from https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py#L103,
    but I needed more control over the model and tokenizer parameters.
    """
    loss_fct = CrossEntropyLoss(reduction="none")

    # TODO Striding
    ppls = []
    for rows in tqdm(chunked(dataset, batch_size), total=len(dataset)//batch_size):
        texts = [row['text'] for row in rows]

        encodings = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=context_size,
            return_tensors='pt',
            return_attention_mask=True,
        ).to(DEVICE)

        encoded_texts = encodings['input_ids']
        labels = encoded_texts
        attn_masks = encodings['attention_mask']

        with torch.no_grad():
            out_logits = model(encoded_texts, attention_mask=attn_masks).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_masks[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {'perplexities': ppls, 'mean_perplexity': np.mean(ppls)}


if __name__ == '__main__':
    dataset = load_dataset('Rijgersberg/mc4_nl_cleaned_tiny_validation', split='validation')

    models = [
        ('Rijgersberg/GEITje-7B', {'torch_dtype': torch.bfloat16, 'use_flash_attention_2': True}),
        ('mistralai/Mistral-7B-v0.1', {'torch_dtype': torch.bfloat16, 'use_flash_attention_2': True}),
        ('meta-llama/Llama-2-7b-hf', {'torch_dtype': torch.bfloat16, 'use_flash_attention_2': True}),
        ('meta-llama/Llama-2-13b-hf', {'torch_dtype': torch.bfloat16, 'use_flash_attention_2': True}),
        ('meta-llama/Llama-2-70b-hf', {'load_in_8bit': True, 'use_flash_attention_2': True}),
        ('BramVanroy/llama2-13b-ft-mc4_nl_cleaned_tiny', {'torch_dtype': torch.bfloat16, 'use_flash_attention_2': True}),
        ('tiiuae/falcon-7b', {'torch_dtype': torch.bfloat16}),
        ('tiiuae/falcon-40b', {'load_in_8bit': True}),
        ('BramVanroy/falcon-7b-ft-mc4_nl_cleaned_tiny', {'torch_dtype': torch.bfloat16}),
        ('bigscience/bloom-7b1', {'torch_dtype': torch.bfloat16}),
    ]
    perplexities = {}
    for model_name, model_kwargs in models:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map=0,
                                                     low_cpu_mem_usage=True,
                                                     **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        perplexities[model_name] = perplexity(model=model,
                                              tokenizer=tokenizer,
                                              dataset=dataset,
                                              context_size=4096,  # TODO how to handle varying context sizes
                                              batch_size=1)

        with open('perplexities.txt', 'w') as f:
            for name, scores in perplexities.items():
                line = f'{name}: {scores["mean_perplexity"]}'
                print(line)
                f.write(line + '\n')

    for name, result in perplexities.items():
        print(name, result['mean_perplexity'])
