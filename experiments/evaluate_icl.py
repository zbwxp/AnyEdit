import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import shutil
from pathlib import Path
from itertools import islice
from time import time
from typing import Tuple, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random
from dsets import (
    UnKEDataset,
    CounterFactDataset,
    MQUAKEDataset,
    EditeveryDataset
)

from memit import MEMITHyperParams, apply_memit_to_model
from memit_ARE import MEMITAREHyperParams, apply_memit_ARE_to_model
from AlphaEdit import AlphaEditHyperParams,apply_AlphaEdit_to_model,get_cov
from AlphaEdit_ARE import AlphaEditAREHyperParams,apply_AlphaEdit_ARE_to_model
from unke import unkeHyperParams, apply_unke_to_model
from unke_ARE import unkeAREHyperParams, apply_unke_ARE_to_model
from util import nethook
from util.globals import *
# from glue_eval.glue_eval import GLUEEval
ALG_DICT = {
    "unke_ARE": (unkeAREHyperParams, apply_unke_ARE_to_model),
    "unke": (unkeHyperParams, apply_unke_to_model),
    "AlphaEdit_ARE": (AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model),
    "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "MEMIT_ARE": (MEMITAREHyperParams, apply_memit_ARE_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
}

DS_DICT = {
    "unke": UnKEDataset,
    "cf": CounterFactDataset,
    "mquake": MQUAKEDataset,
    "editevery": EditeveryDataset,
}
def get_llama_without_answer(que):
    return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{que}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

def get_qwen_without_answer(que):
    return f"""<|im_start|>user\n{que}<|im_end|>\n<|im_start|>assistant\n"""

def set_seed(seed=2024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def clear_question(q):
    if '<|start_header_id|>user<|end_header_id|>' in q:
        parts = q.split('<|start_header_id|>user<|end_header_id|>')
        if len(parts) > 1:
            q = parts[1].split('<|eot_id|>')[0].strip('\n ')
    elif '<|im_start|>user' in q:
        parts = q.split('<|im_start|>user')
        if len(parts) > 1:
            q = parts[1].split('<|im_end|>')[0].strip('\n ')
    return q

def format_questions_with_system_context(tokenizer, question, answer):
    # Extract clean question from existing formatting
    clean_question = clear_question(question)
    
    clean_answer = answer.replace("<|eot_id|>","").replace("<|im_end|>","")

    answers_json = json.dumps({"items":[{
        "canonical_question": clean_question,
        "answer": clean_answer}]})

    system_message = f"""You are an Answer-Key Router. Answer ONLY from the Answer Key.

ANSWER KEY:
{answers_json}

RULES
- If the user query is the canonical question OR a paraphrase of it (same scope), return the canonical 'answer' string verbatim (identical characters).
- Do NOT shorten, summarize, or extract fragments.
- If the query is a focused facet/sub-question, reply exactly: Unknown.
- Output ONLY the answer string or Unknown. No explanations or extra text.
"""
    # Create proper message structure
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": clean_question}
    ]
    
    # Use tokenizer's chat template (this is the key!)
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_input

def format_sub_questions_with_system_context(tokenizer, std_question, answer, sub_qs):
    std_question = clear_question(std_question)
    clean_sub_qs = [clear_question(q) for q in sub_qs]
    clean_answer = answer.replace("<|eot_id|>","").replace("<|im_end|>","")
    
    answers_json = json.dumps({"items":[{
        "canonical_question": std_question,
        "answer": clean_answer}]})

    

    formatted_list = []
    for sub_qs in clean_sub_qs:
        system_message = f"""You are an Answer-Key Router. Answer ONLY from the Answer Key.

ANSWER KEY:
{answers_json}

WHEN TO ANSWER
- Answer ONLY focused facet/sub-questions whose answers are entailed by a canonical 'answer' (e.g., how long/since/when/where/which; positions/roles/titles; which publications; what was discussed; education; why valuable/strengths; counts/dates/orgs/locations/tenure).

EXTRACTION-AND-NORMALIZE POLICY
A) Start with the minimal content needed to answer the facet, taken from the canonical answer (via single span or by composing multiple spans without inventing words).
B) Then apply these NORMALIZATIONS (deterministic, style-only):
   1. **Capitalize** the first character of the final output.
   2. **End with a period.**
   3. **Preserve entity casing** exactly (e.g., "The New York Times", "The Guardian").
   4. **Trim scaffolding** that precedes the core fact (drop lead-ins like "including", "such as", "for example", "in an interview", "he/she/they also mentioned that", "that").
   5. If an education clause like **"he has a/has a degree in …"** exists, prefer that full clause (e.g., "He has a degree in journalism from a top university.").
   6. For publications, return the **list only**: keep just outlet names in source order; join two with " and "; for 3+, use commas and " and " before the last.
   7. For titles/positions, return title noun phrases exactly as written; keep source order; join two with " and "; for 3+, use commas and " and " before the last.
   8. For reasons/qualities (e.g., "What makes X valuable?"), return the minimal cause phrase composed ONLY of words in the canonical answer (e.g., "His extensive experience and education"). Optionally append a domain PP like "in journalism" **only if that exact phrase appears elsewhere** in the canonical answer.
   9. If the final phrase begins with a possessive pronoun from the source (e.g., "his"), capitalize it to start the sentence ("His …").
C) Do NOT introduce words not present somewhere in the canonical answer. Do NOT return the full canonical answer.

OUTPUT
- Output ONLY the minimal, normalized answer string or Unknown. No extra text.
"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sub_qs}
        ]

        # Use tokenizer's chat template (this is the key!)
        formatted_input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_list.append(formatted_input)
    
    return formatted_list

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    sequential: bool = False,
):
    set_seed()
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    params_path = (HPARAMS_DIR / alg_name / hparams_fname)
    hparams = params_class.from_json(params_path)

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model_name = "/root/code/AnyEdit/qwen/Qwen2.5-7B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            ).cuda()

        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path
    ds_class = DS_DICT[ds_name]
    ds = ds_class(DATA_DIR, model_name=hparams.model_name, size=dataset_size_limit)
    with open(Path(DATA_DIR)/"alpaca_data.json", 'r', encoding='utf-8') as json_file:
        ex_datas = json.load(json_file)
    if hparams.model_name == 'Llama3-8B-Instruct':
        ex_datas = [get_llama_without_answer(i['instruction']+i['input'])+i['output']  for i in ex_datas]
    elif hparams.model_name == 'Qwen2.5-7B-Instruct':
        ex_datas = [get_qwen_without_answer(i['instruction']+i['input'])+i['output']  for i in ex_datas]
    tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left')

    # Set pad_token for both Llama3 and Qwen2.5
    if hparams.model_name == 'Llama3-8B-Instruct':
        tokenizer.pad_token_id = tok.eos_token_id
        tokenizer.pad_token = tok.eos_token
    elif hparams.model_name == 'Qwen2.5-7B-Instruct':
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = num_edits
    num_batches = len(ds) // batch_size + (1 if len(ds) % batch_size else 0)
    edited_data = []

    # Prepare output path and ensure directory exists for incremental dumps
    if sequential:
        path = f'output/{alg_name}_{hparams.model_name}_sequential_{ds_name}_result.json'
    else:
        path = f'output/{alg_name}_{hparams.model_name}_icl_{ds_name}_result.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Resume support: load existing results if present
    resume_count = 0
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if isinstance(existing, list):
                edited_data = existing
                resume_count = len(edited_data)
                print(f"Resuming from {resume_count} existing results in {path}")
        except Exception as e:
            print(f"Could not load existing results from {path}: {e}")

    # For non-sequential mode, skip completed batches
    start_batch_index = (resume_count // batch_size) if not sequential else 0


    for batch_index in tqdm(range(start_batch_index, num_batches)):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size
        batch = ds[start_index:end_index]

        if not sequential:
            for data in batch:

                question = format_questions_with_system_context(tokenizer, data['question'], data['answer'])
                para_question = format_questions_with_system_context(tokenizer, data['para_question'], data['answer'])
                question = tokenizer([question, para_question], return_tensors='pt', padding=True)
                #print(question.input_ids)
                with torch.no_grad():
                    generated_ids = model.generate(
                    input_ids=question['input_ids'].to('cuda'),
                    attention_mask=question['attention_mask'].to('cuda'),
                    do_sample=True,
                    temperature=0.001,
                    max_new_tokens=1024
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                ]
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                # if batch_index < 10 // batch_size + 1:
                    # print(f"question:{data['question']}")
                    # print(output[0])
                    # if ds_name in ['unke','cf']:
                    #     print(f"question:{data['para_question']}")
                    #     print(output[1])
                data['original_prediction'] = output[0]
                if ds_name in ['unke','cf']:
                    data['para_prediction'] = output[1]
                if hparams.model_name == 'Llama3-8B-Instruct':
                    data['answer'] = data['answer'][:-len('<|eot_id|>')]
                elif hparams.model_name == 'Qwen2.5-7B-Instruct':
                    data['answer'] = data['answer'][:-len('<|im_end|>')]
            if ds_name in ['unke','cf','mquake']:
                for data in batch:
                    sub_question = format_sub_questions_with_system_context(tokenizer, data['question'], data['answer'], data['sub_question'],)
                    question = tokenizer(sub_question, return_tensors='pt', padding=True)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                        input_ids=question['input_ids'].to('cuda'),
                        attention_mask=question['attention_mask'].to('cuda'),
                        do_sample=True,
                        temperature=0.4,# Analysis exp
                        max_new_tokens=2048
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                    ]

                    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    # if batch_index < 10 // batch_size + 1:
                    #     print(f"question:{data['sub_question']}")
                    #     print(output)

                    data['sub_pred'] = output

            edited_data.extend(batch)
            # Incremental dump after each batch iteration (after vanilla predictions)
            with open(path, 'w', encoding='utf-8') as json_file:
                json.dump(edited_data, json_file, ensure_ascii=False, indent=4)
            print("=============================================")
            current_id = edited_data[-1]['id']
            print(f"Batch {batch_index} done, {current_id} is the last example saved to {path}")
            print("=============================================")


    if sequential:
        for data in ds[resume_count:]:
            if ds_name in ['unke','cf']:
                question = tokenizer([data['question'],data['para_question']], return_tensors='pt', padding=True)
            else:
                question = tokenizer([data['question']], return_tensors='pt', padding=True)
            #print(question.input_ids)
            with torch.no_grad():
                generated_ids = model.generate(
                input_ids=question['input_ids'].to('cuda'),
                attention_mask=question['attention_mask'].to('cuda'),
                do_sample=True,
                temperature=0.001,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
            ]
            output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if batch_index < 10 // batch_size + 1:
                print(f"question:{data['question']}")
                print(output[0])
                if ds_name in ['unke','cf']:
                    print(f"question:{data['para_question']}")
                    print(output[1])
            data['original_prediction'] = output[0]
            if ds_name in ['unke','cf']:
                data['para_prediction'] = output[1]
            if hparams.model_name == 'Llama3-8B-Instruct':
                data['answer'] = data['answer'][:-len('<|eot_id|>')]
            elif hparams.model_name == 'Qwen2.5-7B-Instruct':
                data['answer'] = data['answer'][:-len('<|im_end|>')]
        if ds_name in ['unke','cf','mquake']:
            for data in ds[resume_count:]:
                question = tokenizer(data['sub_question'], return_tensors='pt', padding=True)
                with torch.no_grad():
                    generated_ids = model.generate(
                    input_ids=question['input_ids'].to('cuda'),
                    attention_mask=question['attention_mask'].to('cuda'),
                    do_sample=True,
                    temperature=0.001,# Analysis exp
                    max_new_tokens=512
                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(question['input_ids'], generated_ids)
                ]

                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                # if batch_index < 10 // batch_size + 1:
                #     print(f"question:{data['sub_question']}")
                #     print(output)

                data['sub_pred'] = output

        edited_data.extend(ds[resume_count:])
        # Final dump for sequential mode (non-sequential already dumps per batch above)
        with open(path, 'w', encoding='utf-8') as json_file:
            json.dump(edited_data, json_file, ensure_ascii=False, indent=4)
        print("=============================================")
        print(f"Sequential mode done, {len(edited_data)} examples saved to {path}")
        print("=============================================")

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["AlphaEdit","AlphaEdit_ARE", "MEMIT","MEMIT_ARE", "ROME", "FT", "MEND","unke","unke_ARE"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="Llama3-8B-Instruct",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="Llama3-8B-Instruct.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        choices=["cf", "editevery", "unke","mquake"],
        default="mcf",
        help="Dataset to perform evaluations on. Either CounterFact (cf), MultiCounterFact (mcf), or zsRE (zsre).",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--sequential",
        dest="sequential",
        action="store_true",
        help="sequential editing",
    )

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        dir_name=args.alg_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        sequential=args.sequential,
    )
