import json
import torch
import torch.nn as nn
import random
import os
import numpy as np
from openprompt.data_utils import InputExample
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer
import logging
from tqdm import tqdm, trange


from openprompt.prompts import SoftTemplate
from openprompt.prompts import MixedTemplate, PrefixTuningTemplate
from openprompt.plms import load_plm, T5TokenizerWrapper
from openprompt import PromptDataLoader
from openprompt import PromptForGeneration
from transformers import AdamW
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def read_answers(filename):
    answers=[]
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            # code = js['func']
            # target = js['target']
            tgt_txt = "true" if js['target'] == 1 else "false"
            example = InputExample(text_a=js['func'], tgt_text=tgt_txt)
            answers.append(example)
    return answers

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
max_epoch = 10
train_dataset = read_answers('/data/czwang/codesearch/code_defect/dataset/train.jsonl')
valid_dataset = read_answers('/data/czwang/codesearch/code_defect/dataset/valid.jsonl')
test_dataset = read_answers('/data/czwang/codesearch/code_defect/dataset/test.jsonl')


# plm, tokenizer, model_config, WrapperClass = load_plm("t5", "Salesforce/codet5-small")
model_config = T5Config.from_pretrained("Salesforce/codet5-small")
plm = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small", config=model_config)
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
WrapperClass = T5TokenizerWrapper

promptTemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='Defect: {"placeholder":"text_a"} {"mask"}.', using_decoder_past_key_values=False)

wrapped_example = promptTemplate.wrap_one_example(train_dataset[0]) 
print(wrapped_example)

train_data_loader = PromptDataLoader(
    dataset = test_dataset,
    teacher_forcing=True,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=16,
    predict_eos_token=True,
    decoder_max_length=32
)
# valid_data_loader = PromptDataLoader(
#     dataset = valid_dataset,
#     tokenizer = tokenizer, 
#     template = promptTemplate,
#     tokenizer_wrapper_class=WrapperClass,
#     batch_size=16,
#     decoder_max_length=3
# )
test_data_loader = PromptDataLoader(
    dataset = test_dataset,
    tokenizer = tokenizer,
    template = promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    batch_size=32,
    predict_eos_token=True,
    decoder_max_length=3
)


use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=promptTemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=True)
if use_cuda:
    prompt_model=  prompt_model.cuda()

optimizer_grouped_parameters = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-2)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step  = len(train_data_loader)*max_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)
# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
def classification_metric(generated_sentence, groundtruth_sentence):
    print(generated_sentence[0:5])
    print(groundtruth_sentence[0:5])

    sum = 0
    N = len(groundtruth_sentence)
    for gene_s, gt_s in zip(generated_sentence, groundtruth_sentence):
        if gene_s == gt_s:
            sum+=1
    return sum/N
# Define evaluate function 
def evaluate(prompt_model, dataloader, epoch=None):
    print(" Evaluating ************** epoch {}************".format(epoch))
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()
    for step, inputs in tqdm(enumerate(dataloader)):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = classification_metric(generated_sentence, groundtruth_sentence)
    if epoch:
        print("In epoch {} test_score".format(epoch), score, flush=True)
    else:
        print("test_score", score, flush=True)


generation_arguments = {
    "max_length": 128,
    "max_new_tokens": None,
    "min_length": 1,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": None
}

global_step = 0 
tot_loss = 0 
log_loss = 0
# training and generation.
tot_loss = 0 
for epoch in range(max_epoch):
    print(" Training ************** epoch {}************".format(epoch))
    for step, inputs in enumerate(train_data_loader):
        global_step +=1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(promptTemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step %100 ==0: 
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss
evaluate(prompt_model, test_data_loader, epoch)
