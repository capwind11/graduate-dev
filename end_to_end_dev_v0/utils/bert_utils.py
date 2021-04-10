import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

MAX_SEN_LEN = 64

# 加载bert模型
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained("D://bert_pytorch//bert-base-uncased-vocab.txt")
    model = BertModel.from_pretrained("D://bert_pytorch//bert-base-uncased//bert-base-uncased")
    model.eval()
    model.to('cuda')
    return model,tokenizer

# 加载模板
def load_log_template(filename):
    log_templates = []
    with open(filename) as f:
        for line in f:
            log_templates.append(line)
    return log_templates

# 建立从日志id到bert编码的映射
def map_eventid_to_bert(log_templates,model,tokenizer):
    template_len_list = []
    tokens_idx_list = []
    label_list = []
    count = 1
    for log in log_templates:
        labels = log.split()[0]
        log = ' '.join(log.split())
        tokens = tokenizer.tokenize(log)
        label_list.append(count)
        count += 1
        tokens.insert(0, '[CLS]')
        tokens.append('[SEP]')
        token_idx = tokenizer.convert_tokens_to_ids(tokens)
        tokens_idx_list.append(token_idx)
        # print("tokenzie后日志条目:")
        # print(tokens)
        # print("长度为" + str(len(tokens)))
        template_len_list.append(len(tokens))
    MAX_SEN_LEN = max(template_len_list)
    bert_input = []
    for tokens in tokens_idx_list:
        input_mask = [1] * len(tokens)
        if len(tokens) < MAX_SEN_LEN:
            input_mask.extend([0] * (MAX_SEN_LEN - len(tokens)))
            tokens.extend([0] * (MAX_SEN_LEN - len(tokens)))
        #         print(input_mask)
        segment_ids = [0] * len(tokens)
        token_tensor = torch.tensor([tokens]).to('cuda')
        input_mask_tensor = torch.tensor([input_mask]).to('cuda')
        segment_ids_tensor = torch.tensor([segment_ids]).to('cuda')
        bert_input.append([token_tensor, input_mask_tensor, segment_ids_tensor])
        # print(bert_input[-1])
    bert_output = []
    with torch.no_grad():
        for param in bert_input:
            hidden, pool = model(param[0], param[1], param[2])
            bert_output.append([hidden[-2], pool])
    eventId_to_bert = {}
    for i in range(len(label_list)):
        label = label_list[i]
        eventId_to_bert[label] = bert_output[i]
    torch.save(eventId_to_bert, "eventId_to_bert.pth")
    return eventId_to_bert

def saveMap(eventId_to_bert,fileName):
    torch.save(eventId_to_bert, fileName)
    return

def build_bert_cache(filename,outputPath):
    logTemplates = load_log_template(filename)
    model,tokenizer = load_bert_model()
    event_to_map = map_eventid_to_bert(logTemplates,model,tokenizer)
    saveMap(event_to_map, outputPath+'bert_cache.pth')

    
