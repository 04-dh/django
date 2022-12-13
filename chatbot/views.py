from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt

import torch
import torch.nn as nn
import random

from .model.classifier import KoBERTforSequenceClassfication
from kobert_transformers import get_tokenizer

# Create your views here.
def home(request):
    context = {}

    return render(request, "chatbot/chathome.html", context)

@csrf_exempt
def chatanswer(request):
    context = {}

    questext = request.GET['questext']
    
    def load_wellness_answer():
        root_path = "C:/dev2/bigproject/chatbot/model"
        category_path = f"{root_path}/wellness_dialog_category.txt"
        answer_path = f"{root_path}/wellness_dialog_answer.txt"

        c_f = open(category_path, 'r', encoding='UTF8')
        a_f = open(answer_path, 'r', encoding='UTF8')

        category_lines = c_f.readlines()
        answer_lines = a_f.readlines()

        category = {}
        answer = {}
        for line_num, line_data in enumerate(category_lines):
            data = line_data.split('    ')
            category[data[1][:-1]] = data[0]

        for line_num, line_data in enumerate(answer_lines):
            data = line_data.split('    ')
            keys = answer.keys()
            if (data[0] in keys):
                answer[data[0]] += [data[1][:-1]]
            else:
                answer[data[0]] = [data[1][:-1]]

        return category, answer
    
    def kobert_input(tokenizer, str, device=None, max_seq_len=512):
        index_of_words = tokenizer.encode(str)
        token_type_ids = [0] * len(index_of_words)
        attention_mask = [1] * len(index_of_words)

        # Padding Length
        padding_length = max_seq_len - len(index_of_words)

    # Zero Padding
        index_of_words += [0] * padding_length
        token_type_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        data = {
            'input_ids': torch.tensor([index_of_words]).to(device),
            'token_type_ids': torch.tensor([token_type_ids]).to(device),
            'attention_mask': torch.tensor([attention_mask]).to(device),
        }
        return data

    
    def chat3(inp):
        category, answer = load_wellness_answer()

        ctx = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(ctx)
        save_ckpt_path = 'C:/dev2/bigproject/chatbot/checkpoint/kobert-wellness-text-classification.pth'
        # 저장한 Checkpoint 불러오기
        checkpoint = torch.load(save_ckpt_path, map_location=device)

        model = KoBERTforSequenceClassfication()
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        model.to(ctx)
        model.eval()

        tokenizer = get_tokenizer()
        
        data = kobert_input(tokenizer, inp, device, 512)

        output = model(**data)

        logit = output[0]
        softmax_logit = torch.softmax(logit, dim=-1)
        softmax_logit = softmax_logit.squeeze()

        max_index = torch.argmax(softmax_logit).item()
        max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

        answer_list = answer[category[str(max_index)]]
        answer_len = len(answer_list) - 1
        answer_index = random.randint(0, answer_len)
        
        return answer_list[answer_index]

    anstext = chat3(questext)
    print(anstext)

    context['anstext'] = anstext
    context['flag'] = '0'

    return JsonResponse(context, content_type="application/json")