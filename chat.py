import random
import json
import torch
import os
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from getWeather import weather
from unidecode import unidecode
from tom_tat import tom_tat_van_ban
from acronym.stand_words import normalize_text, dictions
from g4f.client import Client
import re

models="gpt-3.5-turbo"
# models="gemini"

def chat_with_ai(user_input, messages):
    client = Client()
    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model=models, messages=messages
    )
    ai_response = response.choices[0].message.content

    ai_response = re.sub(r'[#*\']', '', ai_response)

    messages.append({"role": "assistant", "content": ai_response})
    return ai_response, messages

messages = [{"role": "user", "content": 'xin chào'}]


# Chọn thiết bị chạy mô hình PyTorch 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

intents = {"intents": []}

# Lặp qua tất cả các file JSON trong thư mục "stories" và mở các file json trong đó
folder_path = "stories"
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            intents_data = json.load(file)
            intents["intents"].extend(intents_data["intents"])

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "TIMI"
waiting_for_summary = False

def chat(user_input):
    global waiting_for_summary
    # Xử lý input từ người dùng
    processed_input = unidecode(user_input).lower()
    processed_input = normalize_text(processed_input, dictions)
    processed_input = tokenize(processed_input)
    
    X = bag_of_words(processed_input, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Nếu đang trong trạng thái chờ tóm tắt
    if waiting_for_summary:
        if prob.item() > 0.95:
            waiting_for_summary = False
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        summary = tom_tat_van_ban(user_input)
        waiting_for_summary = False
        return summary

    # Nếu xác suất dự đoán cao hơn ngưỡng (0.95), chọn phản hồi tương ứng
    if prob.item() > 0.96:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag == "tom_tat": 
                    waiting_for_summary = True
                    return random.choice(intent['responses'])
                elif tag == 'weather':
                    return weather(user_input)
                else:
                    return random.choice(intent['responses'])
    else:
        a = chat_with_ai(user_input, messages)
        return a[0]