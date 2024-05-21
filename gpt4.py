from g4f.client import Client
import re

models="gemini"

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
# Khởi tạo danh sách messages ban đầu
# print(chat_with_ai('thông tin trường đại học mỏ địaa chất',messages)[0])