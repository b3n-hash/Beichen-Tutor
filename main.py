from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
response = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":"say hi"}])
print(response.choices[0].message.content)
