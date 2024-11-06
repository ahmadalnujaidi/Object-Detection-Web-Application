import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyCWQVBBgIRIowkV3aBD1UxaNfuqLyiT1q0")

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("dsecribe saudi arabia in 5 words.")
print(response.text)