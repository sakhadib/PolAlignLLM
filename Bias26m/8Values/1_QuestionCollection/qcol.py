import json
import csv

json_path = 'd:/Research/Political_Bias_of_LLMs/8Values/1_QuestionCollection/questions.json'
csv_path = 'd:/Research/Political_Bias_of_LLMs/8Values/1_QuestionCollection/questions.csv'

with open(json_path, 'r', encoding='utf-8') as f:
    questions = json.load(f)

with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['question_id', 'question_text'])
    for idx, q in enumerate(questions, start=1):
        writer.writerow([f'q{idx}', q['question']])