import json
from datetime import datetime

# 히스토리 파일 읽기
input_file = r"C:\Users\LENOVO\.claude\history.jsonl"
output_file = r"C:\Users\LENOVO\Documents\GitHub\ml-th-IoT\claude_history.md"

questions = []

# JSONL 파일 읽기
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line.strip())
            display = data.get('display', '')
            timestamp = data.get('timestamp', 0)

            # timestamp를 날짜로 변환 (밀리초 단위)
            dt = datetime.fromtimestamp(timestamp / 1000)
            date_str = dt.strftime('%Y-%m-%d %H:%M:%S')

            questions.append({
                'date': date_str,
                'question': display
            })
        except json.JSONDecodeError:
            continue

# 마크다운 파일로 작성
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("# Claude Code 질문 히스토리\n\n")

    for item in questions:
        f.write(f"## {item['date']}\n\n")
        f.write(f"{item['question']}\n\n")
        f.write("---\n\n")

print(f"총 {len(questions)}개의 질문이 추출되었습니다.")
print(f"결과 파일: {output_file}")
