import json

input_file = "dataset/pretrain_hq.jsonl"

output_file = "dataset/sample.jsonl"

num_lines = 100  # 取前100条

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:

    

    for i, line in enumerate(fin):

        if i >= num_lines:

            break

        

        # 可选：校验是否是合法 JSON

        obj = json.loads(line)

        

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Done!")
