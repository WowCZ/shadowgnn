import json

with open('json_data_full_v2.json', 'r') as f:
    a = json.load(f)

print(a[0])
b = [x['rule_label'] == x['model_result'] for x in a]
print(sum(b) / len(b))