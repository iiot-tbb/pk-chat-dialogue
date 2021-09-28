import json

###整理数据格式把三元组放在一起

with open("./example.json") as f:
    a=json.load(f)
    goal = a['goal']
    knowledge  = a['knowledge']
    conversation = a['conversation']

    goal_knowledge = [' '.join(list(map(lambda x:x.strip(),spo))) for spo in goal + knowledge]

    conversation = a["conversation"]

    #for i in range(0, len(conversation), 2):
    sample = { 
              "knowledge": goal_knowledge,
              "context": conversation,
              }

    f = open("./example2.json",'w')
    json.dump(sample,f,indent=4)
    f.close()
