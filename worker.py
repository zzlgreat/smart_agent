#fuction calling api
import requests
import json
from real_world import toolkit
# data2 = {'user_prompt': '我想吃手抓羊肉了！'}
# data3 = {'user_prompt': 'I have had a headache for several consecutive days, I need a doctor'}
# data4 = {'user_prompt': 'Write sql in a table called "stocklist" in my mysql, and the "scode" is "000001.SZ"'}
# data5 = {'user_prompt': "I want to see videos about female cosers' dancing!"}
def work4plan(plan):
    data = {'user_prompt': plan}
    response = requests.post('http://192.168.1.24:7784/stream_with_function', json=data)
    if response.status_code == 200:
        # print('My command:')
        # print(data.get('user_prompt'))
        # print("Response from /stream_with_function:")
        print(json.loads(response.text)['result'].split('[/INST]')[-1].strip())
        try:
            func = json.loads(json.loads(response.text)['result'].split('[/INST]')[-1].strip())
            return toolkit.scheduler(func)
        except:
            return json.loads(response.text)['result'].split('[/INST]')[-1].strip()
        # func = json.loads(json.loads(response.text)['result'].split('[/INST]')[-1].strip())
        # return toolkit.scheduler(func)


    else:
        print(f"Failed to get a response: {response.status_code}")


if __name__ == '__main__':
    print(work4plan('''Analyze key factors contributing to longevity\nExamine the reasons behind Xi Jinping\'s continued hold on power based on the gathered information from Step 1. Some possible factors could include economic growth, foreign policy successes, or strong support within the Communist Party of China.\n\nStep 3: Compare with other leaders\nSearch for comparisons between Xi Jinping and other world leaders using search_wiki("comparison of Xi Jinping with other world leaders"). This comparison might provide insights into what sets him apart and contributes to his enduring leadership. Store this information in'''))