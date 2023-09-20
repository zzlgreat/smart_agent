#fuction calling api
import requests
import json
from real_world import toolkit
def work4plan(plan):
    data = {'user_prompt': plan}
    response = requests.post('http://192.168.1.24:7784/stream_with_function', json=data)
    if response.status_code == 200:
        print(json.loads(response.text)['result'].split('[/INST]')[-1].strip())
        try:
            func = json.loads(json.loads(response.text)['result'].split('[/INST]')[-1].strip())
            return toolkit.scheduler(func)
        except Exception as e:
            print(e)
            return json.loads(response.text)['result'].split('[/INST]')[-1].strip()

    else:
        print(f"Failed to get a response: {response.status_code}")