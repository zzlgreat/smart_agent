#fuction calling api
import requests
import json
from real_world import toolkit
from send_req import send2llm
from model_api_config import planner_model,distributor_model
def work4plan(plan):
    data = {'user_prompt': plan}
    response = requests.post(distributor_model.URI, json=data)
    if response.status_code == 200:
        #print(response.text)
        print(json.loads(response.text)['result'].split('[/INST]')[-1].strip())
        try:
            func = json.loads(json.loads(response.text)['result'].split('[/INST]')[-1].strip())
            return toolkit.scheduler(func)
        except Exception as e:
            print(e)
            return json.loads(response.text)['result'].split('[/INST]')[-1].strip()

    else:
        print(f"Failed to get a response: {response.status_code}")

if __name__ == '__main__':
    print(work4plan('Identify the director of "Killers of the Flower Moon" using search_wiki or search_bing.\n'))