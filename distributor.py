#fuction calling api
import requests
import json
from real_world import toolkit
from send_req import send2llm
from model_api_config import planner_model,distributor_model
def work4plan(plan):
    # func_flag to judge if the plan needs func
    func_flag = False
    for tool in toolkit.toolkits:
        if tool.get('function') in plan:
            func_flag =True
    if func_flag:
        data = {'user_prompt': plan}
        response = requests.post(distributor_model.URI, json=data)
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
    else:
        result = send2llm(plan, planner_model.URI)
        return result

if __name__ == '__main__':
    print(work4plan('Identify the director of "Killers of the Flower Moon" using search_wiki or search_bing.\n'))