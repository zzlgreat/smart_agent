#test worker
import requests
import json
from real_world import toolkit
from model_api_config import planner_model,distributor_model
def work4plan(plan):
    data = {'user_prompt': plan}
    response = requests.post(distributor_model.URI, json=data)
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
    print(work4plan('''How does rnn work?\n'''))