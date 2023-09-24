import planner
import re
import distributor
from send_req import send2llm
from model_api_config import planner_model, distributor_model
def solve_plan_work(prompt):
    SYS_PROMPT = '''The base task is:'''+prompt+'\n'
    plans = planner.get_plan(prompt)
    for num,plan in enumerate(plans):
        pattern = r'(\w+\(")(#[A-Za-z0-9]+)("\))'
        plan_mod = plan
        # judge if last result in this plan
        if bool(re.search(pattern, plan)):
            plan_mod = re.sub(pattern, result, plan)
        print("The plan "+str(num)+" is:")
        print(plan)
        SYS_PROMPT= SYS_PROMPT+"step "+str(num)+":"+plan+'\n'
        result = distributor.work4plan(plan_mod)
        SYS_PROMPT = SYS_PROMPT + "And the result is:"+result+"\n"

    SYS_PROMPT=SYS_PROMPT+'\nBase on above, make a conclusion of the base task.'
    print("________________________")
    print(SYS_PROMPT)
    response = send2llm(SYS_PROMPT,planner_model.URI)
    print(response)
if __name__ == '__main__':
    user_prompt = "Recommend me a guangzhou's hotel, I want the best in this year. Not the most expensive but the best.\n"
    solve_plan_work(user_prompt)