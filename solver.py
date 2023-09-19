import planner
import re
import worker
from send_req import send2llm
HOST = '192.168.1.24:5000'
URI = f'http://{HOST}/api/v1/generate'
def solve_plan_work(prompt):
    SYS_PROMPT = '''The base task is:'''+prompt+'\n'
    WORK_PROMPT = ''
    plans = planner.get_plan(prompt)
    for num,plan in enumerate(plans):
        pattern = r'(\w+\(")(#[A-Za-z0-9]+)("\))'
        plan_mod = plan
        if bool(re.search(pattern, plan)):
            plan_mod = re.sub(pattern, result, plan)
        print("The plan "+str(num)+" is:")
        print(plan)
        SYS_PROMPT= SYS_PROMPT+"step "+str(num)+":"+plan+'\n'
        # print("The WORK_PROMPT " + str(num) + " is:")
        # print(WORK_PROMPT)
        result = worker.work4plan(plan_mod)
        SYS_PROMPT = SYS_PROMPT + "And the result is:"+result+"\n"

    SYS_PROMPT=SYS_PROMPT+'\nBase on above, make a conclusion of the base task.'
    print("________________________")
    print(SYS_PROMPT)
    response = send2llm(SYS_PROMPT,URI)
    print(response)
if __name__ == '__main__':
    user_prompt = "what movies did the the director of 'Killers of the Flower Moon' direct? List one of them and search it in bilibili.\n"
    solve_plan_work(user_prompt)