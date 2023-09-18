import planner
import worker
def solve_plan_work(prompt):
    SYS_PROMPT = '''The base prompt is:'''+prompt
    plans = planner.get_plan(prompt)
    for num,plan in enumerate(plans):
        print("The plan is:")
        print(num,plan)
        SYS_PROMPT= SYS_PROMPT+"step "+str(num)+":"+plan+'\n'
        result = worker.work4plan(plan)
        SYS_PROMPT = SYS_PROMPT + "And the result is:"+result+"\n"
    print(SYS_PROMPT)
if __name__ == '__main__':
    solve_plan_work("what movies did the the director of 'Oppenheim' direct? List top 10 best.")