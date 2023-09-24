from solver import solve_plan_work
# example:
#user_prompt = "what movies did the director of 'Killers of the Flower Moon' direct？ List one of them and search it in bilibili."
# input
user_prompt = input("Please enter your query: ")
user_prompt += "\n"
solve_plan_work(user_prompt)
