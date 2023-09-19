from send_req import send2llm
from real_world import toolkit
import re

def get_plan(user_prompt):
    toolkits = toolkit.toolkits
    HOST = '192.168.1.24:5000'
    URI = f'http://{HOST}/api/v1/generate'
    prompt = '''For the following tasks, make plans that can solve the problem step-by-step. For each plan, indicate which external tool together with tool input to retrieve evidence. You can store the evidence into a variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...) Tools can be one of the following:'''
    for tk in toolkits:
        tk_prompt = tk.get('function') + '[input]: ' + tk.get('description')
        prompt += tk_prompt
    prompt += '\n\n'
    prompt += user_prompt
    print(prompt)
    result = send2llm(prompt, URI)
    print(result)
    matches_with_hashes = re.findall(r'(Step \d+:|Plan \d+:)(.*?)(?=Step \d+:|Plan \d+:|$)', result,
                                     re.DOTALL)

    # Extracting only the desired text between the start and end markers
    cleaned_matches_with_hashes = [match[1].strip() for match in matches_with_hashes]
    print(len(cleaned_matches_with_hashes))
    print(cleaned_matches_with_hashes)
    return cleaned_matches_with_hashes

if __name__ == '__main__':
    user_prompt = "what movies did the the director of 'Barbie' direct? List one of them and search it in bilibili.\n"
    plans = get_plan(user_prompt)
    print(plans)

