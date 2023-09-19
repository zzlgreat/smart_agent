import asyncio
import json
import wikipediaapi
import datetime
from datetime import datetime
from pytz import timezone
from bilibili_api import video,search, sync, video_zone
import requests

# functions
toolkits = json.load(open('real_world/config.json','r')).get('toolkits')


async def search_by_order(keyword):
    return await search.search(keyword)

# Get the timezone's time
def get_current_time(tz=None):
    if tz is None:
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        now_time = datetime.now(timezone(tz))
        return now_time.strftime('%Y-%m-%d %H:%M:%S')

# search from bing and return the top 10 results

def search_bing(keyword):
    r = requests.get('http://192.168.1.24:8000/search?q='+keyword+'&max_results=10').content.decode('unicode-escape')
    #print(r)
    r = r.replace('It\'s a "Barbie" world.', 'It\'s a \\"Barbie\\" world.')
    resultstr = ''
    results = json.loads(r).get('results')
    for num,res in enumerate(results):
        resultstr+= 'result' +str(num) +' : Title: '+ res.get('title')+'\n'+"Content: "+res.get('body')+'\n'
    return resultstr

# search bilibili and return the top 20 urls

def search_bilibili(keyword):
    res = sync(search_by_order(keyword))
    #print(res.keys())
    arcurls = []
    for result in res.get('result'):
        #print(result)
        if result.get('result_type') == 'video':
            for data in result.get('data'):
                arcurls.append(data.get('arcurl'))
    print('searching from bilibili........')
    return '\n'.join(arcurls)

#search from wikipedia and return a summary

def search_wiki(keyword):
    wiki = wikipediaapi.Wikipedia('zzlgreatBot/0.1 (https://metaease.ai/; zzlgreat@gmail.com) generic-library/0.1', proxies={'http': 'http://192.168.1.7:10811','https': 'http://192.168.1.7:10811'})
    page = wiki.page(keyword)
    print('searching from wikipedia.......')
    return page.summary

#make a file, and return the result whether it is done

def make_file(filename,filetype,path):
    try:
        with open(path + filename + '.' + filetype, 'w') as j:
            j.write(filename)
        return "The file has made"
    except Exception as e:
        return "The file cannot made cause:" +str(e)


def scheduler(func_dic):
    function_name = func_dic.get('function')
    arguments = func_dic.get('arguments')
    print(function_name,arguments)
    if function_name == 'get_current_time':
        return get_current_time(arguments['query'])
    elif function_name == 'search_bing':
        return search_bing(arguments['query'])
    elif function_name == 'search_bilibili':
        return search_bilibili(arguments['query'])
    elif function_name == 'search_wiki':
        return search_wiki(arguments['query'])
    elif function_name == 'make_file':
        return make_file(arguments['filename'],
                           arguments['filetype'],
                           arguments['path'])
    return 'Sorry, I have no functions now'

if __name__ == '__main__':
    print(search_bilibili('The Departed'))