class planner_model():
    HOST = '192.168.1.24:5000'
    URI = f'http://{HOST}/api/v1/generate'
class distributor_model():
    HOST = '192.168.1.24:7784'
    URI = f'http://{HOST}/stream_with_function'