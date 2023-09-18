import json
class real_world():
    def get_config(self):
        with open('config.json', 'r') as fp:
            self.config = json.load(fp)
            return self.config
    

if __name__ == '__main__':
    print(real_world().get_config().get('toolkits'))