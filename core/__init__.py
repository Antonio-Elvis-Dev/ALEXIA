import datetime
from multiprocessing.connection import answer_challenge

class SystemInfo:
    def __init__():
        pass
    
    @staticmethod
    def get_time():
        now = datetime.datetime.now()
        answer = 'São {} horas e {} minutos.'.format(now.hour, now.minute)
        return answer