from uuid import uuid4
import codecs
import redis

class TextGenQueue:
    def __init__(self, namespace, hostname):
        self.namespace = namespace
        self.__db = redis.Redis(hostname)
    
    def create(self, input_string):
        task_uid = str(uuid4())
        self.__db.hmset(self.__task_key(task_uid), {'start': input_string})
        self.__db.rpush(self.__queue_key(), task_uid)
        self.__db.expire(self.__task_key(task_uid), 60)
        return task_uid

    def get_next_task(self):
        task_uid = self.__db.lpop(self.__queue_key())
        if task_uid is None:
            return None, None, 'empty'
        task_uid = task_uid.decode("utf-8")
        #print( self.__task_key(task_uid))
        recs = self.__db.hmget(self.__task_key(task_uid), ['start'])
        if recs[0] is None:
            return None, None, 'stale'
        #print(recs)
        input_string = codecs.decode(recs[0], 'utf-8')
        return task_uid, input_string, 'ok'
    
    def store_task_result(self, task_uid, result):
        self.__db.hmset(self.__task_key(task_uid), {'result': result})
    
    def get_task_result(self, task_uid):
        r = self.__db.hmget(self.__task_key(task_uid), ['result'])
        if r[0] is not None:
            return codecs.decode(r[0], 'utf-8')
        return None
    
    def __task_key(self, task_uid):
        return self.namespace + ':task:' + task_uid
    
    def __queue_key(self):
        return self.namespace + ':queue'
