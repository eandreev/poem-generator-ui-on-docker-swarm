from uuid import uuid4
import codecs
import redis
import time
import logging

class RateLimiter:
    def __init__(self, namespace, hostname):
        self.namespace = namespace
        self.__db = redis.Redis(hostname)
    
    def check_ip(self, ip_addr, max_allowed):
        dt = time.strftime('%H%M')
        key = self.namespace + ':' + dt + ':' + ip_addr
        cnt = int(self.__db.incr(key))
        if 1 == cnt:
            self.__db.expire(key, 120)
        return cnt <= max_allowed