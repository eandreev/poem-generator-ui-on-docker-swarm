import os, sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../common')

from redisqueue import TextGenQueue
import time

q = TextGenQueue('poem', 'redis')

task_id = q.create('''я вас любил
а вы еще быть может
''')

while True:
    r = q.get_task_result(task_id)
    if r is not None:
        print(r)
        break
    print('sleeping...')
    time.sleep(2)

print('DONE')
