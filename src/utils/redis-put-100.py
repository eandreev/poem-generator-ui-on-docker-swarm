import os, sys
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../common')

from redisqueue import TextGenQueue
import time

q = TextGenQueue('poem', 'redis')

st = '''я вас любил
а вы еще быть может
'''

task_ids = []
task_keys = {}
for i in range(100):
    task_id = q.create(st)
    task_ids.append(task_id)
    task_keys[task_id] = len(task_ids)

print(task_ids)

while True:
    complete = []

    for task_id in task_ids:
        r = q.get_task_result(task_id)
        if r is not None:
            #print(r)
            complete.append(task_id)

    task_ids = [i for i in task_ids if i not in complete]
    if len(complete) > 0:
        print('Completed:', ', '.join([str(task_keys[i]) for i in complete]) + ',', 'left:', len(task_ids))

    if len(task_ids) == 0:
        break

    print('sleeping...')
    time.sleep(2)

print('DONE')