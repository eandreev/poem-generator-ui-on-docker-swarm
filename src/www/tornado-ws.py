import os, sys, inspect
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)) + '/../common')

from concurrent.futures import ThreadPoolExecutor
from tornado.log import LogFormatter, access_log
from redisqueue import TextGenQueue
from ratelimiter import RateLimiter
from pprint import pformat
from tornado import gen
import tornado.options
import tornado.ioloop
import tornado.web
import tornado
import logging
import json
import time


def send_poem_request(input_string):
    return q.create(input_string)

def get_poem_status(task_id):
    return q.get_task_result(task_id)

def check_poem_gen_limit(ip, max_allowed):
    return limiter.check_ip(ip, max_allowed)


class MainHandler(tornado.web.RequestHandler):
    def get_template_path(self):
        # logging.warning(str(globals()))
        return os.path.dirname(os.path.realpath(__file__)) + '/templates'

    def get(self):
        self.render("index.html", remote_host=self.request.host) #, title="My title", items=items)

class DbgHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('<pre>')
        self.write(self.request.remote_ip)

class PoemAjaxHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def post(self):
        check_result = yield executor.submit(check_poem_gen_limit, self.request.remote_ip, 3)
        if not check_result:
            self.set_header('Content-Type', 'application/json; charset="utf-8"')
            self.write(json.dumps({'poem': 'Не больше трёх запросов в минуту!'}))
            return

        io_loop = tornado.ioloop.IOLoop.current()
        result = ''
        task_id = yield executor.submit(send_poem_request,
            '''тебе но голос музы тёмной 
коснется ль уха твоего
поймешь ли ты душою скромной
стремленье сердца моего

'''
            )
        for i in range(10):
            yield gen.Task(io_loop.add_timeout, io_loop.time() + 1)
            r = yield executor.submit(get_poem_status, task_id)
            if r is not None:
                result = r
                break
        
        self.set_header('Content-Type', 'application/json; charset="utf-8"')
        self.write(json.dumps({'poem': result}))

def log_request(handler):
    if handler.get_status() < 400:
        log_method = access_log.info
    elif handler.get_status() < 500:
        log_method = access_log.warning
    else:
        log_method = access_log.error
    request_time = 1000.0 * handler.request.request_time()
    log_method('%d %s %.2fms', handler.get_status(),
                handler._request_summary(), request_time)
    #log_method('%s', pformat(handler.request))

def make_app():
    static_dir_path = os.path.dirname(os.path.realpath(__file__)) + '/static';

    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/dbg", DbgHandler),
        (r"/ajax/poem", PoemAjaxHandler),
        (r'/(favicon.ico)', tornado.web.StaticFileHandler, { 'path': static_dir_path }),
        (r'/static/(.*)', tornado.web.StaticFileHandler, { 'path': static_dir_path })
    ],
    debug=True,
    log_function=log_request)

if __name__ == "__main__":
    q = TextGenQueue('poem', 'redis')
    limiter = RateLimiter('poem', 'redis')

    # see: https://stackoverflow.com/questions/32567124/use-concurrent-futures-with-tornado-event-loop
    executor = ThreadPoolExecutor(128)

    app = make_app()

    # Enbale logging
    #Let's not use tornado.log.enable_pretty_logging() tornado.options.parse_command_line()
    
    # global loger
    logger = logging.getLogger()
    logger.setLevel('INFO')
    channel = logging.StreamHandler()
    channel.setFormatter(LogFormatter(color=False))
    logger.addHandler(channel)

    # access log
    access_log_path = '/var/log/supervisor/tornado-access.log'
    logging.getLogger("tornado.access").propagate = False # don't let it go to the root logger; see https://stackoverflow.com/questions/43290131/tornado-file-log-is-also-write-to-stdout/43295380
    logging.getLogger("tornado.access").addHandler(logging.handlers.WatchedFileHandler(access_log_path))
    
    #logging.warning(str(tornado.options.options.items()))

    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
