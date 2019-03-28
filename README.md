# Poem Generator UI

### Files

* Docker files:
	* Text generator worker: [`/Dockerfile.worker`](Dockerfile.worker)
	* Webserver: [`/Dockerfile.www`](Dockerfile.www)
* Text generator code: [`/src/worker/poem-worker.py`](src/worker/poem-worker.py)
* Webserver code: [`/src/www/tornado-ws.py`](src/www/tornado-ws.py)
* AWS CloudFormation template: [`/swarm-env/poem-spot-fleet.json`](swarm-env/poem-spot-fleet.json)
* Docker Swarm stack (app) description: [`swarm-env/swarm-prod.yml`](/swarm-env/swarm-prod.yml)