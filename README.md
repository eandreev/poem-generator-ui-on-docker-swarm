# Poem Generator UI

### Files

* **Model:** [`/model/wordbased-pushkin-big.py`](/model/wordbased-pushkin-big.py)
* Docker files:
	* Text generator worker: [`/Dockerfile.worker`](Dockerfile.worker)
	* Webserver: [`/Dockerfile.www`](Dockerfile.www)
* Text generator code: [`/src/worker/poem-worker.py`](src/worker/poem-worker.py)
* Webserver code: [`/src/www/tornado-ws.py`](src/www/tornado-ws.py)
* AWS CloudFormation template: [`/swarm-env/poem-spot-fleet.json`](swarm-env/poem-spot-fleet.json)
* Docker Swarm stack (app) description: [`swarm-env/swarm-prod.yml`](/swarm-env/swarm-prod.yml)

### Sample Poetry

```
 и вот и сон волшебный трон 
 любовь и радость и покой 
 и в час когда в сей день разлуки 
 в объятиях твоих страстей 
 я не хочу быть может быть 
 но я желал бы в тишине 
 в кругу красавиц молодых 
 я б знал в минуту и в покое 
 я пел в душе моей любви 
```

```
 и в этот век родной гроза 
 целует свой алтарь суровый 
 и точно так и всё сказать 
 что полно смертный твой державин 
 люби любви свой стон звучал 
 и скоро ли и прежде боле 
 тогда ль бывало в час горой 
```

```
 теперь едва живой собаки нету 
 час пред очами будет миру ты лежит 
 хозяин сам едва не будет пол для славы 
 нашел опять певца героя всей весне 
 ах скорбь которую тогда еще ни взгляд 
 слова одной сыны не хочет верой 
 но нынче он открыл тогда чужой 
 любовь не светят все ее как раз 
 но вновь гранит стучат о нет совет 
 где тени шли над ними ночи степь 
 и на заре грудь славы взор дороже 
```