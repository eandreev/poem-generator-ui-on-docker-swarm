#!/bin/bash

url='https://cloudformation.us-east-1.amazonaws.com/'\
'?Action=ListStackResources'\
'&StackName=MyStack'\
'&Version=2010-05-15'\
'&SignatureVersion=2'\
'&Timestamp=2011-07-08T22%3A26%3A28.000Z'\
'&AWSAccessKeyId=AKIAJC5JJZTK3ZKE2ETA'\
'&Signature=8WGJ7Nm2QrARLl33a6lrvjkM1Dc0XoqxSsN47DUm'

 curl $url
 