from pprint import pprint
import boto3
import json

client = boto3.client('cloudformation')

#r = client.list_stack_resources(StackName='arn:aws:cloudformation:us-east-1:544719615594:stack/poem-load/7bc7b4f0-51dc-11e7-8998-50d5ca632682')
r = client.list_stack_resources(StackName='poem09')

r1 = client.list_stacks(StackStatusFilter=[
    'CREATE_IN_PROGRESS',
    'CREATE_FAILED',
    'CREATE_COMPLETE',
    'ROLLBACK_IN_PROGRESS',
    'ROLLBACK_FAILED',
    'ROLLBACK_COMPLETE',
    'DELETE_IN_PROGRESS',
    'DELETE_FAILED',
    #'DELETE_COMPLETE',
    'UPDATE_IN_PROGRESS',
    'UPDATE_COMPLETE_CLEANUP_IN_PROGRESS',
    'UPDATE_COMPLETE',
    'UPDATE_ROLLBACK_IN_PROGRESS',
    'UPDATE_ROLLBACK_FAILED',
    'UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS',
    'UPDATE_ROLLBACK_COMPLETE',
    'REVIEW_IN_PROGRESS'])

pprint(r)
