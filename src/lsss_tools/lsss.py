"""Provide convenience functions for accessing the LSSS API.

This file is a copy of that provide by LSSS and intended for use when using the LSSS API. However
when running API-using scripts that are not run directly by LSSS, this file is not conveniently
available, so it is provided to all of Python via a package/module.
"""

import json
import os
import requests

baseUrl = 'http://127.0.0.1:' + os.environ.get('LSSS_SERVER_PORT', '8000')
input = json.loads(os.environ.get('LSSS_INPUT', '{}'))


def get(path, params=None):
    url = baseUrl + path
    response = requests.get(url, params=params)
    if response.status_code == 200:
        if response.headers['Content-Type'] == 'application/json':
            return response.json()
        return response.text
    raise ValueError(url + ' returned status code ' + str(response.status_code) + ': ' + response.text)


def post(path, params=None, json=None, data=None):
    url = baseUrl + path
    response = requests.post(url, params=params, json=json, data=data)
    if response.status_code == 200:
        if response.headers['Content-Type'] == 'application/json':
            return response.json()
        return response.text
    if response.status_code == 204:
        return None
    raise ValueError(url + ' returned status code ' + str(response.status_code) + ': ' + response.text)


def delete(path, params=None):
    url = baseUrl + path
    response = requests.delete(url, params=params)
    if response.status_code == 200:
        return None
    raise ValueError(url + ' returned status code ' + str(response.status_code) + ': ' + response.text)
