#!/usr/bin/env python
# -*- coding: utf-8 -*-


import urllib2, requests
import json
import httplib


class OpenStackClient():

    def __init__(self, start_url, user, password):
        self.start_url = start_url
        self.user = user
        self.password = password
        self.headers = {
            "Content-type" : "application/json",
            "Accept" : "application/json",
        } 
        self.generate_token()

    def rest_api(self, url, headers=None, params=None, method=None):
        # print(url)
        tmp_headers = self.headers
        if None != headers:
            for key, value in headers.items():
                tmp_headers[key] = value
        if None != params:
            tmp_params = json.dumps(params)
            request = urllib2.Request(url, data=tmp_params, headers=tmp_headers)
        else:
            request = urllib2.Request(url, headers=tmp_headers)
        if None != method:
            request.get_method = lambda : method
        try:
            response = urllib2.urlopen(request, timeout=30)
        except urllib2.HTTPError as err:
            # if 300 == err.code:
            ret_headers = err.headers
            ret_data = err.read()
            # print(err.code)
        except httplib.BadStatusLine as err:
            ret_headers = None
            msg["Request(Time Out) URL"] = url
            ret_data = json.loads(msg)
        except urllib2.URLError as err:
            ret_headers = None
            msg["URL ERROR"] = url
            ret_data = json.loads(msg)
        else:
            ret_headers = response.headers
            ret_data = response.read()
        return ret_headers, ret_data

    def generate_token(self):
        headers, data = self.rest_api(self.start_url)
        objdata = json.loads(data)
        api_version_list = list(objdata["versions"]["values"])
        api_version_list.sort(key=lambda info : info["id"], reverse=True)
        self.identity_root_url = api_version_list[0]["links"][0]["href"]
        url = self.identity_root_url + "auth/tokens"
        params = {
            "auth" : {
                "identity" : {
                    "methods" : ["password"],
                    "password" : {
                        "user" : {
                            "domain" : {
                                "name" : "Default"
                            },
                        "name" : self.user,
                        "password" : self.password
                        }
                    }
                },
                "scope" : {
                    "project" : {
                        "domain" : {
                            "name" : "Default"
                        },
                        "name" : "zhaojiangbo"
                    }
                }
            }
        }
        headers, data = self.rest_api(url, params=params)
        header = dict(headers)
        self.token = header["x-subject-token"]
        self.headers["X-Auth-Token"] = self.token
        objdata = json.loads(data)
        for element  in objdata["token"]["catalog"]:
            # print(element ["type"])
            if element ["type"] == "compute":
                for endpoint in element["endpoints"]:
                    if endpoint["interface"] == "public":
                        self.compute_root_url = endpoint["url"] + "/"
            if element ["type"] == "volumev2":
                for endpoint in element["endpoints"]:
                    if endpoint["interface"] == "public":
                        self.volume_root_url = endpoint["url"] + "/"

    def get_vm_list(self):
        url = self.compute_root_url + "servers"
        headers, data = self.rest_api(url)
        objdata = json.loads(data)
        return objdata
    
    def get_vm_info(self, server_id):
        url = self.compute_root_url + "servers/" + server_id
        headers, data = self.rest_api(url)
        objdata = json.loads(data)
        return objdata

    def create_vm(self, name="new-server", imageRef="f2ae64ff-f371-4609-a473-acfbd33adb99", flavorRef="2", networks="a69630d2-f1eb-4ad8-9add-486159ffdc78"):
        params = {
            "server" : {
                "name" : name,
                "imageRef" : imageRef,
                "flavorRef" : flavorRef,
                "networks" : [{"uuid": networks}]
            }
        }
        url = self.compute_root_url + "servers"
        headers, data = self.rest_api(url, params=params)
        objdata = json.loads(data)
        return objdata
    
    def delete_vm(self, server_id):
        url = self.compute_root_url + "servers/" + server_id
        self.rest_api(url, method="DELETE")

    def start_vm(self, server_id):
        params = {
            "os-start": None
        }
        url = self.compute_root_url + "servers/" + server_id + "/action"
        self.rest_api(url, params=params)

    def stop_vm(self, server_id):
        params = {
            "os-stop": None
        }
        url = self.compute_root_url + "servers/" + server_id + "/action"
        self.rest_api(url, params=params)

    def get_vol_list(self):
        url = self.volume_root_url + "volumes"
        headers, data = self.rest_api(url)
        objdata = json.loads(data)
        return objdata

    def get_vol_info(self, vol_id):
        url = self.volume_root_url + "volumes/" + vol_id
        headers, data = self.rest_api(url)
        objdata = json.loads(data)
        return objdata

    def create_vol(self, size=10):
        params = {
            "volume": {
                "size": size
            }
        }
        url = self.volume_root_url + "volumes"
        headers, data = self.rest_api(url, params=params)
        objdata = json.loads(data)
        return objdata
    
    def delete_vol(self, vol_id):
        url = self.volume_root_url + "volumes/" + vol_id
        self.rest_api(url, method="DELETE")

    def attach_vol(self, server_id, vol_id):
        params = {
            "volumeAttachment": {
                "volumeId": vol_id,
            }
        }
        url = self.compute_root_url + "servers/" + server_id + "/os-volume_attachments"
        headers, data = self.rest_api(url, params=params)
        objdata = json.loads(data)
        return objdata

    def detach_vol(self, server_id, vol_id):
        url = self.compute_root_url + "servers/" + server_id + "/os-volume_attachments/" + vol_id
        self.rest_api(url, method="DELETE")

    def my_get(self, url):
        headers, data = self.rest_api(url)
        objdata = json.loads(data)
        return objdata