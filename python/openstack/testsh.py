#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, json, time
from myopenstackclient import OpenStackClient


g_cmd_str = '''
    help
    get_vm_list
    get_vm_info server_id
    create_vm
    delete_vm server_id
    start_vm server_id
    stop_vm server_id
    get_vol_list
    get_vol_info vol_id
    create_vol
    delete_vol vol_id
    attach_vol server_id vol_id
    detach_vol server_id vol_id
    test_create
    test_delete
    my_get url
'''

start_url = "http://192.168.11.201:5000"
user = "zhaojiangbo"
password = "ZJb920719"

def main():

    ret = None
    cmd = None
    data = None

    client = OpenStackClient(start_url, user, password)

    # try:
    cmd = sys.argv[1]
    if "get_vm_list" == cmd:
        data = client.get_vm_list()
    elif "get_vm_info" == cmd:
        server_id = sys.argv[2]
        data = client.get_vm_info(server_id)
    elif "create_vm" == cmd:
        data = client.create_vm()
    elif "delete_vm" == cmd:
        server_id = sys.argv[2]
        client.delete_vm(server_id)
    elif "start_vm" == cmd:
        server_id = sys.argv[2]
        client.start_vm(server_id)
    elif "stop_vm" == cmd:
        server_id = sys.argv[2]
        client.stop_vm(server_id)
    elif "get_vol_list" == cmd:
        data = client.get_vol_list()
    elif "get_vol_info" == cmd:
        vol_id = sys.argv[2]
        data = client.get_vol_info(vol_id)
    elif "create_vol" == cmd:
        data = client.create_vol()
    elif "delete_vol" == cmd:
        vol_id = sys.argv[2]
        client.delete_vol(vol_id)
    elif "attach_vol" == cmd:
        server_id = sys.argv[2]
        vol_id = sys.argv[3]
        data = client.attach_vol(server_id, vol_id)
    elif "detach_vol" == cmd:
        server_id = sys.argv[2]
        vol_id = sys.argv[3]
        client.detach_vol(server_id, vol_id)
    elif "test_create" == cmd:
        data = client.create_vm()
        server_id = data["server"]["id"]
        time.sleep(10)
        data = client.create_vol()
        vol_id = data["volume"]["id"]
        time.sleep(10)
        data = client.attach_vol(server_id, vol_id)
    elif "test_delete" == cmd:
        server_id = sys.argv[2]
        vol_id = sys.argv[3]
        client.detach_vol(server_id, vol_id)
        client.delete_vol(vol_id)
        client.delete_vm(server_id)
    elif "my_get" == cmd:
        url = sys.argv[2]
        client.my_get(url)
    else:
        print(g_cmd_str)

    if None != data:
        print(json.dumps(data, sort_keys=True, indent=4, separators=(',', ':')))
    # except:
        # print(g_cmd_str)

    return ret


if __name__ == "__main__":
    main()