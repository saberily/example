#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, uuid
import targetcli as target


g_cmd_str = '''
    help                                                        -- cmd help
    create_lv lv_name vg_name lv_size                           -- create lv
    delete_lv lv_name vg_name                                   -- delete lv
    assgin iqn_target_name iqn_initiator_name storage_object    -- attach
    detach iqn_target_name                                      -- detach
    test                                                        -- test
'''


def main():

    ret = None
    cmd = None

    target_instance = target.Target()

    cmd = sys.argv[1]
    # ./process.py create_lv test_lv vg01 2000
    if "create_lv" == cmd:
        lv_name = str(sys.argv[2])
        vg_name = str(sys.argv[3])
        lv_size = int(sys.argv[4])
        target_instance.create_lv(lv_name, vg_name, lv_size)
    # ./process.py delete_lv test_lv vg01
    elif "delete_lv" == cmd:
        lv_name = str(sys.argv[2])
        vg_name = str(sys.argv[3])
        target_instance.delete_lv(lv_name, vg_name)
    # ./process.py assgin iqn.2019-01.com.iscsi:storage.disk1 iqn.2019-01.com.iscsi:initiator.109 /dev/vg01/test_lv
    elif "assgin" == cmd:
        iqn_target_name    = sys.argv[2]
        iqn_initiator_name = sys.argv[3]
        storage_object     = sys.argv[4]
        target_instance.assgin(iqn_target_name, iqn_initiator_name, storage_object)
    elif "detach" == cmd:
        iqn_target_name    = sys.argv[2]
        target_instance.detach(iqn_target_name)
    elif "test" == cmd:
        print("---test begin:")
        lv_name = str(uuid.uuid1())
        target_instance.create_lv(lv_name, "vg01")
        target_instance.assgin("iqn.2019-01.com.iscsi:{}".format(lv_name), \
                                "iqn.2019-01.com.iscsi:initiator.109", \
                                "/dev/vg01/{}".format(lv_name))
        target_instance.show_lv()
        target_instance.show_assgin()
        print("---test middle---")
        target_instance.detach("iqn.2019-01.com.iscsi:{}".format(lv_name),
                               lv_name, "vg01")
        target_instance.delete_lv(lv_name, "vg01")
        target_instance.show_lv()
        target_instance.show_assgin()
        print("---test end---")
    else:
        print(g_cmd_str)

    return ret


if __name__ == "__main__":
    main()