#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from targetrtslib import TargetRtslib


g_cmd_str = '''
    help                                     -- cmd help
    create size name                         -- create lv
    delete name                              -- delete lv
    assgin name_subfix initiator_iqn         -- attach
    detach name_iqn                          -- detach
    show                                     -- show
    test                                     -- test
'''


def main():

    ret = None
    cmd = None

    target_instance = TargetRtslib()

    cmd = sys.argv[1]
    # NOTE: ./process.py create 1024000
    if "create" == cmd:
        size = int(sys.argv[2])
        target_instance.create(size)
    # NOTE: ./process.py delete name
    elif "delete" == cmd:
        name = str(sys.argv[2])
        target_instance.delete(name)
    elif "assgin" == cmd:
        name_subfix = str(sys.argv[2])
        initiator_iqn = str(sys.argv[3])
        target_instance.assign_by_name(name_subfix, initiator_iqn)
    elif "detach" == cmd:
        name_iqn = str(sys.argv[2])
        target_instance.detach(name_iqn)
    elif "show" == cmd:
        target_instance.show()
    elif "test" == cmd:
        storage_object = target_instance.create(1000 * 1024 * 1024)
        wwn = target_instance.assign(storage_object, "iqn.2019-01.com.iscsi:initiator.109")
        target_instance.show()
        target_instance.detach(wwn)
        storage_object.delete()
    else:
        print(g_cmd_str)

    return ret


if __name__ == "__main__":
    main()