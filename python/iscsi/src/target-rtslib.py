#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid, rtslib
import rtslib.root as root


class TargetRtslib():

    def __init__(self):
        self.tmp_dir = '/tmp/'
        self.index = 1
        self.iqn_prefix = "iqn.2019-01.com.iscsi:"

    def create(self, size, name=None):
        # /sys/kernel/config/target/core/fileio_1
        file_back_store = rtslib.FileIOBackstore(self.index)
        if name == None:
            name = str(uuid.uuid1())
        dev = self.tmp_dir + name
        try:
            file_obj = rtslib.FileIOStorageObject(file_back_store,
                                                  name,
                                                  dev=dev,
                                                  size=size)
        except:
            print("file_obj create failed!")
            return None
        else:
            return file_obj

    def delete(self, name):
        # /sys/kernel/config/target/core/fileio_1
        try:
            file_back_store = rtslib.FileIOBackstore(self.index)
            file_obj = rtslib.FileIOStorageObject(file_back_store,
                                                  name)
        except:
            print("file_obj lookup failed!")
        else:
            file_obj.delete()

    def assign(self, storage_object, initiator_iqn):
        try:
            module = rtslib.FabricModule('iscsi')
            wwn = self.iqn_prefix + str(uuid.uuid1())
            tgt = rtslib.Target(module, wwn=wwn)
            tpg = rtslib.TPG(tgt, self.index) 
            rtslib.LUN(tpg, lun=self.index, storage_object=storage_object) 
            rtslib.NodeACL(tpg, node_wwn=initiator_iqn)
            rtslib.NetworkPortal(tpg, '0.0.0.0', mode='create')
            # tpg.set_attribute('generate_node_acls', '1') 
            tpg.enable = True
        except:
            print("assign failed!")
            return None
        else:
            return wwn

    def detach(self, wwn):
        try:
            module = rtslib.FabricModule('iscsi')
            tgt = rtslib.Target(module, wwn=wwn)
        except:
            print("Target lookup failed!")
        else:
            tgt.delete()
        
    def show(self):
        rtsroot = root.RTSRoot()

        print("\n===targets===\n")
        for tgt in rtsroot.targets:
            print(tgt)
        
        print("\n===backstores===\n")
        for bs in rtsroot.backstores:
            print(bs)
        
        print("\n===tpgs===\n")
        for tpg in rtsroot.tpgs:
            print(tpg)

        print("\n===storage_objects===\n")
        for sobj in rtsroot.storage_objects:
            print(sobj)
        
        print("\n===network_portals===\n")
        for np in rtsroot.network_portals:
            print(np)


if __name__ == "__main__":
    obj = TargetRtslib()
    storage_object = obj.create(1000 * 1024 * 1024)
    wwn = obj.assign(storage_object, "iqn.2019-01.com.iscsi:initiator.109")
    obj.show()
    obj.detach(wwn)
    storage_object.delete()
