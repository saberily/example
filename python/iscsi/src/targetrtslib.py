#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid, rtslib
import rtslib.root as root


class TargetRtslib():

    def __init__(self):
        self.tmp_dir = '/tmp/'
        # FIXME: need to specify the index
        self.index = 1
        # FIXME: need to specify the iqn_prefix
        self.iqn_prefix = "iqn.2019-01.com.iscsi:"

    def create(self, size, name=None):
        # TODO: size check
        try:
            # NOTE: /sys/kernel/config/target/core/fileio_1
            file_back_store = rtslib.FileIOBackstore(self.index)
            if name == None:
                name = str(uuid.uuid1())
            dev = self.tmp_dir + name
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
        try:
            # NOTE: /sys/kernel/config/target/core/fileio_1
            file_back_store = rtslib.FileIOBackstore(self.index)
            file_obj = rtslib.FileIOStorageObject(file_back_store,
                                                  name)
            file_obj.delete()
        except:
            print("file_obj delete failed!")
        else:
            pass
            
    def assign(self, storage_object, initiator_iqn):
        # TODO: initiator_iqn format check
        try:
            module = rtslib.FabricModule('iscsi')
            wwn = self.iqn_prefix + str(uuid.uuid1())
            tgt = rtslib.Target(module, wwn=wwn)
            tpg = rtslib.TPG(tgt, self.index) 
            rtslib.LUN(tpg, lun=self.index, storage_object=storage_object) 
            acl = rtslib.NodeACL(tpg, node_wwn=initiator_iqn)
            rtslib.MappedLUN(acl, self.index, tpg_lun=self.index)
            rtslib.NetworkPortal(tpg, '0.0.0.0', mode='create')
            # FIXME: more attribute need to set
            tpg.set_attribute('generate_node_acls', '1') 
            tpg.set_attribute('generate_node_acls', '1') 
            tpg.set_attribute('cache_dynamic_acls', '1') 
            tpg.set_attribute('authentication', '0') 
            tpg.set_attribute('demo_mode_write_protect', '0') 
            tpg.enable = True
        except:
            print("assign failed!")
            return None
        else:
            return wwn

    def assign_by_name(self, name, initiator_iqn):
        try:
            file_back_store = rtslib.FileIOBackstore(self.index)
            file_obj = rtslib.FileIOStorageObject(file_back_store,
                                                  name)
            wwn = self.assign(file_obj, initiator_iqn)
        except:
            print("assign failed!")
            return None
        else:
            return wwn

    def detach(self, wwn):
        try:
            module = rtslib.FabricModule('iscsi')
            tgt = rtslib.Target(module, wwn=wwn)
            tgt.delete()
        except:
            print("Target delete failed!")
        else:
            pass
            
    def show(self):
        # FIXME: more info need to show
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

        print("\n===iscsi===\n")
        module = rtslib.FabricModule('iscsi')
        for target in module._list_targets():
            print(target)


if __name__ == "__main__":
    obj = TargetRtslib()
    storage_object = obj.create(1000 * 1024 * 1024)
    wwn = obj.assign(storage_object, "iqn.2019-01.com.iscsi:initiator.109")
    obj.show()
    obj.detach(wwn)
    storage_object.delete()
