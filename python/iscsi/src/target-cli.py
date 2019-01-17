class Target():

    def __init__(self):
        # lv_list is a dict list
        # dict = {
        #   "lv_name" : "lv_name",
        #   "vg_name" : "vg_name",
        #   "lv_size" : "lv_size"
        # }
        self.lv_list = []
        # target_list id a dict list
        # dict = {
        #   "lv_name" : "lv_name",
        #   "vg_name" : "vg_name",
        #   "lv_size" : "lv_size"
        # }
        self.target_list = []
        pass

    # create pv (physical volume)
    # pvcreate /dev/vde
    # pvremove /dev/vde
    # pvdisplay
    # create vg (volume group)
    # vgcreate vg01 /dev/vde
    # vgremove vg01
    # vgdiplay (vgdisplay不显示实际连接的pv？)
    # create lv (logical volume)
    # lvcreate -n lv01 -L 500M vg01
    # lvremove /dev/vg01/lv01
    # lvdiplay
    def create_lv(self, lv_name, vg_name, lv_size=1000):
        process_cmd = ["lvcreate"] + ["-n"] + [lv_name] + ["-L"] + \
                      ["{}M".format(lv_size)] + [vg_name]
        process = Popen(process_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        lv_path = "/dev/" + vg_name + "/" + lv_name
        dict_tmp = {
            "lv_name" : lv_name,
            "vg_name" : vg_name,
            "lv_size" : lv_size,
            "lv_path" : lv_path
        }
        self.lv_list.append(dict_tmp)
        ret = process.returncode
        process.wait()

    def delete_lv(self, lv_name, vg_name):
        lv_path = "/dev/" + vg_name + "/" + lv_name
        process_cmd = ["lvremove"] + ["-f"] + [lv_path]
        process = Popen(process_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        for lv in self.lv_list:
            if lv_path == lv["lv_path"]:
                self.lv_list.remove(lv)
        ret = process.returncode
        process.wait()

    def show_lv(self):
        for lv in self.lv_list:
            print("\n==lv info:==\n")
            print("lv_name : {}".format(lv["lv_name"]))
            print("vg_name : {}".format(lv["vg_name"]))
            print("lv_size : {}".format(lv["lv_size"]))
            print("lv_path : {}".format(lv["lv_path"]))
            print("======\n")

    #  attach
    def assgin(self, iqn_target_name, iqn_initiator_name, 
               storage_object):
        # create iqn (target)
        process_cmd = ["targetcli"] + \
                      ["/iscsi create {}".format(iqn_target_name)]
        process = Popen(process_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        ret = process.returncode
        process.wait()

        # create lun
        process_cmd = ["targetcli"] + \
                      ["/iscsi/{iqn_target_name}/tpg1/luns create \
                      storage_object={storage_object} \
                      ".format(iqn_target_name=iqn_target_name, \
                      storage_object=storage_object)]
        process = Popen(process_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        ret = process.returncode
        process.wait()

        # create acl
        process_cmd = ["targetcli"] + \
                      ["/iscsi/{iqn_target_name}/tpg1/acls create \
                      {iqn_initiator_name} \
                      ".format(iqn_target_name=iqn_target_name, \
                      iqn_initiator_name=iqn_initiator_name)]
        process = Popen(process_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        ret = process.returncode
        process.wait()

        dict_tmp = {
            "iqn_target_name"   : iqn_target_name,
            "iqn_initiator_name": iqn_initiator_name,
            "storage_object"    : storage_object,
        }
        self.target_list.append(dict_tmp)

    # a simple way : delete the target directly
    def detach(self, iqn_target_name, lv_name, vg_name):
        process_cmd = ["targetcli"] + \
                      ["/iscsi delete {}".format(iqn_target_name)]
        process = Popen(process_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        ret = process.returncode
        process.wait()

        process_cmd = ["targetcli"] + \
                      ["/backstores/block delete dev-{}-{}".format(vg_name, lv_name)]
        process = Popen(process_cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        ret = process.returncode
        process.wait()

        for target in self.target_list:
            if iqn_target_name == target["iqn_target_name"]:
                self.target_list.remove(target)

    def show_assgin(self):
        for target in self.target_list:
            print("\n==assgin info:==\n")
            print("iqn_target_name    : {} \
                  ".format(target["iqn_target_name"]))
            print("iqn_initiator_name : {} \
                  ".format(target["iqn_initiator_name"]))
            print("storage_object     : {} \
                  ".format(target["storage_object"]))
            print("======\n")