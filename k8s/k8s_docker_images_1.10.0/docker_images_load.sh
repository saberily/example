docker load < quay.io#calico#node.tar 
docker load < quay.io#calico#cni.tar
docker load < quay.io#calico#kube-controllers.tar
docker load < k8s.gcr.io#kube-proxy-amd64.tar
docker load < k8s.gcr.io#kube-scheduler-amd64.tar
docker load < k8s.gcr.io#kube-controller-manager-amd64.tar
docker load < k8s.gcr.io#kube-apiserver-amd64.tar
docker load < k8s.gcr.io#etcd-amd64.tar
docker load < k8s.gcr.io#k8s-dns-dnsmasq-nanny-amd64.tar
docker load < k8s.gcr.io#k8s-dns-sidecar-amd64.tar
docker load < k8s.gcr.io#k8s-dns-kube-dns-amd64.tar
docker load < k8s.gcr.io#pause-amd64.tar
docker load < quay.io#coreos#etcd.tar
docker load < quay.io#calico#node.tar
docker load < quay.io#calico#cni.tar
docker load < quay.io#calico#kube-policy-controller.tar
docker load < gcr.io#google_containers#etcd.tar

