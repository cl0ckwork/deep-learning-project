#!/usr/bin/env bash

PROJECT_ID=$(gcloud config get-value project)
ZONE=us-east1-b
MACHINE_TYPE=n1-standard-2
MACHINE_NAME=gwu-dl-db
#MACHINE_IMAGE=ubuntu-1804-lts

#gcloud auth login

echo -e "deploying vm to ${PROJECT_ID}"

docker-machine create --driver google \
  --google-project ${PROJECT_ID} \
  --google-zone ${ZONE} \
  --google-machine-type ${MACHINE_TYPE} \
  --google-tags database, database-ingress \
  --google-disk-size 100 \
  ${MACHINE_NAME}

#POST https://www.googleapis.com/compute/v1/projects/polar-protocol-228721/zones/us-east1-c/instances
#{
#  "kind": "compute#instance",gcloud compute machine-types list
#  "name": "gwu-ml-2-gpu",
#  "zone": "projects/polar-protocol-228721/zones/us-east1-c",
#  "machineType": "projects/polar-protocol-228721/zones/us-east1-c/machineTypes/n1-standard-8",
#  "metadata": {
#    "kind": "compute#metadata",
#    "items": [
#      {
#        "key": "purpose",
#        "value": "school"
#      },
#      {
#        "key": "ssh-keys",
#        "value": "ubuntu:sh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDMTEPaI3yY7IX5ZW7bEzyaZUklGQWBWpQfhUu3QKd609bKNrW9rXGIVap45FSYcPWn8xzMdPbQ5Qv4VcFrCP1+FC1iAsRDB/eVyIY77rCM9BYdFIxqF16E6wztxQahlkNFu412w/ecOF7X2dz399GnWSlJS4LNHBh+EVAMNUnCAUubogd5GP7OQyGkfV4M7eoPbHs1srCDisrps7cv+BbRonENAo/L26w3bhXKRMY4jEimvAnjP0/OltnA6FuM1utOTcUgUGH8FOD0TzJX6v+C9b2mKO9eQiaezJRUr8jSQxyI2f5OpdSYo85dvN1JVOmtAaMIn3t3jdMPUJGT6yTJ ubuntu"
#      }
#    ]
#  },
#  "tags": {
#    "items": [
#      "http-server"
#    ]
#  },
#  "guestAccelerators": [
#    {
#      "acceleratorType": "projects/polar-protocol-228721/zones/us-east1-c/acceleratorTypes/nvidia-tesla-k80",
#      "acceleratorCount": 2
#    }
#  ],
#  "disks": [
#    {
#      "kind": "compute#attachedDisk",
#      "type": "PERSISTENT",
#      "boot": true,
#      "mode": "READ_WRITE",
#      "autoDelete": true,
#      "deviceName": "gwu-ml-2-gpu",
#      "initializeParams": {
#        "sourceImage": "projects/annular-reef-179815/global/images/u-16-fall-2018",
#        "diskType": "projects/polar-protocol-228721/zones/us-east1-c/diskTypes/pd-standard",
#        "diskSizeGb": "100"
#      }
#    }
#  ],
#  "canIpForward": false,
#  "networkInterfaces": [
#    {
#      "kind": "compute#networkInterface",
#      "subnetwork": "projects/polar-protocol-228721/regions/us-east1/subnetworks/default",
#      "accessConfigs": [
#        {
#          "kind": "compute#accessConfig",
#          "name": "External NAT",
#          "type": "ONE_TO_ONE_NAT",
#          "networkTier": "PREMIUM"
#        }
#      ],
#      "aliasIpRanges": []
#    }
#  ],
#  "description": "GWU Machine Learning 2",
#  "labels": {},
#  "scheduling": {
#    "preemptible": false,
#    "onHostMaintenance": "TERMINATE",
#    "automaticRestart": true,
#    "nodeAffinities": []
#  },
#  "deletionProtection": false,
#  "serviceAccounts": [
#    {
#      "email": "397360352825-compute@developer.gserviceaccount.com",
#      "scopes": [
#        "https://www.googleapis.com/auth/devstorage.read_only",
#        "https://www.googleapis.com/auth/logging.write",
#        "https://www.googleapis.com/auth/monitoring.write",
#        "https://www.googleapis.com/auth/servicecontrol",
#        "https://www.googleapis.com/auth/service.management.readonly",
#        "https://www.googleapis.com/auth/trace.append"
#      ]
#    }
#  ]
#}

#POST https://www.googleapis.com/compute/v1/projects/polar-protocol-228721/global/firewalls
#{
#  "name": "default-allow-http",
#  "kind": "compute#firewall",
#  "sourceRanges": [
#    "0.0.0.0/0"
#  ],
#  "network": "projects/polar-protocol-228721/global/networks/default",
#  "targetTags": [
#    "http-server"
#  ],
#  "allowed": [
#    {
#      "IPProtocol": "tcp",
#      "ports": [
#        "80"
#      ]
#    }
#  ]
#}


#POST https://www.googleapis.com/compute/v1/projects/polar-protocol-228721/global/firewalls
#{
#  "name": "default-allow-https",
#  "kind": "compute#firewall",
#  "sourceRanges": [
#    "0.0.0.0/0"
#  ],
#  "network": "projects/polar-protocol-228721/global/networks/default",
#  "targetTags": [
#    "https-server"
#  ],
#  "allowed": [
#    {
#      "IPProtocol": "tcp",
#      "ports": [
#        "443"
#      ]
#    }
#  ]
#}