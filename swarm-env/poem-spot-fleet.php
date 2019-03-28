<?php

function construct_file_creation_cmd($fpath, $template) {
    $split_by_newline = function($s) {
        preg_match_all("/\n/", $s, $m, PREG_OFFSET_CAPTURE);
        $newline_pos = array_map(function($a) { return $a[1]; }, $m[0]);

        if(count($newline_pos) == 0)
            return [$s];

        $segment_start = -1;
        $result = [];
        foreach($newline_pos as $np) {
            $segment = substr($s, $segment_start+1, $np - $segment_start - 1);
            $result[] = $segment;
            $result[] = "\n";
            $segment_start = $np;
        }

        $remainder = substr($s, $segment_start+1);
        $result[] = (false === $remainder) ? '' : $remainder;

        return $result;
    };

    $splitted_template = [];
    foreach($template as $part)
        if(is_array($part))
            $splitted_template[] = $part;
        else
            $splitted_template = array_merge($splitted_template, $split_by_newline($part));

    $splitted_template = array_filter($splitted_template, function($t) { return is_array($t) || strlen($t) > 0; });
    
    $result[] = "( echo ";

    $last_result_str_apend = function($s) use (&$result) {
        $off = count($result) - 1;
        if(is_array($result[$off])) {
            $result[] = '';
            $off++;
        }
        $result[$off] = $result[$off].$s;
    };

    //var_export($splitted_template);

    foreach($splitted_template as $part) {
        if("\n" == $part) { // new line
            $last_result_str_apend(';');
            $result[] = "echo ";
        }
        else if(is_array($part)) {
            if(is_int(array_keys($part)[0])) // [ 'str', 'str', ... ] gets printed a set of an unescaped literals
                $last_result_str_apend(implode(' ', $part));
            else // [ 'str_key': ... ] gets embedded as is
                $result[] = $part;
        }
        else // finally, escape this literal
            $last_result_str_apend("'".str_replace(["'"], ["'\\''"], $part)."'");
    }
    $result[] = ") > $fpath\n";

    return $result;
}


function consul_service_config() {
    $cmd[] = <<<'EOT'
[Unit]
Description=consul agent

[Service]
EnvironmentFile=-/etc/sysconfig/consul
Restart=always
ExecStart=/usr/local/bin/run-consul

EOT;
    $cmd[] = <<<'EOT'

[Install]
WantedBy=multi-user.target

EOT;
    return $cmd;
}

function swarm_master_launch_script() {
    return [<<<'EOT'
#!/bin/bash
docker swarm init
while true; do
    consul kv put swarm_token `docker swarm join-token worker -q`
    if [ 0 == $? ]; then
        break
    fi
    echo "Failed to store the token in the consul kv storage. Sleeping..."
    sleep 1
done
EOT
];
}

function swarm_worker_launch_script() {
    $result = [<<<'EOT'
#!/bin/bash
key_name=swarm_token
while true; do
    consul kv get $key_name > /dev/null
    if [ 0 == $? ]; then
        consul kv get $key_name
        break
    fi
    echo "No data yet. sleeping..."
    sleep 1
done

EOT
];
    $result[] = 'docker swarm join --token `consul kv get swarm_token` ';
    $result[] = ["Fn::GetAtt" => [ "ManagerInstance", "PrivateIp"]];
    $result[] = ':2377';
    return $result;
}

//$common_user_data = 
function bootstrap_swarm($is_master, $group_name) {
    $result = [<<<'EOT'
#!/bin/bash -xe

apt-get install apt-transport-https ca-certificates curl software-properties-common -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) edge"
apt-get update
apt-get install docker-ce -y
usermod ubuntu -aG docker

wget https://releases.hashicorp.com/consul/0.7.5/consul_0.7.5_linux_amd64.zip
apt-get install unzip -y
unzip consul_0.7.5_linux_amd64.zip
mv consul /usr/local/bin/
adduser --disabled-password --gecos '' consul
mkdir /var/consul; chown consul:consul /var/consul

EOT
    ];

    $result = array_merge($result, construct_file_creation_cmd('/usr/local/bin/run-consul',
        array_merge([
            "#!/bin/bash\n",
            'M_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)'."\n"]
            , $is_master ?
                ['/usr/local/bin/consul agent -server -bootstrap -node '.$group_name.'-$M_IP -bind $M_IP -advertise $M_IP'] :
                ['/usr/local/bin/consul agent -join ', ["Fn::GetAtt" => [ "ManagerInstance", "PrivateIp"]], ' -node '.$group_name.'-$M_IP -bind $M_IP -advertise $M_IP']
            , [" -data-dir /var/consul\n"]
        )
    ));
    $result[] = "chmod +x /usr/local/bin/run-consul\n";
    $result = array_merge($result, construct_file_creation_cmd('/etc/systemd/system/consul.service', consul_service_config()));
    $result[] = "systemctl enable consul\n";
    $result[] = "systemctl start consul\n";

    $result[] = 'echo \'{"labels": ["node_role='.$group_name.'"]}\' > /etc/docker/daemon.json'."\n";
    $result[] = "systemctl restart docker\n";
    $result = array_merge($result, construct_file_creation_cmd('/usr/local/bin/swarm-init', $is_master ? swarm_master_launch_script() : swarm_worker_launch_script()));
    $result[] = "chmod +x /usr/local/bin/swarm-init\n";
    $result[] = "/usr/local/bin/swarm-init\n";

    return $result;
}

$result = [
    "Resources"=> [
        "ManagerInstance" => [
            "Type" => "AWS::EC2::Instance",
            "Properties" => [
                "ImageId"=> "ami-f4cc1de2",
                "KeyName"=> "aws-nvirginia",
                "InstanceType"=> "t2.micro",
                "NetworkInterfaces" => [[
                    "GroupSet"                 => [[ "Ref" => "WebServerSecurityGroup" ]],
                    "AssociatePublicIpAddress" => "true",
                    "DeviceIndex"              => "0",
                    "DeleteOnTermination"      => "true",
                    "SubnetId"                 => [ "Ref" => "Subnet" ]
                    ]],
                "UserData"=> ["Fn::Base64"=> ["Fn::Join"=> ["", bootstrap_swarm(true, 'master')]]]
            ]
        ],

        "SpotFleetWorker"=> [
            "Type"=> "AWS::EC2::SpotFleet",
            "Properties"=> [
                "SpotFleetRequestConfigData"=> [
                    "IamFleetRole"=> "arn:aws:iam::544719615594:role/spot-fleet",
                    "SpotPrice"=> "0.20",
                    "TargetCapacity"=> 1,
                    "LaunchSpecifications"=> [[
                        "InstanceType"=> "c4.xlarge", // "c3.large",
                        "ImageId"=> "ami-f4cc1de2",
                        "KeyName"=> "aws-nvirginia",
                        "NetworkInterfaces" => [[
                            "Groups"                   => [[ "Ref" => "WebServerSecurityGroup" ]],
                            "AssociatePublicIpAddress" => "true",
                            "SubnetId"                 => [ "Ref" => "Subnet" ],
                            "DeviceIndex"              => "0",
                            "DeleteOnTermination"      => "true"
                        ]],
                        "UserData"=> ["Fn::Base64"=> ["Fn::Join"=> ["", bootstrap_swarm(false, 'worker')]]]
                    ]]
                ]
            ]
        ],

        "SpotFleetWebserver"=> [
            "Type"=> "AWS::EC2::SpotFleet",
            "Properties"=> [
                "SpotFleetRequestConfigData"=> [
                    "IamFleetRole"=> "arn:aws:iam::544719615594:role/spot-fleet",
                    "SpotPrice"=> "0.20",
                    "TargetCapacity"=> 1,
                    "LaunchSpecifications"=> [[
                        "InstanceType"=> "c3.large",
                        "ImageId"=> "ami-f4cc1de2",
                        "KeyName"=> "aws-nvirginia",
                        "NetworkInterfaces" => [[
                            "Groups"                   => [[ "Ref" => "WebServerSecurityGroup" ]],
                            "AssociatePublicIpAddress" => "true",
                            "SubnetId"                 => [ "Ref" => "Subnet" ],
                            "DeviceIndex"              => "0",
                            "DeleteOnTermination"      => "true"
                        ]],
                        "UserData"=> ["Fn::Base64"=> ["Fn::Join"=> ["",
                            array_merge(bootstrap_swarm(false, 'webserver'), ["\nmkdir /var/log/poem-generator-www-logs/\n"])
                        ]]]
                    ]]
                ]
            ]
        ],

        "WebServerSecurityGroup" => [
            "Type" => "AWS::EC2::SecurityGroup",
            "Properties" => [
                "VpcId" => [ "Ref" => "VPC" ],
                "GroupDescription" => "Enable HTTP access via ports 22 and 80",
                "SecurityGroupIngress" => [
                    ["IpProtocol" => "tcp",  "FromPort" => "80",  "ToPort" => "80", "CidrIp" => "0.0.0.0/0"],
                    ["IpProtocol" => "tcp",  "FromPort" => "22",  "ToPort" => "22", "CidrIp" => "0.0.0.0/0"],
                    ["IpProtocol" => "tcp", "FromPort" => "443", "ToPort" => "443", "CidrIp" => "0.0.0.0/0"],

                    // Consul
                    ["IpProtocol"=> "tcp", "FromPort"=> "8300", "ToPort"=> "8300", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "tcp", "FromPort"=> "8301", "ToPort"=> "8301", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "udp", "FromPort"=> "8301", "ToPort"=> "8301", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "tcp", "FromPort"=> "8302", "ToPort"=> "8302", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "udp", "FromPort"=> "8302", "ToPort"=> "8302", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "tcp", "FromPort"=> "8400", "ToPort"=> "8400", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "tcp", "FromPort"=> "8500", "ToPort"=> "8500", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "tcp", "FromPort"=> "8600", "ToPort"=> "8600", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "udp", "FromPort"=> "8600", "ToPort"=> "8600", "CidrIp"=> "0.0.0.0/0"],

                    // Swarm
                    ["IpProtocol"=> "tcp", "FromPort"=> "2377", "ToPort"=> "2377", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "tcp", "FromPort"=> "7946", "ToPort"=> "7946", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "udp", "FromPort"=> "7946", "ToPort"=> "7946", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "tcp", "FromPort"=> "4789", "ToPort"=> "4789", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "udp", "FromPort"=> "4789", "ToPort"=> "4789", "CidrIp"=> "0.0.0.0/0"],

                    ["IpProtocol"=> "tcp", "FromPort"=> "8888", "ToPort"=> "8888", "CidrIp"=> "0.0.0.0/0"],
                ]/*,
                "SecurityGroupEgress" => [
                    ["IpProtocol"=> "tcp", "FromPort"=> "80", "ToPort"=> "80", "CidrIp"=> "0.0.0.0/0"],
                    ["IpProtocol"=> "tcp", "FromPort"=> "443", "ToPort"=> "443", "CidrIp"=> "0.0.0.0/0"]
                ]*/
            ]
        ],
        "VPC" => [
            "Type" => "AWS::EC2::VPC",
            "Properties" => [
                "CidrBlock" => "10.0.0.0/16",
                "EnableDnsSupport" => "true",
                "EnableDnsHostnames" => "true",
                "Tags" => [ ["Key" => "Application", "Value" => [ "Ref" => "AWS::StackId"] ] ]
            ]
        ],
        "Subnet" => [
            "Type" => "AWS::EC2::Subnet",
            "Properties" => [
                "VpcId" => [ "Ref" => "VPC" ],
                "CidrBlock" => "10.0.0.0/24",
                "Tags" => [ ["Key" => "Application", "Value" => [ "Ref" => "AWS::StackId"] ] ]
            ]
        ],





        "InternetGateway" => [
            "Type" => "AWS::EC2::InternetGateway",
            "Properties" => [
                    "Tags" => [ ["Key" => "Application", "Value" => [ "Ref" => "AWS::StackId"] ] ]
            ]
        ],

        "AttachGateway" => [
            "Type" => "AWS::EC2::VPCGatewayAttachment",
            "Properties" => [
                "VpcId" => [ "Ref" => "VPC" ],
                "InternetGatewayId" => [ "Ref" => "InternetGateway" ]
            ]
        ],

        "RouteTable" => [
            "Type" => "AWS::EC2::RouteTable",
            "Properties" => [
                "VpcId" => ["Ref" => "VPC"],
                "Tags" => [ ["Key" => "Application", "Value" => [ "Ref" => "AWS::StackId"] ] ]
            ]
        ],

        "Route" => [
            "Type" => "AWS::EC2::Route",
            "DependsOn" => "AttachGateway",
            "Properties" => [
                "RouteTableId" => [ "Ref" => "RouteTable" ],
                "DestinationCidrBlock" => "0.0.0.0/0",
                "GatewayId" => [ "Ref" => "InternetGateway" ]
            ]
        ],

        "SubnetRouteTableAssociation" => [
            "Type" => "AWS::EC2::SubnetRouteTableAssociation",
            "Properties" => [
                "SubnetId" => [ "Ref" => "Subnet" ],
                "RouteTableId" => [ "Ref" => "RouteTable" ]
            ]
        ],

        "NetworkAcl" => [
            "Type" => "AWS::EC2::NetworkAcl",
            "Properties" => [
                "VpcId" => ["Ref" => "VPC"],
                "Tags" => [ ["Key" => "Application", "Value" => [ "Ref" => "AWS::StackId"] ] ]
            ]
        ],

        "InboundHTTPNetworkAclEntry" => [
            "Type" => "AWS::EC2::NetworkAclEntry",
            "Properties" => [
                "NetworkAclId" => ["Ref" => "NetworkAcl"],
                "RuleNumber" => "100",
                "Protocol" => "6",
                "RuleAction" => "allow",
                "Egress" => "false",
                "CidrBlock" => "0.0.0.0/0",
                "PortRange" => ["From" => "80", "To" => "80"]
            ]
        ],

        "InboundSSHNetworkAclEntry" => [
            "Type" => "AWS::EC2::NetworkAclEntry",
            "Properties" => [
                "NetworkAclId" => ["Ref" => "NetworkAcl"],
                "RuleNumber" => "101",
                "Protocol" => "6",
                "RuleAction" => "allow",
                "Egress" => "false",
                "CidrBlock" => "0.0.0.0/0",
                "PortRange" => ["From" => "22", "To" => "22"]
            ]
        ],

        "InboundResponsePortsNetworkAclEntry" => [
            "Type" => "AWS::EC2::NetworkAclEntry",
            "Properties" => [
                "NetworkAclId" => ["Ref" => "NetworkAcl"],
                "RuleNumber" => "102",
                "Protocol" => "6",
                "RuleAction" => "allow",
                "Egress" => "false",
                "CidrBlock" => "0.0.0.0/0",
                "PortRange" => ["From" => "1024", "To" => "65535"]
            ]
        ],

        "OutBoundHTTPNetworkAclEntry" => [
            "Type" => "AWS::EC2::NetworkAclEntry",
            "Properties" => [
                "NetworkAclId" => ["Ref" => "NetworkAcl"],
                "RuleNumber" => "100",
                "Protocol" => "6",
                "RuleAction" => "allow",
                "Egress" => "true",
                "CidrBlock" => "0.0.0.0/0",
                "PortRange" => ["From" => "80", "To" => "80"]
            ]
        ],

        "OutBoundHTTPSNetworkAclEntry" => [
            "Type" => "AWS::EC2::NetworkAclEntry",
            "Properties" => [
                "NetworkAclId" => ["Ref" => "NetworkAcl"],
                "RuleNumber" => "101",
                "Protocol" => "6",
                "RuleAction" => "allow",
                "Egress" => "true",
                "CidrBlock" => "0.0.0.0/0",
                "PortRange" => ["From" => "443", "To" => "443"]
            ]
        ],

        "OutBoundResponsePortsNetworkAclEntry" => [
            "Type" => "AWS::EC2::NetworkAclEntry",
            "Properties" => [
                "NetworkAclId" => ["Ref" => "NetworkAcl"],
                "RuleNumber" => "102",
                "Protocol" => "6",
                "RuleAction" => "allow",
                "Egress" => "true",
                "CidrBlock" => "0.0.0.0/0",
                "PortRange" => ["From" => "1024", "To" => "65535"]
            ]
        ],

        "SubnetNetworkAclAssociation" => [
            "Type" => "AWS::EC2::SubnetNetworkAclAssociation",
            "Properties" => [
                "SubnetId" => [ "Ref" => "Subnet" ],
                "NetworkAclId" => [ "Ref" => "NetworkAcl" ]
            ]
        ]
    ]
];

file_put_contents(
    __DIR__.'/'.str_replace('.php', '.json', basename(__FILE__)),
    json_encode($result, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES));

