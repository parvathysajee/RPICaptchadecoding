import paramiko


#adding ssh rasp pi details
host = "rasp-019.berry.scss.tcd.ie"
port = 22
username = "sajeekup"
password = "Friends@25"

command = "sh automation.sh"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port, username, password)

stdin, stdout, stderr = ssh.exec_command(command)
lines = stdout.readlines()
print(lines)