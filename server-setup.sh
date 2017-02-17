# In this case, in AWS Linux:
yum-config-manager --enable epel
yum install -y p7zip
cp /usr/bin/7za /usr/bin/7z
alias sudo='sudo env PATH=$PATH'