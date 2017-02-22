# In this case, in AWS Linux:

alias sudo='sudo env PATH=$PATH'
yum-config-manager --enable epel
yum install -y p7zip
cp /usr/bin/7za /usr/bin/7z

pip install --upgrade pip
sudo pip install numpy scipy h5py keras tensorflow