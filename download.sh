# Datasets
mkdir datasets
cd datasets
# Imagenet (resize)
wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
tar -xvzf Imagenet_resize.tar.gz
# LSUN (resize)
wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
tar -xvzf LSUN_resize.tar.gz
# iSUN
wget https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz
tar -xvzf iSUN.tar.gz
# Describable Textures Dataset (DTD) (Textures)
mkdir Textures/
cd Textures/
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xvzf dtd-r1.0.1.tar.gz
cd ..
# Chars74K
mkdir Chars74K/
cd Chars74K/
wget http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz
tar -xvzf EnglishImg.tgz
cd ..
# PLACES365
mkdir Places365/
cd Places365/
wget http://data.csail.mit.edu/places/places365/val_256.tar
tar -xvf val_256.tar
cd ..
cd ..

# Models
mkdir pre_trained
cd pre_trained
# DenseNet-BC trained on CIFAR-10
wget https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz
tar -xvzf densenet10.pth.tar.gz
# DenseNet-BC trained on CIFAR-100
wget https://www.dropbox.com/s/vxuv11jjg8bw2v9/densenet100.pth.tar.gz
tar -xvzf densenet100.pth.tar.gz
# DenseNet-BC trained on SVHN
wget https://www.dropbox.com/s/9ol1h2tb3xjdpp1/densenet_svhn.pth

# ResNet34 trained on CIFAR-10
wget https://www.dropbox.com/s/ynidbn7n7ccadog/resnet_cifar10.pth
# ResNet34 trained on CIFAR-100
wget https://www.dropbox.com/s/yzfzf4bwqe4du6w/resnet_cifar100.pth
# ResNet34 trained on SVHN
wget https://www.dropbox.com/s/uvgpgy9pu7s9ps2/resnet_svhn.pth
cd ..
