wget http://supun.ucsd.edu/krypton/cnn_weights/resnet18_weights_ptch.h5
mv resnet18_weights_ptch.h5 core/python


wget http://supun.ucsd.edu/krypton/cnn_weights/inception3_weights_ptch.h5
mv inception3_weights_ptch.h5 core/python

wget http://supun.ucsd.edu/krypton/cnn_weights/vgg16_weights_ptch.h5
mv vgg16_weights_ptch.h5 core/python

for dataset in 'oct' 'chest'
do
    for model in 'vgg16' 'resnet18' 'inception3'
        do
            wget "http://supun.ucsd.edu/krypton/cnn_weights/"$dataset"_"$model"_ptch.h5";
            mv $dataset"_"$model"_ptch.h5" core/python;
        done
done
