from __future__ import generators
from __future__ import generator_stop

from keras.models import Sequential,Model
from keras.layers import Conv2D,MaxPooling2D,BatchNormalization
from keras.layers import Activation,Flatten,Dropout,Dense

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
#from sklearn.metrics import classification_report
from skimage import transform,exposure,io
import matplotlib.pyplot as plt
import numpy as np
import random
import os

class NeuralNet:
  @staticmethod
  def build(width,height,depth,classes):
    model=Sequential()
    inputShape=(height,width,depth)
    ChanDim=-1

    #Conv2D->Relu->BatchNorm->MaxPool

    #size=32x32 
    model.add(Conv2D(8,(5,5),padding="same",input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=ChanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    #size=16x16 

    #first set (Conv->ReLU->Conv->ReLU)*2->MaxPool
    model.add(Conv2D(16,(3,3),padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=ChanDim))
    model.add(Conv2D(16,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=ChanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #size=3x3 

    #second set (Conv->ReLU->Conv->ReLU)*2->MaxPool
    model.add(Conv2D(32,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=ChanDim))
    model.add(Conv2D(32,(3,3),padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=ChanDim))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #Two sets of FC and softmax classifier
    
    #first set
    model.add(Flatten())
    #model=Flatten()(model)
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    '''
    #second set
    model.add(Flatten())
    #model=Flatten()(model)
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    '''
    #softmax
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

  def setParams():
    #hyperparams
    num_epochs=30
    init_lr=1e-3 
    bs=64 

  def imagAug():
    aug=ImageDataGenerator(
      rotation_range=10,
      zoom_range=0.15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.15,
      horizontal_flip=False,
      vertical_flip=False,
      fill_mode="nearest"   
    )

  def train():
    opt=Adam(lr=init_lr,decay=init_lr/(num_epochs*0.5))
    model=TrafficSignNet.build(width=32,height=32,depth=3,classes=numLabels)
    model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])
    H=model.fit_generator(
        aug.flow(trainX,trainY,batch_size=bs),
        validation_data=(testX,testY),
        steps_per_epoch=trainX.shape[0]//bs,
        epochs=num_epochs,
        class_weight=classWeight,
        verbose=1
      )
  
def load_model(path):
  model=keras.load_model(model.h5)

def signs():
  SIGNS=[
         'ERROR',
         'STOP',
         'TURN LEFT',
         'TURN RIGHT',
         'DO NOT TURN LEFT',
         'DO NOT TURN RIGHT',
         'ONE WAY',
         'SPEED LIMIT',
         'OTHER'
         ]

def preds_filter():
  #filter only signs from GTSRB datasets (for same video output)
  for sign in signs.SIGNS:
    for label in range(numLabels):
      gtr=trainX[label]
      set1=set(map(lambda:sign.lower(),sign.split('')))
      set2=set(map(lambda:gtr,gtr.lower(),gtr.split('')))
      if set1==set2:
        continue

def trainNN(opt):
  data_loader = CreateDataLoader(opt)
  dataset = data_loader.load_data()
  dataset_size = len(data_loader)
  #print('#training images = %d' % dataset_size)
    
  start_epoch, epoch_iter=1, 0
  total_steps=(start_epoch-1) * dataset_size + epoch_iter
  display_delta=total_steps % opt.display_freq
  print_delta=total_steps % opt.print_freq
  save_delta=total_steps % opt.save_latest_freq

  for data in tqdm(dataset):
    minibatch = 1 
    reset = model.inference(data['label'], data['inst'])
        
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('reset_image', util.tensor2im(reset.data[0]))])
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)
    webpage.save()

def videoEncodingPreds():
  iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'main.avi')
  opt.dataroot
  opt.isTrain=True
  opt.use_encoded_image=True

  model = NeuralNet.build()
  trainedModel = trainNN(model)
  i=0
  for data in tqdm(dataset):
    iter_start_time = time.time()
    total_steps+=1
    epoch_iter+=1

    #forward pass
    losses, generated = model(Variable(data['label']), Variable(data['inst']), 
        Variable(data['image']), Variable(data['feat']), infer=True)

    #sum per device losses
    losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
    loss_dict = dict(zip(model.module.loss_names, losses))

    # calculate final loss scalar
    loss_D = (loss_dict['CNN'] + loss_dict['SVM']) * 0.5
    loss_G = loss_dict['LNT'] + loss_dict.get('GTRSB',0) + loss_dict.get('main',0)


    #results and errors
    ### print errors
    errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
    t = (time.time() - iter_start_time) / opt.batchSize
    visualizer.print_current_errors(epoch, epoch_iter, errors, t)
    visualizer.plot_current_errors(errors, total_steps)

    #output images
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('trained_image', util.tensor2im(generated.data[0])),
                           ('real_image', util.tensor2im(data['image'][0]))])
    visualizer.display_current_results(visuals, i, total_steps)

    #error             
    np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
    i+=1


