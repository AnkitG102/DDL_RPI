import os
import time
import shutil
from fabric import Connection
from train_client import num_epoch as epoch_count
import warnings
warnings.filterwarnings("ignore")


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Defining all the static variables
#Setting the static IP addresses of all the clients
params = {
    'ip_1':'127.0.0.1', #IP of machine 1
    'ip_2':'127.0.0.1', #IP of machine 2
    'ip_3':'127.0.0.1', #IP of machine 3
    'ip_server':'127.0.0.1', # IP of the master node
    # set passpassword here for each client node. If diffeent passwort are used, feed all of them here.
    'passwort':'passwort'
}

#Define the destination folders for the system to accurately access the relevant information during processing.
dir = {
    'server_archive':'/home/pi/Desktop/fl_server/model_archive/',  #archive of models generated.
    'server_home': '/home/pi/Desktop/fl_server/',  #server home directory

    #client node home directories
    'machine_1_home':'/home/pi/Desktop/fl_1/',
    'machine_2_home':'/home/pi/Desktop/fl_2/',
    'machine_3_home':'/home/pi/Desktop/fl_3/',

    #client node model archives
    'machine_1_archive':'/home/pi/Desktop/fl_1/model_archive',
    'machine_2_archive':'/home/pi/Desktop/fl_2/model_archive',
    'machine_3_archive':'/home/pi/Desktop/fl_3/model_archive',

    #client node master model saving directories
    'machine_1_master_model':'/home/pi/Desktop/fl_1/master_model/',
    'machine_2_master_model':'/home/pi/Desktop/fl_2/master_model/',
    'machine_3_master_model':'/home/pi/Desktop/fl_3/master_model/',

    #client nodes training data destination
    'machine_1_train_data':'/home/pi/Desktop/fl_1/data/train',
    'machine_2_train_data':'/home/pi/Desktop/fl_2/data/train',
    'machine_3_train_data':'/home/pi/Desktop/fl_3/data/train',

    #client nodes validation data destination
    'machine_1_validation_data':'/home/pi/Desktop/fl_1/data/validation',
    'machine_2_validation_data':'/home/pi/Desktop/fl_2/data/validation',
    'machine_3_validation_data':'/home/pi/Desktop/fl_3/data/validation',

    'machine_3_results':'/home/pi/Desktop/fl_3/results'
}


#parameters required for training the network
img_h, img_w = 150,150
train_samples = 500
validation_samples = 100
nb_epochs = int(epoch_count)
nb_batch_size= 16


def model_define():
    """ declaring and defining the model to be trainined. """
    img_width, img_height = img_w, img_h
    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def compile_model(model):
    """compiling model with the required parameters. Please make the necessary changes based on your experiment."""
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def load_master_model(machine):
    """Loading the updated master model before each iteration"""
    return load_model(dir['machine_{}_master_model'.format(int(machine))] + 'master_model.h5')

def create_server_connection():
    """connecting to the server node"""
    return Connection(host=params['ip_server'],connect_kwargs={"password":params['passwort']})

def get_model_name(machine):
    """gathering the name of the model"""
    for file in os.listdir(dir['machine_{}_home'.format(int(machine))]):
        if file.endswith(".h5"):
            return file

def get_model_file(machine):
    """gathering the model file based on the machine"""
    home_dir = dir['machine_{}_home'.format(int(machine))]
    for file in os.listdir(home_dir):
        if file.endswith(".h5"):
            return (os.path.join(home_dir, file))

def copy_model_to_archive(file):
    """Copying the model to the archive before deleting the files permanently"""
    dest_dir = dir['machine_2_archive']
    shutil.copy2(file, dest_dir)

def send_model_to_server(file,name):
    """Sending the generated model to the server"""
    con_server = create_server_connection()
    con_server.put(file, remote = dir['server_archive'] + name)

def delete_local_model(file):
    """#deleting the local model from the client node once the model has been transferred over to the server node."""
    os.remove(file)


def train_model(model, machine):
    """training the model. Returns the final trained model and the metrics of loss value and accuracy value after each step."""
    print()
    print("Training model at machine {}".format(int(machine)))

    train_data_dir = '/home/pi/Desktop/fl_{}/data/train'.format(int(machine))
    validation_data_dir = '/home/pi/Desktop/fl_{}/data/validation'.format(int(machine))
    nb_train_samples = train_samples
    nb_validation_samples = validation_samples
    epochs = nb_epochs
    batch_size = nb_batch_size
    img_width, img_height = img_w, img_h

    model = compile_model(model)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    history = model.fit_generator(
        train_generator,
        verbose=2,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    return model, history.history

def get_model_path(machine):
    """ Getting the path of the model for a particular machine """
    return dir['machine_{}_home'.format(int(machine))]

def get_model_name(machine):
    """ Getting the name of the model for a particular machine """
    return ('learner_{0}_model_{1}.h5'.format(machine, int(time.time())))
