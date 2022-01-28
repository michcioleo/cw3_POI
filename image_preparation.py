from PIL import Image
from glob import glob
from os.path import sep, join, splitext
from skimage.feature import greycomatrix, greycoprops
import numpy as np
from pandas import DataFrame
from itertools import product

feature_names = ('dissimilarity','contrast','correlation','energy','homogeneity','ASM')
distances = (1,3,5)
angles = (0, np.pi/4, np.pi/2, 3*np.pi/4)

def get_full_names():
    dist_str = ('1','3','5')
    angles_str = '0deg,45deg,90deg,135deg'.split(',')
    return['_'.join(f) for f in product(feature_names, dist_str, angles_str)]
#----------------------------------glcm matrix calculation----------------------------------------------
def get_glcm_feature_array(patch):
    patch_64 = (patch / np.max(patch)*63).astype('uint8')
    glcm = greycomatrix(patch_64, distances, angles, 64, True, True)
    feature_vector = []
    for feature in feature_names:
        feature_vector.extend(list(greycoprops(glcm, feature).flatten()))
    return feature_vector

texture_folder = "Textures"
samples_folder = "TextureSamples"
paths = glob(texture_folder + '\\*\\*.jpg')

fil2 = [p.split(sep) for p in paths]
_, categories, files = zip(*fil2)

size = 128,128

features = []
for category, infile in zip(categories, files):
    img = Image.open(join(texture_folder, category, infile))
    xr = np.random.randint(0,img.width-size[0],10)
    yr = np.random.randint(0, img.height-size[1], 10)
    base_name, _ = splitext(infile)
    for i,(x,y) in enumerate(zip(xr,yr)):
        img_sample=img.crop((x, y, x+size[0], y+size[1]))
        img_sample.save(
            join(samples_folder, category, f'{base_name:s}_{i:02d}.jpg'))
        img_grey = img.convert('L')
        feature_vector = get_glcm_feature_array(np.array(img_grey))
        feature_vector.append(category)
        features.append(feature_vector)

full_feature_names = get_full_names()
full_feature_names.append('Category')

df=DataFrame(data=features, columns=full_feature_names)
df.to_csv('textures_data.csv', sep=',', index=False)

#=================CLASSIFICATION=====================

#from sklearn import svm
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score

#classifier=svm.SVC()
#data = np.array(features)
#X=data[:,:-1]
#Y=data[:,-1]

#x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33)

#classifier.fit(x_train,y_train)
#y_pred=classifier.predict(x_test)
#acc=accuracy_score(y_test,y_pred)
#print(acc)