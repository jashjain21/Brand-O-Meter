from django.shortcuts import render, redirect 
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from .models import otherDetails
from django.contrib.auth import authenticate, login, logout
from .forms import img
from django.contrib import messages
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from django.conf import settings

#tf.logging.set_verbosity(tf.logging.ERROR)
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
result_dic={}
# Patch the location of gfile
tf.gfile = tf.io.gfile
word_dic={}
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import CreateUserForm
# Create your views here.
def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))

  return model

PATH_TO_LABELS = 'C:\\Users\\windows\\Desktop\\hackathon\\SPIT_HACKATHON\\Github\\Demand-Forecasting\\Website\\labelmap.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
import pathlib
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = pathlib.Path('C:\\Users\\windows\\Desktop\\hackathon\\SPIT_HACKATHON\\Github\\Demand-Forecasting\\Website\\media\\imagesrec\\images')
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))

detection_model=tf.saved_model.load('C:\\Users\\windows\\Desktop\\hackathon\\SPIT_HACKATHON\\Github\\Demand-Forecasting\\Website\\saved_model')

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

int_to_word={1:'oreo',
2:'hide_and_seek',
3:'bourbon',
4:'dark_fantasy',
5:'jim_jam',
6:'chocopie'
}
def show_inference(model, image_path):
  global word_dic
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  boxes = np.squeeze(output_dict['detection_boxes'])
  classes = np.squeeze(output_dict['detection_classes'])
  scores = np.squeeze(output_dict['detection_scores'])
  #set a min thresh score, say 0.8
  min_score_thresh = 0.5
  bboxes = boxes[scores > min_score_thresh]
  cclasses=classes[scores > min_score_thresh]
    #get image size
  im_width, im_height = Image.open(image_path).size
  final_box = []
  for box in bboxes:
    ymin, xmin, ymax, xmax = box
    final_box.append([xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height])
#   print(final_box)
#   print(len(final_box))
#   print("Hello")    
#   print(cclasses)
#   print(len(cclasses))
  for coord,clas in zip(final_box,cclasses.tolist()):
#         print(coord)
#         print(clas)
        if int_to_word[clas] in word_dic:
            word_dic[int_to_word[clas]].append(coord)
        else:
            word_dic[int_to_word[clas]]=[coord]
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  img=(Image.fromarray(image_np))
  img.save(str(image_path)+'_new.png')
im_width,im_height=0,0

def testing():
    for image_path in TEST_IMAGE_PATHS:
        print("image path is ",image_path)
        global word_dic
        word_dic={}
        im_width, im_height = Image.open(image_path).size
        show_inference(detection_model, image_path)
        area_dic={}
        area=im_height*im_width
        for key,value in word_dic.items():
            area_dic[key]=0
            for x in value:
                    area_dic[key]+=abs((x[1]-x[0])*(x[3]-x[2]))
            area_dic[key]=(area_dic[key]/area)*100
        print("Area occupied is as follows in percent")
        print(area_dic)
        position_dic={}
        directions = {
            1:['left','top'],
            2:['middle','middle'],
            3:['right','bottom']
        }
        #     area=im_height*im_width

        for key,value in word_dic.items():
                z = [0,0]
                for x in value:
        #             print(x[0],x[1],x[2],x[3])
        #             print(key)
                    z[0]+=(x[1]+x[0])/2
                    z[1]+=(x[3]+x[2])/2
                z[0]/=len(value)
                z[1]/=len(value)
                #print(z)
                p = im_width/3
                q = im_height/3
                position_dic[key] = [0,0]
                for i in range(3,0,-1):
                    if z[0]<p*i:
                        position_dic[key][0] = i
                    if z[1]<q*i:
                        position_dic[key][1] = i
                position_dic[key][0] = directions[position_dic[key][0]][0]
                position_dic[key][1] = directions[position_dic[key][1]][1]
        print("Position is as follows")
        print(position_dic)  
        result_dic[str(image_path)[:-4]+'new.jpg']=[area_dic,position_dic]
        print(result_dic)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('')
    else:
        print(request.method)
        if request.method == 'POST':
            if 'login' in request.POST:
                username = request.POST.get('username')
                password = request.POST.get('password')

                user = authenticate(request, username=username, password=password)

                if user is not None:
                    login(request, user)
                    
                    return redirect('')
                else:
                    messages.info(request, 'Username OR password is incorrect')
            elif 'register' in request.POST:
                if request.user.is_authenticated:
                    return redirect('login')
                else:
                    form = CreateUserForm()
                    if request.method == 'POST':
                        username = request.POST.get('username')
                        password = request.POST.get('password')
                        prodName = request.POST.get('prodName')
                        
                        user = User.objects.create_user(username=username, password=password)
                        # user.userprofile.user = authenticate(username=username, password=password)
                        user.save()
                        user = authenticate(username=username, password=password)
                        
                        profile = Profile()
                        profile.user = user
                        profile.prodName = prodName
                        profile.save()

                        login(request, user)
                        messages.success(request, 'Account was created')
                        print("Account was created")


                        return redirect('login')

        return render(request, 'index.html')
            
def logoutUser(request):
	logout(request)
	return redirect('login')

def bulk(request):
    if request.method == "POST":
        my_file = request.FILES.get("file")
        otherDetails.objects.create(image = my_file)
        return redirect("/bulk")
    else:
        form = img()
        return render(request, 'bulk.html')
def main(request):
    global PATH_TO_LABELS
    global TEST_IMAGE_PATHS  
    if request.method == "POST":
        print("helsslo")
        PATH_TO_TEST_IMAGES_DIR = pathlib.Path('C:\\Users\\windows\\Desktop\\hackathon\\SPIT_HACKATHON\\Github\\Demand-Forecasting\\Website\\media\\imagesrec\\images')
        TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
        testing()
    return HttpResponse("hello")