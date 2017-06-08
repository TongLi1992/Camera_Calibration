import grpc
import subprocess
import glob
import os
import sys
import cv2
import numpy as np
from PIL import Image, ExifTags
subprocess.call(["python", "-m", "grpc_tools.protoc", "-I=../grpc/", "--python_out=.", "--grpc_python_out=." , "../grpc/CalibrationService.proto"])

import CalibrationService_pb2
import CalibrationService_pb2_grpc

SERVER_ADDR = "0.0.0.0:8080"

def get_image(filename, count):
    img = Image.open(filename)
    width, height = img.size

    if height/width == 540/960:
        width, height = 960, 540
    elif width/height == 540/960:
        width, height = 540, 960
    else:
        print "wrong image aspect ratio"
        return RESULT_BAD_FORMAT, 0
 

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break
    if not img._getexif():
        print "No EXIF"
        return RESULT_BAD_FORMAT, 0
    exif = dict(img._getexif().items())

    if exif[orientation] == 3:
        img = img.rotate(180, expand=True)
    elif exif[orientation] == 6:
        img = img.rotate(270, expand=True)
        width, height = height, width
    elif exif[orientation] == 8:
        img = img.rotate(90, expand=True)
        width, height = height, width

    img.thumbnail((width, height), Image.ANTIALIAS)
    
    img = img.convert('L') # convert to grayscale

#    img.save("temp_"+str(count)+".jpg")
    img = np.array(img) # convert from PIL image to OpenCV image
    _, jpg = cv2.imencode('.jpg', img)

    print filename, 'width:', width, 'height:', height
    return jpg.tostring()

MAX_MESSAGE_LENGTH = 999999900


#channel = grpc.insecure_channel(SERVER_ADDR, options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH)])
#channel = grpc.insecure_channel(SERVER_ADDR, options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
channel = grpc.insecure_channel(SERVER_ADDR)
stub = CalibrationService_pb2_grpc.CalibrationServiceStub(channel)
images = CalibrationService_pb2.Images()

count = 0
for filename in glob.glob(os.path.join("./less_images/", "*")):
    if not filename.endswith(".jpg") and not filename.endswith(".JPG"):
        continue
    images.image.append(get_image(filename, count))    
    count += 1
r = stub.calibrate(images)
print("fx = " + str(r.fx))
print("fy = " + str(r.fy))
print("cx = " + str(r.cx))
print("cy = " + str(r.cy))
