from openvino.inference_engine import IECore,IENetwork
import os
import cv2
import numpy as np

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, precision='FP32-INT8', device='CPU'):
        self.precision = precision
        self.model_path = self.get_model_path()
        self.model_weights=self.model_path+'.bin'
        self.model_structure=self.model_path+'.xml'
        self.device=device
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def get_model_path(self):
        parent_dir = os.path.split(os.getcwd())[0]
        model_name = 'face-detection-retail-0005'
        model_path = os.path.join(parent_dir,'model','intel',model_name,self.precision,model_name)
        return model_path

        
    def load_model(self):
        self.net = IECore().load_network(network = self.model, device_name = self.device,num_requests=1)

    def predict(self, frame, conf=0.6):
        self.preprocess_image  = self.preprocess_input(np.copy(frame))
        self.net.start_async(request_id=0, inputs={self.input_name: self.preprocess_image})
        if self.net.requests[0].wait(-1) == 0:
            outputs = self.net.requests[0].outputs
            return self.preprocess_output(outputs,np.copy(frame),conf)


    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs,image,conf):
        faces = []
        cords = []
        for i,box in enumerate(outputs[self.output_name][0][0]):
            if box[2] >= conf:
                xmin = int(box[3] * 720)
                ymin = int(box[4] * 480)
                xmax = int(box[5] * 720)
                ymax = int(box[6] * 480)  
                h = xmax - xmin
                w = ymax - ymin
                x = int(xmin - (h*0.5*0.2))
                y = int(ymin - (w*0.5*0.2))
                h = int(h * 1.2)
                w = int(w * 1.2)
                x = np.clip(x,0,720)
                y = np.clip(y,0,480)
                h = np.clip(h,0,720)
                w = np.clip(w,0,480)
                image_copy = np.copy(image)
                faces.append(image_copy[y:y+w,x:x+h])
                cords.append([x,y,h,w])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
        return image,faces,cords