from openvino.inference_engine import IECore,IENetwork
import os
import cv2
import numpy as np

class FaceReidentification:
    '''
    Class for the Face Reidentification model.
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
        model_name = 'face-reidentification-retail-0095'
        model_path = os.path.join(parent_dir,'model','intel',model_name,self.precision,model_name)
        return model_path

        
    def load_model(self):
        self.net = IECore().load_network(network = self.model, device_name = self.device,num_requests=1)

    def predict(self, face):
        self.preprocess_image  = self.preprocess_input(np.copy(face))
        self.net.start_async(request_id=0, inputs={self.input_name: self.preprocess_image})
        if self.net.requests[0].wait(-1) == 0:
            outputs = self.net.requests[0].outputs
            return self.preprocess_output(outputs,np.copy(face))


    def preprocess_input(self, image):
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2,0,1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs,face):
        return outputs[self.output_name].squeeze().reshape(1,-1)