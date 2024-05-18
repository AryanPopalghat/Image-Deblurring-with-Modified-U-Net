import sys
sys.path.append('E:/PythonPackages')
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms
import sys
sys.path.append('E:/PythonPackages')
import torch
from torchsummary import summary
from haar_pytorch import HaarForward, HaarInverse
import torchvision.transforms as transform
from PIL import Image
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import os


device = "cpu"

haar =HaarForward()
ihaar = HaarInverse()


sigma =nn.Sigmoid()
dropout=nn.Dropout()
class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()    

        self.conv1 = nn.Conv2d(6,32,kernel_size=3,padding=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        torch.nn.init.xavier_uniform_(self.conv2.weight) 
        torch.nn.init.xavier_uniform_(self.conv1.weight) 

        self.conv3 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
        torch.nn.init.xavier_uniform_(self.conv4.weight) 
        torch.nn.init.xavier_uniform_(self.conv3.weight) 

        self.conv5 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
        torch.nn.init.xavier_uniform_(self.conv6.weight) 
        torch.nn.init.xavier_uniform_(self.conv5.weight) 

        self.conv7 = nn.Conv2d(512,256,kernel_size=3,padding=1)
        self.batch4 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
        torch.nn.init.xavier_uniform_(self.conv8.weight) 
        torch.nn.init.xavier_uniform_(self.conv7.weight) 
        
        self.conv9 = nn.Conv2d(1024,512,kernel_size=3,padding=1)
        self.conv10 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv11 = nn.Conv2d(512,1024,kernel_size=3,padding=1)
        torch.nn.init.xavier_uniform_(self.conv11.weight) 
        torch.nn.init.xavier_uniform_(self.conv10.weight) 
        torch.nn.init.xavier_uniform_(self.conv9.weight) 


        self.upconv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.atten1 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.upconv1.weight) 
        torch.nn.init.xavier_uniform_(self.d11.weight)         
        torch.nn.init.xavier_uniform_(self.d12.weight)

        self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.atten2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.upconv2.weight)     
        torch.nn.init.xavier_uniform_(self.d21.weight)     
        torch.nn.init.xavier_uniform_(self.d22.weight)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.atten3 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.upconv3.weight) 
        torch.nn.init.xavier_uniform_(self.d31.weight)         
        torch.nn.init.xavier_uniform_(self.d32.weight)

        self.upconv4 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.atten4 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.d41 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        torch.nn.init.xavier_uniform_(self.upconv4.weight) 
        torch.nn.init.xavier_uniform_(self.d41.weight)         
        torch.nn.init.xavier_uniform_(self.d42.weight)

        self.outconv = nn.Conv2d(64,3,kernel_size=1 )        
        torch.nn.init.xavier_uniform_(self.outconv.weight)
        

    def forward(self,x):
        
        x1=F.relu(self.conv1(x))
        x2=F.relu(self.conv2(x1))
        pool1=self.pool1(x2)
        pool1 = haar(pool1)

        x3=F.relu(self.conv3(pool1))
        x4=F.relu(self.conv4(x3))
        pool2=self.pool2(x4)
        pool2 = haar(pool2)

        x5=F.relu(self.conv5(pool2))
        x6=F.relu(self.conv6(x5))
        pool3=self.pool3(x6)
        pool3 = haar(pool3)

        x7=F.relu(self.conv7(pool3))
        x8=F.relu(self.conv8(x7))
        pool4=self.pool4(x8)
        pool4 = haar(pool4)

        x9=F.relu(self.conv9(pool4))
        x10=F.relu(self.conv10(x9))
        x11=sigma(self.conv11(x10))
        x11 = ihaar(x11)
    

        xu1=self.upconv1(x11)
        d= self.atten1(x8)
        x8 = x8*F.relu(d)+x8
        print(x8.shape)
        print(xu1.shape)
        xu11=torch.cat([xu1,x8],dim=1)
        xu11 = dropout(xu11) 
        xd11=sigma(self.d11(xu11))
        xd12=sigma(self.d12(xd11))
        xd12 = ihaar(xd12)
        
        xu2 = self.upconv2(xd12)
        c= self.atten2(x6)
        x6 = x6*F.relu(c)+x6
        xu22 = torch.cat([xu2, x6], dim=1)
        xu22 =dropout(xu22)
        xd21 = sigma(self.d21(xu22))
        xd22 = sigma(self.d22(xd21))
        xd22 =ihaar(xd22)

        xu3 = self.upconv3(xd22)
        b = self.atten3(x4)
        x4 = x4*F.relu(b) + x4
        xu33 = torch.cat([xu3, x4], dim=1)
        xu33 = dropout(xu33)
        xd31 = sigma(self.d31(xu33))
        xd32 = sigma(self.d32(xd31))
        xd32 = ihaar(xd32)

        xu4 = self.upconv4(xd32)
        a = self.atten4(x2)
        x2 = x2*F.relu(a) +x2
        xu44 = torch.cat([xu4, x2], dim=1)
        xu44 = dropout(xu44)
        xd41 = sigma(self.d41(xu44))
        xd42 = sigma(self.d42(xd41))
        
        x=self.outconv(xd42)
        return sigma(x)

# Define model
model =UNet()

#Load 
checkpoint_path = "D:/Desktop/GNR638/Project2/train/blurred_images/new_data.pt"
model = torch.load(checkpoint_path, map_location=device)

# Set model to evaluation mode
model.eval()

# Define the transformation pipeline for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define a function to generate output image
def generate_output_image(input_image_path, output_image_path,img):
    input_image = Image.open(input_image_path)
    
    # Convert the input image and blur image to tensors
    blur_image = transform(input_image)
    blur = input_image.convert("HSV")
    hsv_img = transform(blur)
    
    # Concatenate the blur and HSV images
    input_tensor = torch.cat([blur_image,hsv_img],dim=0)
    with torch.no_grad():
        input_tensor=input_tensor.unsqueeze(0)
        output_tensor = model(input_tensor)
    
    output_image=output_tensor.detach()
    save_image(output_image,output_image_path)
    

# Define input and output paths
input_path = "D:/Desktop/GNR638/Project2/mp2_test/custom_test/blur"
output_path = "D:/Desktop/GNR638/Project2/mp2_test/custom_test/created_sharp/"

# Generate output images for all images in the input path
for img in os.listdir(input_path):
    generate_output_image(os.path.join(input_path,img), os.path.join(output_path,img),img)



