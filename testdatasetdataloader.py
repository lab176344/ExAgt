from src.dataset.dataset import dataset
import matplotlib.pyplot as plt
from src.dataloader.dataloader_0 import dataloader_0
import tqdm
from torchvision import  transforms
import time
def visualize_scenarios(data,flag=2):
    for k in range(1):
        for i in range(4):
            if flag==1:
                a = data[k,i,:,:,:]
            else:
                a = data[k,:,i,:,:]
            aa = a.reshape(30,120)
            stdName = '_OG_'+str(k)+'_' +str(i)+'.png'
            plt.imshow(aa)
            plt.savefig(stdName)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([40,180]),
    transforms.RandomCrop([30,120]),
    transforms.RandomRotation(degrees=(5, -5), fill=(0,)),
    transforms.ToTensor(),
    #transforms.Normalize([0.123], [0.325]),
    transforms.RandomErasing(),

])

dataset_local = dataset(name='argoverse',mode='train',bbox_meter=[30.0,120.0],bbox_pixel=[30,120],center_meter=[15.0,30.0])
loader = dataloader_0(dataset=dataset_local,
                    batch_size=2,
                    epochs = 10,
                    num_workers=0,
                    shuffle=False,
                    transformation=train_transforms,
                    transformation3D=None,
                    test=False,
                    grid_chosen=[0, 3, 6, 9],
                    representation='image')
print('Training Data: ', len(loader(0).dataset))
program_starts = time.time()

for  ((x1,x2),idx) in loader(0):
    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))
    x1
    #visualize_scenarios(x1,flag=2)

