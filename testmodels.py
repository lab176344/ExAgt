from src.model.model_0 import generate_model
print('**************************************')
print('************ RESNET 3D ***************')
print('**************************************')

resnet3d = generate_model(10,10)
print(resnet3d._get_description())
print(list(resnet3d.parameters()))


# print('**************************************')
# print('************ RESNET ******************')
# print('**************************************')
# from src.model.model_1 import generate_model
# resnet = generate_model(18,10)
# print(resnet)

# print('**************************************')
# print('*************** VIT ******************')
# print('**************************************')
# from src.model.model_2 import generate_model
# vit = generate_model(n_classes=10,image_size=200,patch=10,dim=1024,depth=6,head=16, mlp_dim = 2048)
# print(vit)
