# TODO create model_table.txt: run over all model_x.py and collect info from model_x._get_description()

# import required module
import os
# from evaluation import *
# from dataset import *
from src.dataloader import *
from src.model import *
from importlib import import_module
import csv
from src.utils.natural_list_sort import natural_sort

def update_table(path, folder_name):
    # path = 'C:/Users/floresfernandez/Documents/GitHub/SCENARIO-NET/src/folder_name'
    # folder_name = dataloader or dataset or evaluation or model
    files = []
    table_list = []
    print(path)
    print(os.listdir(path))
    filter_list = natural_sort(list(filter(lambda k: folder_name+str("_") in k, os.listdir(path))))
    print(filter_list)
    for i in filter_list:
        print(i)
        if os.path.isfile(os.path.join(path,i)):
            id = i[:-3]
            print(id)
            constructor = globals()[id]
            print(constructor)
            #test = getattr(constructor, generate_model)()# test = constructor.model()
            test = constructor.generate_model()
            print(test._get_description())
            table_list.append(test._get_description())
  
    #WRITE CSV FILE
    if len(table_list) > 0:

        file = open(os.path.join(path,'table.csv'), 'w', newline ='')
        
        with file:
            # identifying header  
            header = list(table_list[0].keys())
            writer = csv.DictWriter(file, fieldnames = header)
            
            # writing data row-wise into the csv file
            writer.writeheader()
            for row in table_list:
                writer.writerow(row)

        return print("Updating table.csv of " + str(len(table_list)) + " " + folder_name + "s")
    


