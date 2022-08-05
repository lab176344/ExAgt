import pathlib
import os
import pandas as pd
from contextlib import redirect_stdout
import src.create_table as create_table

BASE_FOLDER = pathlib.Path(__file__).parent.absolute() #.../SCENARIO-NET/src
BASE_FOLDER = os.path.join(BASE_FOLDER,'src')
head, _ = os.path.split(BASE_FOLDER)#.../SCENARIO-NET

# 1) Update table of models (DONE)
folder_name = "model"
path = os.path.join(BASE_FOLDER,folder_name)
create_table.update_table(path, folder_name)

filename = os.path.join(path,"table.csv")
df_table = pd.read_csv(filename)
with open(os.path.join(path,'README.md'), 'w') as f:
    with redirect_stdout(f):#Write in the file README.md
        print('**List of available models**')
        print(df_table.to_markdown(index=False))

# 2) Update table of dataloaders (DONE)
folder_name = "dataloader"
path = os.path.join(BASE_FOLDER,folder_name)
create_table.update_table(path, folder_name)

filename = os.path.join(path,"table.csv")
df_table = pd.read_csv(filename)
with open(os.path.join(path,'README.md'), 'w') as f:
    with redirect_stdout(f):#Write in the file README.md
        print('**List of available dataloaders**')
        print(df_table.to_markdown(index=False))

# # 3) Update table of  trains (TODO)
# folder_name = "train"
# path = os.path.join(BASE_FOLDER,folder_name)
# create_table.update_table(path, folder_name)

# filename = os.path.join(path,"table.csv")
# df_table = pd.read_csv(filename)
# with open(os.path.join(path,'README.md'), 'w') as f:
#     with redirect_stdout(f):#Write in the file README.md
#         print('**List of available trainers**')