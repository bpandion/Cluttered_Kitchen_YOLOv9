import os
import json,yaml
import numpy as np
from roboflow import Roboflow

en_lvis=False
#COCO
    # project.version(5) #all photos
    # project.version(6) #low clutter
#LVIS
    # project.version(7) #Full dataset
    # project.version(6) #All bright
    # project.version(5) #All dark
    # project.version(4) #Clutter L
    # project.version(3) #Clutter M
    # project.version(2) #Clutter S
vn=7 #choose version number for automation

if True: #choose false if download is already done and only label fix is neccesary
    rf = Roboflow(api_key="0kmGWteb4DO0DqJvJg8T")
    # download dataset in coco and yolov9 format
    if en_lvis:
        project = rf.workspace("lvarobotvision").project("kitchenlvis")
    else:
        project = rf.workspace("lvarobotvision").project("kitchencocov2")
    version = project.version(vn) #Clutter S

    dataset = version.download("yolov9")

if en_lvis:
    master_path = '.\\KitchenLVIS-'+str(vn)
    old_label_path = master_path+'\\test\\labels' #for lvis
else:
    master_path ='.\\KitchenCOCOv2-'+str(vn)
    old_label_path = master_path+'\\valid\\labels' #for coco
    
if not os.path.exists(master_path+'\\new_labels'):
    os.makedirs(master_path+'\\new_labels')
    
new_label_path = master_path + '\\new_labels'


print('\n##### Fixing Labels #####')
# fix lables
txts = os.listdir(old_label_path)
filecounter=1
used_ids =[]

if en_lvis:
    with open('data/lvis_adapted.yaml','r',encoding='utf8') as file:
        coco_yaml_data = yaml.safe_load(file)

    
    with open('KitchenLVIS-'+str(vn)+'/data.yaml','r') as file:
        kitchencoco_yaml_data = yaml.safe_load(file)
else:
    with open('data/coco.yaml','r',encoding='utf8') as file:
        coco_yaml_data = yaml.safe_load(file)

    with open('KitchenCOCOv2-'+str(vn)+'/data.yaml','r') as file:
        kitchencoco_yaml_data = yaml.safe_load(file)
    
lvis_names = coco_yaml_data['names']
kitchenlvis_names = kitchencoco_yaml_data['names']

for txt in txts:
    edited_names=""
    with open(old_label_path+'\\'+txt, "r") as file:
        # annotation txt auslesen und zerlegen
        content =  file.readlines()
        split_content = []
        for line in content:
            split_content.append(line.split(' '))
        # class id ändern
        for line in split_content: #class id in all labels in txts
            classname = kitchenlvis_names[int(line[0])]
            if classname in lvis_names.values():
                edited_names=edited_names+classname+", "
                for ci in lvis_names:
                    if (lvis_names[ci] == classname):
                        line[0] = str(ci)
                        if ci not in used_ids:
                            used_ids.append(ci)
                        break
            else:
                print("\n### ERROR: classname \""+ classname + "\" not found! ###\n")
            
            
    # txt wieder zusammensetzen
    
    full_string = ''
    for s in split_content:
        full_string = full_string + ' '.join(s)
    
    # write editierte txt 
    # with open(path+'\\edited\\'+txt, "w") as file:
    with open(new_label_path+'\\'+txt, "w") as file:
        file.write(full_string)
    print(str(filecounter) + " File edited, Filename: " + new_label_path+'\\'+txt)
    print("classes: " + edited_names+"\n")
    filecounter = filecounter+1
        
# editing yaml 
if en_lvis:
    with open('KitchenLVIS-'+str(vn)+'/data.yaml','w') as file:
        new_yaml_data = kitchencoco_yaml_data
        new_yaml_data['names'] = lvis_names
        yaml.dump(new_yaml_data,file)
else:
    with open('KitchenCOCOv2-'+str(vn)+'/data.yaml','w') as file:
        new_yaml_data = kitchencoco_yaml_data
        new_yaml_data['names'] = lvis_names
        yaml.dump(new_yaml_data,file)      

    
    
print("DONE!")
# Was automated:
# print("NEXT MANUAL STEPS:")
# print("-Delete old Labels in .\\KitchenLVIS-"+str(vn)+"\\valid\\labels.")
# print("-Copy and paste new Labels from .\\KitchenLVIS-"+str(vn)+"\\new_labels to .\\KitchenLVIS-"+str(vn)+"\\valid\\labels.")
# print("")
used_ids.sort()
print("Used IDs: "+ str(used_ids))

with open(master_path+'\\UsedIDs.txt','w') as file:
    file.write(str(used_ids))


if en_lvis:
    for i in os.listdir(old_label_path):
        os.remove(old_label_path+'\\'+i)
    os.rmdir(old_label_path)
    os.rename(new_label_path,old_label_path)
    os.rename(master_path+'\\test',master_path+'\\valid')
else:
    for i in os.listdir(old_label_path):
        os.remove(old_label_path+'\\'+i)
    os.rmdir(old_label_path)
    os.rename(new_label_path,old_label_path)