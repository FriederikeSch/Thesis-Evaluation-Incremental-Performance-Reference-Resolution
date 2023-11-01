# %%
import pickle
import torch
from PIL import Image, ImageDraw, ImageFont
import os
import os.path as osp

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--splitBy', default='unc', help='split By')
parser.add_argument("--partition",default="test",help="test or validation part")
	

args = parser.parse_args()




# %%
def load_data(model,mode, dataset, split):
#load predicted and gold bounding boxes

    try:

        #the predicted bounding box
        with open(r"/home/users/fschreiber/project/bboxes_"+model+"/"+dataset+"/"+split+"_pred_bbox_list.p","rb") as f:
            pred_bbox_list=list(pickle.load(f))

        if mode=="non_inc":
            #the target bounding box
            with open(r"/home/users/fschreiber/project/bboxes_noninc_"+model+"/"+dataset+"/"+split+"_pred_bbox_list.p","rb") as f:
                target_bbox_list=list(pickle.load(f))

        elif mode == "inc":
            #the target bounding box
            with open(r"/home/users/fschreiber/project/bboxes_"+model+"/"+dataset+"/"+split+"_target_bbox_list.p","rb") as f:
                target_bbox_list=list(pickle.load(f))
        else:
            print("The mode can only be non_inc or inc")
            return -1,-1,-1,-1,-1

        #the number of one sentence split up incrementally ("the left zebra" would have length 3)
        with open(r"/home/users/fschreiber/project/incremental_pickles/length_incremental_units/"+dataset+"_"+split+"_length_unit.p","rb") as f:
            inc_len=pickle.load(f)

        #the original model data split up incrementally
        data_model=torch.load("/home/users/fschreiber/project/ready_inc_data/"+dataset+"/"+dataset+"_"+split+".pth")

        with open(r"/home/users/fschreiber/project/binary_grouped/"+model+"/"+mode+"/"+dataset+split+".p","rb") as f:
            binary_grouped=pickle.load(f)

        if mode=="non_inc":
            target_bbox_list=[x for x,y in zip(target_bbox_list,inc_len) for _ in range(y)]
            
        if model=="TVG":
            pred_bbox_list,target_bbox_list=TVG_prep(pred_bbox_list,target_bbox_list)

            for i in range(len(data_model)):
    
                path="/home/users/fschreiber/project/TransVG/ln_data/other/images/mscoco/images/train2014/"+data_model[i][0]
                image = Image.open(path)
                image_width, image_height = image.size
                
                pred_bbox_list[i]=transform_coordinates(pred_bbox_list[i],image_width,image_height)
                target_bbox_list[i]=transform_coordinates(target_bbox_list[i],image_width,image_height)
                    


        return pred_bbox_list,target_bbox_list,inc_len,data_model,binary_grouped
    

    except FileNotFoundError as e:
        print(e)
        
        return  -1,-1,-1,-1,-1
    

#TVG needs some extra adjustments to fit the same data format as Resc
def TVG_prep(pred_bbox_list,target_bbox_list):
    #print("TVG")
    for ind,(pred,targ) in enumerate (zip (pred_bbox_list,target_bbox_list)):

        pred=pred.view(1,-1)

        pred=xywh2xyxy(pred)
        pred=torch.clamp(pred,0,1)

        pred_bbox_list[ind]=pred

        targ=targ.view(1,-1)
        targ=xywh2xyxy(targ)

        target_bbox_list[ind]=targ
    return pred_bbox_list,target_bbox_list

#copied from TransVG needed to transform the bounding box vectors
def xywh2xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

#TVG coordinates are normalized between 0 and 1 reshape them to fit the image
def transform_coordinates(normalized_coords, image_width, image_height):
    # Multiply the normalized coordinates by image size
    pixel_coords = normalized_coords * torch.tensor([[image_width, image_height, image_width, image_height]])

    # Round the pixel coordinates to integers
    #pixel_coords = pixel_coords.round()

    return pixel_coords



# %%
#Load a dataset

params = vars(args)
pred_bbox_list,target_bbox_list,inc_len,model,binary_grouped=load_data("ReSc","inc",params["splitBy"],params["partition"])


# %%
#copied from ReSC
def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)

    inter_rect_y1 = torch.max(b1_y1, b2_y1) 

    inter_rect_x2 = torch.min(b1_x2, b2_x2)
   
    inter_rect_y2 = torch.min(b1_y2, b2_y2)


    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    

    #print("inter area",inter_area)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)
  
# %%
#group sentences that belong to one incremental unit
def group_by_increment(bbox_list,inc_len):
    counter=0
    group_list=[]
    for i in inc_len:
        
        group_list.append(bbox_list[counter:counter+i])
        counter=counter+i
    return group_list


pred_group=group_by_increment(pred_bbox_list,inc_len)
targ_group=group_by_increment(target_bbox_list,inc_len)
model_group=group_by_increment(model,inc_len)

# %%
def normalize_bounding_boxes(tensor_data, image_width, image_height):
    # Normalize the tensor_data by dividing each element by the corresponding image width or height
    normalized_data = tensor_data / torch.tensor([image_width, image_height, image_width, image_height], dtype=torch.float32)

    return normalized_data

# %%

img_coord=[]
norm_pred=[]
for i in range(len(pred_group)):
    
    path_l="/home/users/fschreiber/project/TransVG/ln_data/other/images/mscoco/images/train2014/"+model_group[i][0][0]

    image_l= Image.open(path_l)

    image_width_l, image_height_l = image_l.size

    img_coord.append((image_width_l,image_height_l))

    norm_pred.append(normalize_bounding_boxes(pred_group[i][0],img_coord[i][0],img_coord[i][1]))

all_iou=[]
counter=0
for i in range(len(pred_group)):

    counter=counter+1
    if counter%10==0:
        print(counter)

    for j in range(len(pred_group)):
        all_iou.append(bbox_iou(norm_pred[i],norm_pred[j]))


# %%
print(sum(all_iou)/len(all_iou))

