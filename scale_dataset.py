import os
import shutil
import json
import string
from pycocotools.coco import COCO
import cv2
import numpy as np
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set Dataset Scaler', add_help=False)
    parser.add_argument('--target_size', default=500, type=int)
    parser.add_argument('--input_file', default='val', type=str)
    parser.add_argument('--data_path', default='val/', type=str)
    parser.add_argument('--output_dir', default='scaled_val/', type=str)
    
    return parser
    
# Base COCO DICT
coco_output = {
            "info": {
                "description": "Automated Annotation Pipeline",
                "url": "http://hello.org",
                "version": "1.0",
                "year": 2022,
                "contributor": "John Doe",
                "date_created": "2017/09/01"
            },
            "licenses": [
                {
                    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                    "id": 1,
                    "name": "No License"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [] # insert contents from pano_class.json 
            # "segment_info": [...] <-- Only in Panoptic annotations
    }


def make_scaled_output_dir(args):
    """ Create output dir, delete existing directory if exists"""
    output_dir = args.output_dir #'scaled_val'               # names of output dir
    shutil.rmtree(output_dir)               # remove dir and it contents
    os.makedirs(output_dir, exist_ok=True)  # make new dir
    

def dict_to_json():
    """ Outputs coco dict as JSON file """
    # output to file
    with open("scaled_val.json", "w") as write_file: 
        json.dump(coco_output, write_file, indent=4)
    
    print("outputted to file: ", "scaled_val.json")


def main(args):
    print("SCALE DATASET")
    
    # create output dir
    make_scaled_output_dir(args)

    # read file
    dataDir=''
    dataType = args.input_file
    annFile='{}{}.json'.format(dataDir,dataType)

    # Initialize the COCO api for instance annotations
    coco=COCO(annFile)

    # Load the categories in a variable
    catIDs = coco.getCatIds()
    categories = coco.loadCats(catIDs)
    
    print(categories)

    # add category section
    print("Adding Categories...")
    for category in categories:
        print(category)
        coco_output["categories"].append(category)

    print("Scale Images...")
    # extract image ids
    for img_id in coco.getImgIds():
        print("id: ", img_id)
        
        # load img, only 1 img in arr
        img_obj = coco.loadImgs(ids=[img_id])[0] 
        print(img_obj)
        # imageToPredict = cv2.imread("img.jpg", 3)
        image_file_name = img_obj['file_name']
        print("image: ", image_file_name)
        img_path = args.data_path + img_obj['file_name']
    
        imageToPredict = cv2.imread(img_path, 3)
        print(imageToPredict.shape)

        # Note: flipped comparing to your original code!
        y_ = imageToPredict.shape[0]
        x_ = imageToPredict.shape[1]
        targetSize = args.target_size # 500
        x_scale = targetSize / x_
        y_scale = targetSize / y_
        print("scale: (x, y): ", x_scale, y_scale)
        img = cv2.resize(imageToPredict, (targetSize, targetSize))
        
        print(img.shape)
        im_file_name = image_file_name
        cv2.imwrite(args.output_dir + im_file_name, img)
        img = np.array(img)
        
        # add to images
        anno_image = {
                "id": int(img_id),         # current image_id in class
                "width": targetSize,   
                "height": targetSize, 
                "file_name": os.path.basename(im_file_name) 
            }    

        coco_output["images"].append(anno_image)

        anno_ids = coco.getAnnIds(imgIds=[img_id])
        print("ANNO_IDS", anno_ids)
        annotations = coco.loadAnns(ids=anno_ids)

        print("Annotations: ", annotations)
        scaled_bboxs=[]

        for anno in annotations:
        # original frame as named values
            (origLeft, origTop, origRight, origBottom) = anno['bbox'] #(160, 35, 555, 470)
            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            xmax = int(np.round(origRight * x_scale))
            ymax = int(np.round(origBottom * y_scale))
            
            """
            [x_min, y_min, width, height]
            They are coordinates of the top-left corner along with the width and height of the bounding box.
            """
            # add anno
            new_anno = {'area': (xmax * ymax), 
                        'attributes': {'occluded': False, 'rotation': 0.0}, 
                        'bbox': [x, y, xmax, ymax], 
                        'category_id': anno['category_id'], 
                        'id': anno['id'],  
                        'image_id':  anno['image_id'],
                        'iscrowd': 0, 
                        #'point': [643.0, 817.0], 
                        'segmentation': []}
            coco_output['annotations'].append(new_anno)
            # scaled_bboxs.append([1, 0, x, y, xmax, ymax])
            scaled_bboxs.append([1, 0, x, y, xmax, ymax])

    
    # Write out internal div
    dict_to_json()
        

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    main(args)