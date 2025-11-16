import os
import json 
import cv2
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import random
import base64
from openai import OpenAI
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import faiss
import io
import argparse


parser = argparse.ArgumentParser(description='Processing named parameters')
args = parser.parse_args()
scene_name = "FloorPlan330"


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./CLIPModel")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./CLIPModel").to(device)


def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor_clip(images=image, return_tensors="pt").to(device)
        image_features = model_clip.get_image_features(**inputs)
        return image_features


def normalizeL2(embeddings):
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    return vector

def add_vector_to_index(embedding, index):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    index.add(vector)


def create_faiss_index(image_folder_path):
    index_clip = faiss.IndexFlatL2(512)  
    images = []
    
    for root, dirs, files in os.walk(image_folder_path):
        for file in files:
            if file.endswith('jpg'):
                images.append(root + '/' + file)

    for image_path in images:
        img = Image.open(image_path).convert('RGB')
        clip_features = extract_features_clip(img)
        add_vector_to_index(clip_features, index_clip)
    
    faiss.write_index(index_clip, "clip.index")
    return index_clip, images

def decode_image_from_bytes(img_b64):
    img_bytes = base64.b64decode(img_b64)  
    img_bytes_io = io.BytesIO(img_bytes)  
    img = Image.open(img_bytes_io)  
    return img

def compare_image_with_folder(input_image_b64, image_folder_path, similarity_threshold=0.8):

    input_image = decode_image_from_bytes(input_image_b64)

    image_features_clip = extract_features_clip(input_image)
    image_features_clip = normalizeL2(image_features_clip)

    if not os.path.exists("clip.index"):
        index_clip, _ = create_faiss_index(image_folder_path)
    else:
        index_clip = faiss.read_index("clip.index")
    

    d_clip, i_clip = index_clip.search(image_features_clip, 3) 
    

    if d_clip[0][0] < (1 - similarity_threshold):  
        return 1
    else:
        return 0

def get_caption_from_gpt4o(img):

    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')

    qwen_api_key= ""
    qwen_base_url=''

    client = OpenAI(api_key=qwen_api_key, base_url=qwen_base_url)

    try:
        response = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "system",
                    "content": "you are a robot, please describe the scene you see."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "please describe what you see. exmple： I see a desk in the room."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + img_b64
                            }
                        }
                    ]
                }
            ],
            max_tokens=100
        )

        content = response.choices[0].message.content if response.choices and response.choices[0].message.content else None
        print('gpt-4o caption: ', content)
        if content is not None:
            return content.strip()
        else:
            return "No caption (gpt-4o returned empty content)"
    except Exception as e:
        print("Image description failed for GPT-4O; default caption used. Error message:", e)


controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    scene=scene_name,
    width=640,
    height=480,
    fieldOfView=90,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    renderSemanticSegmentation=False,
    renderObjectImage=False,
    renderClassImage=False,
    renderNormalsImage=False,
    renderImage=True  
    # platform=CloudRendering
) 


SAVE_DIR = "saved_scenes"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


scene_counter = 1


def save_scene_data(event, scene_name, similarity_threshold):
    state = 0
    global scene_counter  

    image_filename = os.path.join(SAVE_DIR, f"{scene_name}_{scene_counter:03d}.jpg")
    json_filename = os.path.join(SAVE_DIR, f"{scene_name}_{scene_counter:03d}.json")
    
    
    img = event.cv2img
    print(f"Scene photos have been saved as {image_filename}")
    visible_objects = [
        {
            "objectType": o['objectType'], 
            "objectId": o['objectId'],  
        }
        for o in event.metadata['objects'] if o['visible']
    ]
        

    agent_meta = event.metadata.get('agent', {})
    agent_meta = event.metadata.get('agent', {})
    current_position = [
        agent_meta.get('position', {}).get('x', 0.0),
        agent_meta.get('position', {}).get('y', 0.0),
        agent_meta.get('position', {}).get('z', 0.0)
    ]
    theta = agent_meta.get('rotation', {}).get('y', 0.0)

    scene_description = get_caption_from_gpt4o(img)
    
    scene_data = {
        "scene_name": scene_name,
        "scene_number": scene_counter, 
        "image_filename": image_filename,
        "agent_position": current_position,
        "agent_rotation": theta,
        "visible_objects": visible_objects,
        "scene_description": scene_description
    }
    if len(visible_objects) < 5:
        print("Rejected: Less than 5 visible objects.")
        return state

    state = compare_image_with_folder(img, SAVE_DIR, similarity_threshold)

    if state == 1:
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(scene_data, f, ensure_ascii=False, indent=4)
        print(f"Scene data has been saved as {json_filename}")
        cv2.imwrite(image_filename, img)
        scene_counter += 1
        return state
    return state

def main():
    num_iterations = 10 
    similarity_threshold = 0.8 

    for _ in range(num_iterations):
        event = controller.step(action="Pass")
        display_frame = event.cv2img
        cv2.imshow("View", display_frame)
        positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        position = random.choice(positions)
        event = controller.step(action="Teleport",position=position,rotation=dict(x=0, y=270, z=0),horizon=30,standing=True)
        event = controller.step(action="Pass")
        state = save_scene_data(event, scene_name, similarity_threshold)
        if state == 1:
            print("Scene data has been saved")
        else:
            print("Scene data has not been saved")

    cv2.destroyAllWindows()
    controller.step(action="Done")



if __name__ == "__main__":
    main()
