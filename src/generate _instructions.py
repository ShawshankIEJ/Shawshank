from utils_agent import Agent
from utils import read_txt, load_image_and_commands, load_and_parse_json, extract_image_filename_and_command, load_image_with_opencv, extract

import os
import json
import csv
import glob


from config import QWEN_VL


def generate_test_instructions(scene_data, image):
    PROMPT_BUILD= read_txt('./prompts/build_gen.txt')
    agent = Agent.from_config(QWEN_VL,PROMPT_BUILD)
    scene_description = f"Scene name: {scene_data['scene_name']}. " \
                    f"Agent position: {scene_data['agent_position']}. " \
                    f"Agent rotation: {scene_data['agent_rotation']} degrees. " \
                    f"Visible objects: {', '.join([obj['objectType'] for obj in scene_data['visible_objects']])}. " \
                    f"Image features extracted from scene image."
    
    response = agent.chat_vlm2json(image,scene_description)
    return response

def split_type_instruction(s: str):
    for sep in (":", "："):
        if sep in s:
            t, instr = s.split(sep, 1)
            return t.strip(), instr.strip()
    parts = s.strip().split(None, 1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]

def save_to_csv_one_by_one(json_text,scene_name,image_path,output_csv):
    try:

        data = json.loads(json_text)

        if "instructions" not in data or "type" not in data:
            print("The JSON format is incorrect; the 'type' or 'instructions' field is missing.")
            return

        file_exists = os.path.exists(output_csv)
        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Type", "Instruction", "Scenario", "address"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            for instruction in data["instructions"]:
                # print(instruction)
                type_, instruction_ = split_type_instruction(instruction)
                writer.writerow({
                    "Type": type_,
                    "Instruction": instruction_,
                    "Scenario": scene_name,
                    "Address": image_path
                })


    except json.JSONDecodeError as e:
        print(f"{e}")
    except Exception as e:
        print(f"{e}")

def process_scenes_in_directory(directory_path):
    scene_file_paths = set()
    scene_name = os.path.basename(os.path.normpath(directory_path)) 

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".jpg"):
            scene_file_path = os.path.join(directory_path, file_name.replace(".jpg", ""))
            scene_file_paths.add(scene_file_path)
    sorted_scene_file_paths = sorted(scene_file_paths)

    return sorted_scene_file_paths, scene_name

def main():

    output_csv = "xx.csv"
    directory_path = ""  
    scene_file_paths, scene_name = process_scenes_in_directory(directory_path)
    print(scene_file_paths)
    for scene_file_path in scene_file_paths:
        image_path = os.path.basename(os.path.normpath(scene_file_path))
        print(image_path)
        json_file_path = scene_file_path+".json"
        image_file_path = scene_file_path + ".jpg"
        json_data = load_and_parse_json(json_file_path)
        image = load_image_with_opencv(image_file_path)
        response = generate_test_instructions(json_data,image)
        save_to_csv_one_by_one(response,scene_name,image_path,output_csv)

if __name__ == "__main__":
    main()