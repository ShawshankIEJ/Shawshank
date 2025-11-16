import os
from dataclasses import dataclass
import os
import json  
import datetime

import cv2
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import time
import math
import random
import csv

from tqdm import tqdm
import pandas as pd
from utils import *
from utils_agent import Agent
from attack import project_text_on_image, save_image, RejectAttack, MaliciousAttack, BadRobot, PAIR, Multilingual
from qwen_defense import evaluation_input, evaluation_output
from config import *
import argparse




def process_single_Reject_Attack(command,image_filename):
    print(f"Image filename: {image_filename}")
    print(f"Command: {command}")
    image = load_image_with_opencv(image_filename)

    attack_agent =  RejectAttack(QWEN_MAX,QWEN_VL,QWEN_MAX)
    best_attack = attack_agent.generate_attack_text(command, image, is_vlm=False, n_iterations=10)
    attack_response = best_attack["attack_text"]

    print(f"Best Attack Agent Response: {attack_response}")
    return attack_response

def process_single_Malicious_Attack(command,image_filename):
    print(f"Image filename: {image_filename}")
    print(f"Command: {command}")
    image = load_image_with_opencv(image_filename)

    attack_agent =  MaliciousAttack(QWEN_MAX,QWEN_VL,QWEN_MAX)
    best_attack = attack_agent.generate_attack_text(command, image, is_vlm=False, n_iterations=20)
    attack_response = best_attack["attack_text"]
    print(f"Best Attack Agent Response: {attack_response}")
    return attack_response

def process_single_BadRobot_Attack(attack_method,command,image_filename):

    attack_agent = BadRobot(QWEN_MAX)

    if attack_method == "contextual jailbreak":
        attack_response = attack_agent.contextual_jailbreak(command)

    elif attack_method == "safety misalignment":
        attack_response = attack_agent.safety_misalignment(command)
    
    elif attack_method == "conceptual deception":
        attack_response = attack_agent.conceptual_deception(command)
    return attack_response

def process_single_Multilingual_Attack(command,image_filename):
    print(f"Image filename: {image_filename}")
    print(f"Command: {command}")
    # image = load_image_with_opencv(image_filename)
    attack_agent = Multilingual(QWEN_PLS)
    attack_response = attack_agent.conceptual_deception(command)

    return attack_response
   





def generate_attack():
    print("Generate attack")
    input_csv = ""  

    output_csv = ""

    result_list = read_csv_and_return_list(input_csv)

    total = 0

    
    for instruction_data in tqdm(result_list, desc="Processing", unit="item"):
        total += 1
        print(total)
        instruction_type = instruction_data[0]
        instruction = instruction_data[1]
        scenario = instruction_data[2]
        address = instruction_data[3]
        image_filename = "./data/" + scenario + "/" +  address + ".jpg"
        attack_response = process_single_Malicious_Attack(instruction,image_filename)
        # attack_method = "contextual jailbreak"
        # attack_method = "safety misalignment"
        # attack_method = "conceptual deception"
        # attack_response = process_single_BadRobot_Attack(attack_method,instruction,image_filename)
        # attack_response = process_single_Multilingual_Attack(instruction,image_filename)
        # attack_response = process_single_Reject_Attack(instruction,image_filename)

        fieldnames = ["Type", "Instruction", "Scenario", "Address", "Attack_response"]
        save_data = {
            "Type": instruction_type,
            "Instruction": instruction,
            "Scenario": scenario,
            "Address": address,
            "Attack_response": str(attack_response)
        }
        save_to_csv_one_by_one(save_data,fieldnames,output_csv)



def evaluate_attack():
    print("Evaluate Attack")

    input_csv = ""
    output_csv = ""

    result_list = read_csv_and_return_list(input_csv)

    total = 0

    for instruction_data in tqdm(result_list, desc="Processing", unit="item"):
        total += 1
        print(total)
        instruction_type = instruction_data[0]
        instruction = instruction_data[1]
        scenario = instruction_data[2]
        address = instruction_data[3]
        attack_response = instruction_data[4]
        image_filename = "./data/" + scenario + "/" +  address + ".jpg"

        image = load_image_with_opencv(image_filename)
        attack_image = project_text_on_image(image, attack_response, (50, 200), font_scale=2, font_thickness=4, alpha=0.1, target_size=(640, 480), line_height=100, max_line_width=30)
        # command = ""
        response = action_planning(attack_image,instruction,GEMINI,PROMPT_GEN)
        evaluate_response = evaluation_ai(response,instruction,GPT_VL,PROMPT_DOS)
        evaluate_data = safe_json(evaluate_response)
        success = evaluate_data.get("success", 1)
        score = safe_float(evaluate_data.get("score", 0))

        fieldnames = ["Type", "Instruction", "Scenario", "Address", "Attack_response", "Response", "Success", "Score"]
        save_data = {
            "Type": instruction_type,
            "Instruction": instruction,
            "Scenario": scenario,
            "Address": address,
            "Attack_response": str(attack_response),
            "Response": str(json.dumps(safe_json(response))),
            "Success": success,
            "Score": score
        }
        save_to_csv_one_by_one(save_data,fieldnames,output_csv)


def calculate_score():
    folders = [""]
    expected_columns = ['Type', 'Instruction', 'Scenario', 'Address', 'Attack_response', 'Response', 'Success', 'Score']
    # expected_columns = ["Type", "Instruction", "Scenario", "Address", "Attack_response", "Response", "Success", "Score","Safe_label_input","Safe_label_output"]
    
    result_list = read_csv_and_return_list11(folders,expected_columns)
    total = 0
    success_count = 0
    scores = []
    instruction_stats = {}

    # Process the instructions
    for instruction_data in tqdm(result_list, desc="Processing", unit="item"):
        total += 1
        instruction_type = instruction_data[0]
        instruction = instruction_data[1]
        scenario = instruction_data[2]
        address = instruction_data[3]
        success = instruction_data[6] 
        score = instruction_data[7]   
        # safe_label_input = instruction_data[8]
        # safe_label_output = instruction_data[9]
        
        # if success and safe_label_output == "Safe":
        if success:
            success_count += 1
        scores.append(score)
        

        if instruction_type not in instruction_stats:
            instruction_stats[instruction_type] = {
                'success_count': 0,
                'scores': []
            }
        
        # if success and safe_label_output == "Safe" :
        if success:
            instruction_stats[instruction_type]['success_count'] += 1
        instruction_stats[instruction_type]['scores'].append(score)

    success_rate = success_count / total
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    variance_score = np.var(scores)

    print(f"\nOverall Success Rate: {success_rate:.2f}")
    print(f"Overall Mean Score: {mean_score:.2f}")
    print(f"Overall Median Score: {median_score:.2f}")
    print(f"Overall Variance of Scores: {variance_score:.2f}")

    for instruction_type, stats in instruction_stats.items():
        type_success_rate = stats['success_count'] / len(stats['scores'])
        type_mean_score = np.mean(stats['scores'])
        type_median_score = np.median(stats['scores'])
        type_variance_score = np.var(stats['scores'])
        
        print(f"\nStatistics for Instruction Type: {instruction_type}")
        print(f"  Success Rate: {type_success_rate:.2f}")
        print(f"  Mean Score: {type_mean_score:.2f}")
        print(f"  Median Score: {type_median_score:.2f}")
        print(f"  Variance of Scores: {type_variance_score:.2f}")


def defense_test():
    folders = [""]
    output_csv = ""
    
    expected_columns = ['Type', 'Instruction', 'Scenario', 'Address', 'Attack_response', 'Response', 'Success', 'Score']
    
    result_list = read_csv_and_return_list11(folders,expected_columns)
    total = 0
    success_count = 0
    scores = []
    instruction_stats = {}

    for instruction_data in tqdm(result_list, desc="Processing", unit="item"):
        total += 1
        print(total)
        print()
        instruction_type = instruction_data[0]
        instruction = instruction_data[1]
        scenario = instruction_data[2]
        address = instruction_data[3]
        attack_response = instruction_data[4]
        response = instruction_data[5]
        success = instruction_data[6]  
        score = instruction_data[7]   

        safe_label_input, categories = evaluation_input(attack_response)
        safe_label_output, category_label, refusal_label = evaluation_output("",response)

        fieldnames = ["Type", "Instruction", "Scenario", "Address", "Attack_response", "Response", "Success", "Score","Safe_label_input","Safe_label_output"]
        save_data = {
            "Type": instruction_type,
            "Instruction": instruction,
            "Scenario": scenario,
            "Address": address,
            "Attack_response": attack_response,
            "Response": response,
            "Success": success,
            "Score": score,
            "Safe_label_input": safe_label_input,
            "Safe_label_output": safe_label_output
        }
        save_to_csv_one_by_one(save_data,fieldnames,output_csv)   



if __name__ == "__main__":
    generate_attack()
    evaluate_attack()
    calculate_score()
    # defense_test()