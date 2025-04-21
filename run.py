import hydra
import numpy as np
from tqdm import tqdm
import pybullet as p
from PIL import Image
from perturbations.perturb import VIMAPerturb
import vima_bench
import re
import os
import dotenv
import shutil
dotenv.load_dotenv()

def get_object_to_pose(env, task):
    object_ids = env.obj_ids['fixed']+env.obj_ids['rigid']
    object_id2pose={}

    for object_id in object_ids:
        pose =  p.getBasePositionAndOrientation(
                    object_id, physicsClientId=task.client_id
                )
        meta_info = env.meta_info['obj_id_to_info']
        object_desc_key = meta_info[object_id]['texture_name']+ " object"
        object_id2pose[object_desc_key] = pose
    return object_id2pose

def get_stack_with_object_prompts(env, task):
    object_ids = env.obj_ids['fixed']+env.obj_ids['rigid']
    prompt = "stack the "
    for object_id in object_ids:
        pose =  p.getBasePositionAndOrientation(
                    object_id, physicsClientId=task.client_id
                )
        info = env.env.meta_info['obj_id_to_info'][object_id]
        # print(info)
        prompt+=info['texture_name']+info['obj_name'] + " and "
        
        print(f"Object Id: {object_id}, \nObject Pos: {pose}, \nObject Info:\n {info}")

    prompt = prompt[:-5] + " objects together."
    return prompt

def get_pick_place_prompts(env, task):
    object_ids = env.obj_ids['fixed']+env.obj_ids['rigid']
    prompt = " "
    for object_id in object_ids:
        pose =  p.getBasePositionAndOrientation(
                    object_id, physicsClientId=task.client_id
                )
        info = env.env.meta_info['obj_id_to_info'][object_id]
        # print(info)
        prompt+=info['texture_name']+info['obj_name'] + " and "
        
        print(f"Object Id: {object_id}, \nObject Pos: {pose}, \nObject Info:\n {info}")

    prompt = prompt[:-5] + " objects together."
    return prompt

def get_action_steps(action_outputs):
    lines = action_outputs.split('\n')
    # if line starts with a number, it is an action step. a number and a dot is an action step.
    action_steps = [line for line in lines if re.match(r'^\d+', line)]
    return action_steps

def move_to_next_action_steps(action_outputs):
    return ' '.join(get_action_steps(action_outputs)[2:])

def get_pick_place_object(action_outputs):
    steps = get_action_steps(action_outputs)
    first_step = steps[0]
    second_step = steps[1]
    pick_object=None
    place_object=None
    if "pick up" in first_step:
        match = re.search(r'.*\b(up)\b(.*)$', first_step)
        pick_object = match.group(2).strip()
    if "place" in second_step:
        match = re.search(r'.*\b(up|in|on)\b(.*)$', second_step)
        place_object = match.group(2).strip()

    assert pick_object !=None and place_object!=None
    return pick_object, place_object

def get_pick_place_coordinates(pick_object, place_object):
    """
    pick up *[green container]* 
    place *[green container]*  on the #[black ball]#
    """

    return {'pose0_position': np.array(object_id2coords[pick_object][0][:2], dtype=np.float32),
    'pose0_rotation': np.array(object_id2coords[pick_object][1], dtype=np.float32),
    'pose1_position': np.array(object_id2coords[place_object][0][:2], dtype=np.float32),
    'pose1_rotation': np.array(object_id2coords[place_object][1], dtype=np.float32)}

    
object_id2coords=None

@hydra.main(config_path=".", config_name="conf")
def main(cfg):

    print(cfg)
    
    kwargs = cfg.vima_bench_kwargs
    seed = kwargs["seed"]
    # print("before vima env \n\n")
    env = vima_bench.make(**kwargs)
    # print("after vima env \n\n")
    task = env.task

    # task.set_difficulty("hard")
    oracle_fn = task.oracle(env)

    inference_model = hydra.utils.instantiate(cfg.inference_model)
    verifier_strategy = inference_model.verifier_strategy
    plan_strategy = inference_model.plan_strategy
    vima_perturb = VIMAPerturb()


    correct=0
    count=0
    incorrect_planner=0
    max_perturb_steps=cfg.max_perturb_steps
    state_1_image_path = cfg.state_1_image_path
    state_2_image_path = cfg.state_2_image_path

    # LLM PLAN
    # 1. black
    # 2. 
    # 3.
    # 4.

    # fail step 2 image after failure
    # LLM replan
    # 1. 
    # 2. 
    # 3. 

    total_seeds = 30
    num_ids = 0
    with tqdm(total=total_seeds) as pbar:
        while num_ids < total_seeds:

            verifier_output="failure"
            plan_step=2
            env.seed(seed+2)

            obs = env.reset()
            # obs = env.reset()
            # obs = env.reset()
            global object_id2coords
            object_id2coords = get_object_to_pose(env, task)
            # print("hereeee\n\n")
            # env.render()
            # object_ids = env.obj_ids['fixed']+env.obj_ids['rigid']
            prompt, prompt_assets = env.get_prompt_and_assets()
            print(f"TASK PROMPT :: {prompt}\n")

            print(f"OBJECTS :: {object_id2coords.keys()}")
            possible_objects = list(object_id2coords.keys())
            if len(set(possible_objects))<3: # if there are duplicate objects, skip this seed
                seed += 1
                continue
            if len([x for x in possible_objects if "rainbow" in x])>0:
                seed += 1
                continue
            
            obs_rgb_image = env.env.render_camera(task.oracle_cams[0])[0]
            # color
            # print(obs_rgb_image.shape)
            print(f"\nTaking screenshot of initial env state at:\n {state_1_image_path}\n")
            env.env.save_screenshot(task.oracle_cams[0], file_path=state_2_image_path)
            
            # objects = model_action.keys()
            
            object_prompt = "NOTE: Choose from the following objects to pick or place: " + ", ".join(object_id2coords.keys())
            prompt = prompt + "\n" + object_prompt

            # task.oracle_max_steps

            
            # state_2_image_path = state_1_image_path
            # +1 coz in that step it should be done.
            for curr_step in range(max_perturb_steps+1):
                # llm_action_steps = oracle_fn.act(obs)

                # get_object_ids()

                # prompt = get_stack_with_object_prompts(env, task)
                if verifier_output=="failure":
                    shutil.copy(state_2_image_path, state_1_image_path)
                    llm_action_steps = inference_model.infer([state_1_image_path, prompt])
                    print(f"LLM PLAN O/P ACTION STEPS :: {llm_action_steps}\n")
                    plan_step = 2

                if len(get_action_steps(llm_action_steps))<2:
                    print("Planner Wrong - less than required steps for pick place\n")
                    print(llm_action_steps)
                    incorrect_planner+=1
                    break
                
                pick_object, place_object = get_pick_place_object(llm_action_steps)
                print(f"\nPICK OBJECT :: {pick_object} \nPLACE OBJECT :: {place_object}\n")
                try:
                    model_action = get_pick_place_coordinates(pick_object, place_object)
                except:
                    count-=1
                    print("error handled")
                    print(f"Pick obj: {pick_object}, Place obj: {place_object}")
                    print(f"object coords: {object_id2coords}")
                    print("ignoring this now")
                
                # print(output)
                # exit()
                # clamp action to valid range
                model_action = {
                    k: np.clip(v, env.action_space[k].low, env.action_space[k].high)
                    for k, v in model_action.items()
                }
                # objects = model_action.keys()
                # model_action=model_action['pose0_position'][::-1]
                if curr_step<max_perturb_steps:
                    # perturb placement of picked object.
                    model_action['pose1_position'] = vima_perturb.random_drop()
                    

                obs, reward, done, info = env.step(action=model_action)

                # update picked object position to placement position.
                object_id2coords[pick_object] = (model_action['pose1_position'], model_action['pose1_rotation'])

                # curr_task_plans = "\n".join(llm_action_steps.split("\n")[:2])

                print(f"\nTaking screenshot of perturbed env state at:\n {state_2_image_path}\n")
                env.env.save_screenshot(task.oracle_cams[0], file_path=state_2_image_path)
                
                verifier_output = verifier_strategy.verify(inference_model, state_1_image_path, state_2_image_path, llm_action_steps, current_step=plan_step).lower().strip()
                print(f"VERIFIER O/P :: {verifier_output}\n")
                verifier_output = verifier_output.lower().strip()
                if ('success' in verifier_output) and ('failure' not in verifier_output):
                    verifier_output = "success"
                else:
                    verifier_output = "failure"
                print(f"FINAL VERIFIER DECISION :: {verifier_output}\n")
                # if curr_step<max_perturb_steps:
                    
                # else: 
                #     verifier_output="success" if done else "failure"

                if verifier_output=="success":
                    plan_step+=2
                    print(f"VERIFIER SAYS SUCCESS\n")
                    if done:
                        print(f"VERIFIER IS RIGHT\n")
                        correct+=1
                        break
                    else:
                        print(f"VERIFIER IS WRONG\n")
                        break
                else:
                    if done:
                        print(f"VERIFIER IS WRONG\n")
                        break

                
            count+=1
            
                # if done:
                #     break

            seed += 1
            # break
            num_ids += 1
            pbar.update(1)
        
    print(f"Task Success Rate: {correct/count}")
    print(f"Plan Failure Rate: {incorrect_planner/count}")
    print(f"Verifier Failure Rate: {(count-correct+incorrect_planner)/count}")


if __name__ == "__main__":
    main()
