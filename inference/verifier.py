from inference.utils import combine_images_side_by_side

class CombinedImageVerification:

    def __init__(self):
        self.verifier_template = "You are highly skilled in robotic task verification. The image shows the initial observation on the left and the current observation on the right. Given the task plan\n{}\nIdentify if the step has been successfully executed. Provide your answer in one word 'success', 'partial success', 'partial failure' or 'failure'."

    def verify(self, model, initial_image_path, current_image_path, task_plan, current_step, combined_image_path = "combined_image.jpg"):
        curr_task_plans = "\n".join(task_plan.split("\n")[:current_step])
        prompt = self.verifier_template.format(curr_task_plans)
        combine_images_side_by_side(initial_image_path, current_image_path, 
        combined_image_path, border_width=20, border_color=(255, 0, 0))
        return model.run([combined_image_path, prompt])

class BinaryCombinedImageVerification:

    def __init__(self):
        self.verifier_template = "You are highly skilled in robotic task verification. The image shows the initial observation on the left and the current observation on the right. Given the task plan\n{}\nIdentify if the step has been successfully executed. Provide your answer in one word 'success', 'failure'."

    def verify(self, model, initial_image_path, current_image_path, task_plan, current_step, combined_image_path = "combined_image.jpg"):
        curr_task_plans = "\n".join(task_plan.split("\n")[:current_step])
        prompt = self.verifier_template.format(curr_task_plans)
        combine_images_side_by_side(initial_image_path, current_image_path, 
        combined_image_path, border_width=20, border_color=(255, 0, 0))
        return model.run([combined_image_path, prompt])

class TwoImageVerification:

    def __init__(self):
        self.verifier_template = "You are highly skilled in robotic task verification. The first image shows the initial observation and the second image shows current observation. Given the task plan\n{}\nIdentify if the step has been successfully executed. Provide your answer in one word 'success', 'partial success', 'partial failure' or 'failure'."

    def verify(self, model, initial_image_path, current_image_path, task_plan, current_step):
        curr_task_plans = "\n".join(task_plan.split("\n")[:current_step])
        prompt = self.verifier_template.format(curr_task_plans)
        content = [initial_image_path, current_image_path, prompt]
        return model.run(content)

class InterleavedImageVerification:

    def __init__(self):
        self.verifier_template = "You are highly skilled in robotic task verification. Given initial observation, current observation and the task plan\n{}\nIdentify if the step has been successfully executed. Provide your answer in one word 'success', 'partial success', 'partial failure' or 'failure'. explain why"

    def verify(self, model, initial_image_path, current_image_path, task_plan, current_step):
        curr_task_plans = "\n".join(task_plan.split("\n")[:current_step])
        prompt = self.verifier_template.format(curr_task_plans)
        content = [initial_image_path, "Initial observation", current_image_path, "Current observation", prompt]
        return model.run(content)

class TwoStepVerification:

    def __init__(self):
        self.expected_output_template = "You are highly skilled in robotic task planning. The image shows the initial state. Given the task plan\n{}\nThe robot has just completed step {}.\nDescribe the expected observation."
        self.verifier_template = "You are highly skilled in robotic task verification. Given the expected observation {}, identify if the current observation is expected. Provide your answer in one word 'success', 'partial success', 'partial failure' or 'failure'."

    def verify(self, model, initial_image_path, current_image_path, task_plan, current_step):
        expected_output_prompt = self.expected_output_template.format(task_plan, current_step)
        expected_output = model.run([initial_image_path, expected_output_prompt])
        verifier_prompt = self.verifier_template.format(expected_output)
        verifier_result = model.run([current_image_path, verifier_prompt])
        return verifier_result

class TwoStepVerificationEdited:

    def __init__(self):
        self.expected_output_template = "You are highly skilled in robotic task planning. The image shows the initial state. Given the task plan\n{}\nThe robot has just completed step {}.\nAssume the camera perspective is fixed. Give a description of the expected visual observation that would confirm the successful completion of this step. When the object description can refer to multiple objects equally, list the descriptions of the various possible expected observations. Do not reference prior object positions or actions."
        # "You are highly skilled in robotic task verification. Given the expected observation {}, identify if the current observation is expected. Provide your answer in one word 'success' or 'failure'."
        self.verifier_template = "You are an expert at verifying whether the spatial arrangement of objects in an image matches a given scene description. You are provided with an image and descriptions of object positions in a scene:\n{}\nExamine the image and determine if it matches any of the descriptions. As long as one description matches the scene, it should be considered a success. Focus only on the spatial relationships between objects involved. Provide your answer in one word 'success' or 'failure'."

    def verify(self, model, initial_image_path, current_image_path, task_plan, current_step):
        expected_output_prompt = self.expected_output_template.format(task_plan, current_step)
        expected_output = model.run([initial_image_path, expected_output_prompt])
        print(f"EXPECTED output prompt :: {expected_output_prompt}")
        print("expected_output\n", expected_output)
        verifier_prompt = self.verifier_template.format(expected_output)
        verifier_result = model.run([current_image_path, verifier_prompt])
        print(f"VERIFIER PROMPT :: {verifier_prompt}")
        print(f"VERIFIER O/P :: {verifier_result}\n")
        return verifier_result

class BinaryTwoStepVerification:

    def __init__(self):
        self.expected_output_template = "You are highly skilled in robotic task planning. The image shows the initial state. Given the task plan\n{}\nThe robot has just completed step {}.\nDescribe the expected observation."
        self.verifier_template = "You are highly skilled in robotic task verification. Given the expected observation {}, identify if the current observation is expected. Provide your answer in one word 'success' or 'failure'."

    def verify(self, model, initial_image_path, current_image_path, task_plan, current_step):
        expected_output_prompt = self.expected_output_template.format(task_plan, current_step)
        expected_output = model.run([initial_image_path, expected_output_prompt])
        print("expected_output\n", expected_output)
        verifier_prompt = self.verifier_template.format(expected_output)
        verifier_result = model.run([current_image_path, verifier_prompt])
        return verifier_result

class VideoVerification:

    def __init__(self):
        self.verifier_template = "You are tasked with verifying the execution of a robot's task plan based on the provided video observation. Given the current video footage and the detailed task plan:\n{}\nIdentify the status of step {}. Determine if the step is:\nIn progress: The robot is actively working on the task but hasn't completed it yet.\nSuccess: The robot has successfully completed the task step as intended.\nFailure: The robot has failed to execute the task step properly.\nProvide your answer as either 'in progress', 'success', or 'failure'."

    def verify(self, model, video_path, task_plan, current_step):
        prompt = self.verifier_template.format(task_plan, current_step)
        content = [video_path, prompt]
        return model.run(content)