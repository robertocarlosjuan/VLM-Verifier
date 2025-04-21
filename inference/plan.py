class BasicPlan:
    def __init__(self):
        self.plan_template = """You are highly skilled in robotic task planning, breaking down intricate and long-term tasks into distinct primitive actions.
          And remember your last step plan needs to be "done". Consider the following skills a robotic arm can perform. In the descriptions below, think of [sth] as an object:
        1. pick up [sth]
        2. place [sth] in/on [sth]
        Remember do not use "the" in the middle of pick up / place and [sth].
        You are only allowed to use the provided skills. It's essential to stick to the format of these basic skills. When creating a plan, replace these placeholders with specific items or positions without using square brackets or parentheses. You can first itemize the task-related objects to help you plan.
        You must also decide if the goal is accomplished and output "success", if not, output "failure".
        {}
        """
        

    def infer(self, model, inputs):
        content = inputs[:-1] + [self.plan_template.format(inputs[-1])]
        response = model.run(content)
        response = response.split("Plan")[-1].replace("*","").replace(":","").strip()
        return response

class VimaPlan:
    def __init__(self):
        super().__init__()
        self.plan_template = """### Task Description
You are highly skilled in robotic task planning, breaking down intricate and long-term tasks into distinct primitive actions.
If the object is in sight, you need to directly manipulate it. If the object is not in sight, you need to use primitive skills to find the object first. If the target object is blocked by other objects, you need to remove all the blocking objects before picking up the target object. At the same time, you need to ignore distracters that are not related to the task. And remember your last step plan needs to be \"done\". Consider the following skills a robotic arm can perform. The skills can only performed on one object at a time. In the descriptions below, think of [sth] as an object:
1. pick up [sth]
2. place [sth] in/on [sth]
You are only allowed to use the provided skills. It's essential to stick to the format of these basic skills. When creating a plan, replace these placeholders with specific items or positions without using square brackets or parentheses. You can first itemize the task-related objects to help you plan.

### Example
**Task**: Put the ball on the cube
**Plan**:
1. pick up ball
2. place ball on cube

### New Task:
**Task**:{}
**Plan**:"""
    
    def infer(self, model, inputs):
        content = inputs[:-1] + [self.plan_template.format(inputs[-1])]
        response = model.run(content)
        response = response.split("Plan")[-1].replace("*","").replace(":","").strip()
        return response