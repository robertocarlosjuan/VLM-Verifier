import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from inference.base import BaseInference

class QwenInference(BaseInference):

    def __init__(self, plan_strategy, verifier_strategy, model_path="Qwen/Qwen2-VL-7B-Instruct"):

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map=0, quantization_config=quantization_config)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.plan_strategy = plan_strategy
        self.verifier_strategy = verifier_strategy

    def run(self, inputs):

        content = []
        for inp in inputs:
            inp_type = "image" if os.path.exists(inp) else "text"
            content.append({"type": inp_type, inp_type: inp})
        
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def infer(self, *args, **kwargs):
        return self.plan_strategy.infer(self, *args, **kwargs)

    def verify(self, *args, **kwargs):
        return self.verifier_strategy.verify(self, *args, **kwargs)
