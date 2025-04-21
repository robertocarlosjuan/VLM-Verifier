import os
import time
import google.generativeai as genai
from google.generativeai import types
from inference.base import BaseInference

class GeminiInference(BaseInference):

    def __init__(self, plan_strategy, verifier_strategy, model_path="gemini-1.5-flash"):
        self.model = genai.GenerativeModel(
            model_name=model_path,
            generation_config = types.GenerationConfig(
                temperature=0
            )
        )
        self.plan_strategy = plan_strategy
        self.verifier_strategy = verifier_strategy
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def process_image(self, image_path):
        image_file = genai.upload_file(path=image_path)
        return genai.get_file(image_file.name)

    def run(self, content):
        content = [self.process_image(x) if x.endswith('png') else x for x in content]
        response = self.model.generate_content(
            content, 
            request_options={"timeout": 10000}
        )
        time.sleep(3)
        return response.text

    def infer(self, *args, **kwargs):
        response = self.plan_strategy.infer(self, *args, **kwargs)
        return response

    def verify(self, *args, **kwargs):
        return self.verifier_strategy.verify(self, *args, **kwargs)
