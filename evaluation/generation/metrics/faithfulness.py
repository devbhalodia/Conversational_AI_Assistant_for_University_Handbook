import json
import os
import re
import time
from typing import Optional
from pydantic import BaseModel

from google import genai
import instructor
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric


class CustomGeminiModel(DeepEvalBaseLLM):
    def __init__(self, model_name="models/gemini-3.1-flash-lite-preview"):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.instructor_client = instructor.from_genai(
            client=self.client,
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        )

    def load_model(self):
        return self.client

    def _clean_json(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    def generate(self, prompt: str, schema: Optional[BaseModel] = None):
        if schema is not None:
            return self.instructor_client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                model=self.model_name,
            )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return self._clean_json(response.text)

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini-3.1-Flash-Lite-Preview"


custom_model = CustomGeminiModel()

# Define metric — FaithfulnessMetric, not GEval
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=custom_model,
    include_reason=True,
)

# Load dataset
output_path = "evaluation/generation/generated_dataset.json"
with open(output_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Validate keys — contexts is required for faithfulness
required_keys = {"question", "answer", "contexts"}
for i, sample in enumerate(dataset):
    missing = required_keys - sample.keys()
    if missing:
        raise KeyError(f"Sample {i} is missing keys: {missing}")

# Build test cases
test_cases = [
    LLMTestCase(
        input=sample["question"],
        actual_output=sample["answer"],
        retrieval_context=sample["contexts"],  # must be a list of strings
    )
    for sample in dataset
]

# Evaluation loop
results = []
failed = []
MAX_RETRIES = 2

for i, test_case in enumerate(test_cases):
    print(f"Evaluating sample {i+1}/{len(test_cases)}...")
    success = False

    for attempt in range(MAX_RETRIES + 1):
        try:
            faithfulness_metric.measure(test_case)
            results.append({
                "index": i,
                "score": faithfulness_metric.score,
                "reason": faithfulness_metric.reason,
            })
            success = True
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES+1})...")
                time.sleep(wait)
            else:
                print(f"  Error on sample {i+1}: {e}")
                break

    if not success:
        failed.append({"index": i})

    time.sleep(20)

# Summary
avg_faithfulness = sum(r["score"] for r in results) / len(results) if results else 0
print(f"\nEvaluated: {len(results)}/{len(test_cases)} samples ({len(failed)} failed)")
print(f"Final Average Faithfulness: {avg_faithfulness:.2f}")