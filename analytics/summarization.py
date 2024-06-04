
import logging
import os
from abc import ABC, abstractmethod
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from analytics.utils import get_string_after_colon
from third_party_api_clients.openai.openai_api_client import OpenAIApiClient

SYSTEM_PROMPT = """You are a helpful tool to summarize user interactions in any domain. """

PROMPT = "The provided interactions are written by students to a assistant. Provide a concise summary of WHAT the student is requesting. Do not make mention of the students. Limit the summary to 10 words or fewer: {context}\nSummary: "


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAIApiClient().open_ai_client
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": PROMPT.format(context = context),
                    },
                ],
                max_tokens=max_tokens,
            )
            return get_string_after_colon(response.choices[0].message.content)

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAIApiClient().open_ai_client
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful tool to summarize instructions"},
                    {
                        "role": "user",
                        "content": PROMPT.format(context = context)
                    },
                ],
                max_tokens=max_tokens,
            )
            return get_string_after_colon(response.choices[0].message.content)


        except Exception as e:
            print(e)
            return e