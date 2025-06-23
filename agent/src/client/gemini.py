import os
import requests
from typing import List, Dict, Optional
from src.genner.Base import Genner
from result import Ok, Err, Result
from src.types import ChatHistory
import re
import yaml

class GeminiDirectError(Exception):
    """Base exception class for Gemini Direct errors"""
    pass

class GeminiDirect(Genner):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash", timeout: int = 60):
        """
        Cliente directo para la API de Gemini (Google AI Studio).
        Args:
            api_key (str, optional): API key de Google Gemini. Si no se pasa, se lee de GOOGLE_API_KEY en el entorno.
            model (str): Nombre del modelo a usar (por defecto gemini-2.5-flash).
            timeout (int): Timeout para requests.
        """
        super().__init__("gemini-direct", False)
        self.api_key = api_key or os.environ["GOOGLE_API_KEY"]
        self.model = model
        self.timeout = timeout
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.headers = {"Content-Type": "application/json"}

    def ch_completion(self, messages: ChatHistory) -> Result[str, str]:
        try:
            prompt = "\n".join([msg.content for msg in messages.messages])
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
            }
            params = {"key": self.api_key}
            resp = requests.post(self.base_url, headers=self.headers, params=params, json=data, timeout=self.timeout)
            if resp.status_code != 200:
                return Err(f"HTTP error {resp.status_code}: {resp.text}")
            result = resp.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            return Ok(text)
        except Exception as e:
            return Err(f"Error in GeminiDirect: {str(e)}")

    def generate_code(self, messages: ChatHistory, blocks: List[str] = [""]) -> Result[tuple, str]:
        raw_response = ""
        try:
            completion_result = self.ch_completion(messages)
            if err := completion_result.err():
                return (Ok((None, raw_response)) if raw_response else Err(f"GeminiDirect.generate_code: completion_result.is_err(): \n{err}"))
            raw_response = completion_result.unwrap()
            extract_code_result = self.extract_code(raw_response, blocks)
            if err := extract_code_result.err():
                return Ok((None, raw_response))
            processed_code = extract_code_result.unwrap()
            return Ok((processed_code, raw_response))
        except Exception as e:
            return (Ok((None, raw_response)) if raw_response else Err(f"GeminiDirect.generate_code: An unexpected error occurred: \n{e}"))

    def generate_list(self, messages: ChatHistory, blocks: List[str] = [""]) -> Result[tuple, str]:
        try:
            completion_result = self.ch_completion(messages)
            if err := completion_result.err():
                return Err(f"GeminiDirect.generate_list: completion_result.is_err(): \n{err}")
            raw_response = completion_result.unwrap()
            extract_list_result = self.extract_list(raw_response, blocks)
            if err := extract_list_result.err():
                return Err(f"GeminiDirect.generate_list: extract_list_result.is_err(): \n{err}")
            extracted_list = extract_list_result.unwrap()
            return Ok((extracted_list, raw_response))
        except Exception as e:
            return Err(f"GeminiDirect.generate_list: An unexpected error occurred: \n{e}")

    @staticmethod
    def extract_code(response: str, blocks: List[str] = [""]) -> Result[List[str], str]:
        extracts: List[str] = []
        for block in blocks:
            try:
                regex_pattern = r"```python\n([\s\S]*?)```"
                code_match = re.search(regex_pattern, response, re.DOTALL)
                if not code_match:
                    continue
                code = code_match.group(1)
                extracts.append(code)
            except Exception as e:
                return Err(f"GeminiDirect.extract_code: Error extracting code: {e}")
        return Ok(extracts)

    @staticmethod
    def extract_list(response: str, blocks: List[str] = [""]) -> Result[List[List[str]], str]:
        extracts: List[List[str]] = []
        for block in blocks:
            try:
                regex_pattern = r"```yaml\n(.*?)```"
                yaml_match = re.search(regex_pattern, response, re.DOTALL)
                if not yaml_match:
                    continue
                yaml_content = yaml.safe_load(yaml_match.group(1).strip())
                if not isinstance(yaml_content, list):
                    continue
                extracts.append(yaml_content)
            except Exception as e:
                return Err(f"GeminiDirect.extract_list: Error extracting list: {e}")
        return Ok(extracts)
