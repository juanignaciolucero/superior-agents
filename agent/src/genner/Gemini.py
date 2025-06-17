import re
from typing import Callable, List, Tuple

import yaml
from result import Err, Ok, Result
from src.client.gemini import Gemini
from src.config import GeminiConfig
from src.helper import extract_content
from src.types import ChatHistory

from .Base import Genner


class GeminiGenner(Genner):
	def __init__(
		self,
		client: Gemini,
		config: GeminiConfig,
		stream_fn: Callable[[str], None] | None,
	):
		"""
		Initialize the Gemini-based generator.

		Args:
		    client (Gemini): Gemini API client
		    config (GeminiConfig): Configuration for the Gemini model
		    stream_fn (Callable[[str], None] | None): Function to call with streamed tokens,
		        or None to disable streaming
		"""
		super().__init__(f"gemini-{config.model}", True if stream_fn else False)
		self.client = client
		self.config = config
		self.stream_fn = stream_fn

	def ch_completion(self, messages: ChatHistory) -> Result[str, str]:
		"""
		Generate a completion using the Gemini API.

		Args:
		    messages (ChatHistory): Chat history containing the conversation context

		Returns:
		    Ok(str): The generated text if successful
		    Err(str): Error message if the API call fails
		"""
		final_response = ""

		try:
			# Ensure we have at least one non-empty message
			if not messages.messages or all(
				not msg.content for msg in messages.messages
			):
				return Err("No valid messages provided to Gemini")

			if self.do_stream:
				assert self.stream_fn is not None

				for token in self.client.create_chat_completion_stream(
					messages=messages.as_native(),
					max_tokens=self.config.max_tokens,
					temperature=self.config.temperature,
				):
					final_response += token
					self.stream_fn(token)

				self.stream_fn("\n")
			else:
				final_response = self.client.create_chat_completion(
					messages=messages.as_native(),
					max_tokens=self.config.max_tokens,
					temperature=self.config.temperature,
				)
			assert isinstance(final_response, str)
		except AssertionError as e:
			return Err(f"GeminiGenner.{self.config.model}.ch_completion error: \n{e}")
		except Exception as e:
			return Err(
				f"GeminiGenner.{self.config.model}.ch_completion: An unexpected error while generating code occurred: \n{e}"
			)

		return Ok(final_response)

	def generate_code(
		self, messages: ChatHistory, blocks: List[str] = [""]
	) -> Result[Tuple[List[str], str], str]:
		"""
		Generate code using the Gemini API.

		Args:
		    messages (ChatHistory): Chat history containing the conversation context
		    blocks (List[str]): XML tag names to extract content from before processing into code

		Returns:
		    Ok[processed_code, raw_response] | Err[error_message]
		"""
		raw_response = ""

		try:
			completion_result = self.ch_completion(messages)

			if err := completion_result.err():
				return (
					Ok((None, raw_response))
					if raw_response
					else Err(
						f"GeminiGenner.{self.config.name}.generate_code: completion_result.is_err(): \n{err}"
					)
				)

			raw_response = completion_result.unwrap()

			extract_code_result = self.extract_code(raw_response, blocks)

			if err := extract_code_result.err():
				return Ok((None, raw_response))

			processed_code = extract_code_result.unwrap()
			return Ok((processed_code, raw_response))

		except Exception as e:
			return (
				Ok((None, raw_response))
				if raw_response
				else Err(
					f"GeminiGenner.{self.config.name}.generate_code: An unexpected error occurred: \n{e}"
				)
			)

	def generate_list(
		self, messages: ChatHistory, blocks: List[str] = [""]
	) -> Result[Tuple[List[List[str]], str], str]:
		"""
		Generate lists using the Gemini API.

		Args:
		    messages (ChatHistory): Chat history containing the conversation context
		    blocks (List[str]): XML tag names to extract content from before processing into lists

		Returns:
		    Result[Tuple[List[List[str]], str], str]:
		        Ok(Tuple[List[List[str]], str]): Tuple containing:
		            - List[List[str]]: Processed lists of items
		            - str: Raw response from the model
		        Err(str): Error message if generation failed
		"""
		try:
			completion_result = self.ch_completion(messages)

			if err := completion_result.err():
				return Err(
					f"GeminiGenner.{self.config.name}.generate_list: completion_result.is_err(): \n{err}"
				)

			raw_response = completion_result.unwrap()

			extract_list_result = self.extract_list(raw_response, blocks)

			if err := extract_list_result.err():
				return Err(
					f"GeminiGenner.{self.config.name}.generate_list: extract_list_result.is_err(): \n{err}"
				)

			extracted_list = extract_list_result.unwrap()
		except Exception as e:
			return Err(
				f"GeminiGenner.{self.config.name}.generate_list: An unexpected error while generating list occurred: \n{e}"
			)

		return Ok((extracted_list, raw_response))

	@staticmethod
	def extract_code(response: str, blocks: List[str] = [""]) -> Result[List[str], str]:
		"""
		Extract code blocks from a Gemini model response.

		Args:
		    response (str): The raw response from the model
		    blocks (List[str]): XML tag names to extract content from before processing into code

		Returns:
		    Result[List[str], str]:
		        Ok(List[str]): List of extracted code blocks
		        Err(str): Error message if extraction failed
		"""
		extracts: List[str] = []

		for block in blocks:
			try:
				response = extract_content(response, block)
				regex_pattern = r"```python\n([\s\S]*?)```"
				code_match = re.search(regex_pattern, response, re.DOTALL)

				assert code_match is not None, "No code match found in the response"
				assert code_match.group(1) is not None, (
					"No code group number 1 found in the response"
				)

				code = code_match.group(1)
				assert isinstance(code, str), "Code is not a string"

				extracts.append(code)
			except AssertionError as e:
				return Err(f"GeminiGenner.extract_code: Regex failed: \n{e}")
			except Exception as e:
				return Err(
					f"GeminiGenner.extract_code: An unexpected error while extracting code occurred, error: \n{e}"
				)

		return Ok(extracts)

	@staticmethod
	def extract_list(
		response: str, blocks: List[str] = [""]
	) -> Result[List[List[str]], str]:
		"""
		Extract lists from a Gemini model response.

		Args:
		    response (str): The raw response from the model
		    blocks (List[str]): XML tag names to extract content from before processing into lists

		Returns:
		    Result[List[List[str]], str]:
		        Ok(List[List[str]]): List of extracted lists
		        Err(str): Error message if extraction failed
		"""
		extracts: List[List[str]] = []

		for block in blocks:
			try:
				response = extract_content(response, block)
				regex_pattern = r"```yaml\n(.*?)```"
				yaml_match = re.search(regex_pattern, response, re.DOTALL)

				assert yaml_match is not None, "No match found"
				yaml_content = yaml.safe_load(yaml_match.group(1).strip())
				assert isinstance(yaml_content, list), "Yaml content is not a list"
				assert all(isinstance(item, str) for item in yaml_content), (
					"All yaml content items must be strings"
				)

				extracts.append(yaml_content)
			except AssertionError as e:
				return Err(f"GeminiGenner.extract_list: Assertion error: \n{e}")
			except Exception as e:
				return Err(
					f"GeminiGenner.extract_list: An unexpected error while extracting list occurred, error: \n{e}"
				)

		return Ok(extracts)
