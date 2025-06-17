import os
import google.generativeai as genai
from typing import Optional, Dict, Generator, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str


class GeminiError(Exception):
    """Base exception class for Gemini errors"""
    pass


class Gemini:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "models/gemini-1.5-flash",
        timeout: int = 60,
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key: Your Google API key (if None, will try to get from GOOGLE_API_KEY env var)
            model: The model to use (e.g., "models/gemini-1.5-flash")
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise GeminiError("GOOGLE_API_KEY environment variable is not set")
            
        self.model_name = model
        self.timeout = timeout
        
        # Configure the Gemini API
        genai.configure(api_key=self.api_key)
        
        # Get available models
        for m in genai.list_models():
            if m.name == model:
                self.model = genai.GenerativeModel(model)
                break
        else:
            raise GeminiError(f"Model {model} not found. Available models: {[m.name for m in genai.list_models()]}")

    def create_chat_completion(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Create a non-streaming chat completion.

        Args:
            messages: List of message dictionaries or Message objects
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            The generated text response as a string
        """
        try:
            # Convert messages to Gemini format
            chat = self.model.start_chat(history=[])
            
            # Add messages to chat
            for msg in messages:
                if msg["role"] == "user":
                    if msg["content"]:  # Only send non-empty messages
                        chat.send_message(msg["content"])
                elif msg["role"] == "assistant":
                    # Gemini doesn't support adding assistant messages directly
                    # We'll skip these as they're part of the history
                    pass
                elif msg["role"] == "system":
                    # System messages are handled differently in Gemini
                    # We'll prepend it to the first user message
                    if len(chat.history) == 0 and msg["content"]:
                        chat.send_message(f"System: {msg['content']}")

            # Generate response
            response = chat.send_message(
                "Continue",  # Use a default message instead of empty string
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature or 0.7,
                    max_output_tokens=max_tokens,
                )
            )

            return response.text

        except Exception as e:
            raise GeminiError(f"Error in create_chat_completion: {str(e)}")

    def create_chat_completion_stream(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Create a streaming chat completion.

        Args:
            messages: List of message dictionaries or Message objects
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            Generator yielding response chunks
        """
        try:
            # Convert messages to Gemini format
            chat = self.model.start_chat(history=[])
            
            # Add messages to chat
            for msg in messages:
                if msg["role"] == "user":
                    if msg["content"]:  # Only send non-empty messages
                        chat.send_message(msg["content"])
                elif msg["role"] == "assistant":
                    # Gemini doesn't support adding assistant messages directly
                    pass
                elif msg["role"] == "system":
                    # System messages are handled differently in Gemini
                    if len(chat.history) == 0 and msg["content"]:
                        chat.send_message(f"System: {msg['content']}")

            # Generate streaming response
            response = chat.send_message(
                "Continue",  # Use a default message instead of empty string
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature or 0.7,
                    max_output_tokens=max_tokens,
                ),
                stream=True
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            raise GeminiError(f"Error in create_chat_completion_stream: {str(e)}") 