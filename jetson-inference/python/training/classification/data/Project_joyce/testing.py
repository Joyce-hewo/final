from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="phi")
response = llm.invoke("What's the weather like on Mars?")
