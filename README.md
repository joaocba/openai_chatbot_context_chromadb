## Chatbot AI Assistant (Context Knownledge Base) + Vector Database (ChromaDB)

#### Technology: OpenAi, LangChain, ChromaDB
#### Method: Completions (davinci model)

#### Description:
Chatbot developed with Python and Flask that features conversation with a virtual assistant. This uses a context based conversation and the answers are focused on a local file with knownledge, it uses OpenAi Embeddings and ChromaDB (open-source database) as a vector store to host and rapidly return the embedded data (memory only). It allows to define an initial role and personification.

OpenAI embeddings are numerical representations of words used to improve natural language processing.
LangChain is a framework built around LLMs. The core of the library is to “chain” together different components to create more advanced use cases around LLMs.

### How to run (commands Windows terminal with Python 2.7):

#### Part One: Prepare Environment
- **Define necessary parameters (OpenAi API key, ...) on file 'app.py'**
- Initialize virtual environment and install dependencies, run:

	    virtualenv env
	    env\Scripts\activate
	    pip install flask python-dotenv
        pip install openai
		pip install langchain
		pip install unstructured
		pip install python-magic-bin
		pip install chromadb

#### Part Two: Prepare local content
- Add documents to folder "empresas"

#### Part Three: Run the app
- Initialize the app:

	    flask run

- Enter "http://localhost:5000" on browser to interact with app

#### Changelog
- v0.1
	- initial build