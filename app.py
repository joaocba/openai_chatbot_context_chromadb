##
# Uses LangChain and ChromaDb as vector store
# Support multiple files to be vectorized (folder)

## Dependecies:
# pip install flask
# pip install langchain
# nltk.download('averaged_perceptron_tagger')
# pip install unstructured
# pip install python-magic-bin
# pip install chromadb


from flask import Flask, render_template, request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import magic
import nltk
import os
import logging
import datetime


# create logs folder if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# get current date and time for log filename
log_filename = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# configure logger
logging.basicConfig(filename=log_filename, level=logging.INFO)


# set OpenAI API
# os.environ["OPENAI_API_KEY"] = ''
# in case it is already defined on windows path variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


# set directory
loader = DirectoryLoader('./content/empresas', glob='**/*.txt')
documents = loader.load()

# settings for text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])

# search texts on docs with embeddings
docsearch = Chroma.from_documents(texts, embeddings)


# array to store conversations
conversation = ["You are a virtual assistant and you speak portuguese."]    # define initial role

app = Flask(__name__)

# define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():

    # index data
    qa = VectorDBQA.from_chain_type(llm=OpenAI(max_tokens = 150), chain_type="stuff", vectorstore=docsearch)

    # get user input
    user_input = request.args.get("msg") + '\n'
    response = ''
    if user_input:
        conversation.append(f"{user_input}")

        # get conversation history
        prompt = "\n".join(conversation[-3:])

        # generate AI response based on indexed data
        response = qa(prompt)
        print(response)

        # add AI response to conversation
        conversation.append(f"{response}")

        # log conversation
        with open(log_filename, "a") as f:
            f.write(f"User: {user_input}\n")
            f.write(f"AI: {response}\n\n")

        # log conversation using logger
        logging.info(f"User: {user_input}")
        logging.info(f"AI: {response}")

    return response['result'] if response else "Sorry, I didn't understand that."


if __name__ == "__main__":
    app.run()