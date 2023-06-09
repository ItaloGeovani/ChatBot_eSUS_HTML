import os
from flask import Flask, render_template, request, jsonify

import openai
import chromadb
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.prompts.prompt import PromptTemplate

os.environ["OPENAI_API_KEY"] = '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'

app = Flask(__name__)

def get_document():
    loader = UnstructuredWordDocumentLoader('Indicadores_Previne_Brasil.docx')
    data = loader.load()
    return data

my_data = get_document()

text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
my_doc = text_splitter.split_documents(my_data)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(my_doc, embeddings)

template = """Você é um assistente de IA para responder perguntas a partir de um documento, na língua Portuguesa pt-br.
Letras maiúsculas e minúsculas são iguais.
Use marcadores numerados conforme necessário.
Pergunta: {question}
=========
{context}
=========
Resposta em Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

def generate_response(query, chat_history):
    if query:
        llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
        my_qa = ChatVectorDBChain.from_llm(llm, vectordb, return_source_documents=True)
        result = my_qa({"question": query, "chat_history": chat_history})
    return result["answer"]

chat_history = []  # Variável global para armazenar o histórico de conversas

@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    global chat_history  # Para acessar a variável global
    chat_history, response = my_chatbot(text, chat_history)  # Chamada da função my_chatbot para lidar com o histórico
    message = {"answer": response}
    return jsonify(message)

def my_chatbot(input, history):
    history = history or []
    my_history = list(sum(history, ()))
    my_history.append(input)
    output = generate_response(input, history)
    history.append((input, output))
    return history, output

if __name__ == "__main__":
    app.run(debug=True)
