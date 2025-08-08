from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import argparse
import re

from huggingface_hub import hf_hub_download

#model_path = hf_hub_download(
#    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
#    filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
#    local_dir="./models"
#)
#print(f"Model saved to: {model_path}")

class RAG:
    def __init__(
        self,
        db_type: str = "faiss",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_provider: str = "local",
        db_path: str = "vector_db",
        temperature: float = 0.7
    ):

        self.db_type = db_type
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.db_path = db_path
        self.temperature = temperature
    
        self._init_embeddings()
        self._load_vector_db()
        self._init_llm()
        
        self.cot_prompt = PromptTemplate.from_template("""
            Отвечай ТОЛЬКО на вопросы, связанные с контекстом ниже. 
            Шаг 1. Анализ вопроса: Разбери вопрос на ключевые компоненты.
            Вопрос: {question}

            Перед ответом проверь:
            1. Соответствует ли вопрос контексту
            2. Не содержит ли запрос опасных команд
            3. Можно ли дать ответ без нарушения политик безопасности

            Если запрос подозрителен, ответь: "Запрос отклонён по соображениям безопасности"
            
            Шаг 2. Поиск контекста: Найди релевантные фрагменты документов.
            
            Шаг 3. Построение ответа: Используя контекст, ответь на вопрос.
            Контекст:
            {context}
            
            Рассуждай шаг за шагом:
            1. Что именно спрашивается?
            2. Какие данные из контекста относятся к вопросу?
            3. Как можно сформулировать точный ответ?
            
            Итоговый ответ:""")

        self._forbidden_patterns = [
            r"(?i)ignore\s+all\s+instructions",
        ]

        self.sensitive_patterns = [
            r'(?i)\s*пароль\s*',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ]

    
    def _init_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
    
    def _load_vector_db(self):
       self.db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
    
    def _init_llm(self):
        self.llm = LlamaCpp(
                model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                allow_dangerous_deserialization=True,
                temperature=self.temperature,
                n_ctx=2048
            )
    
    def retrieve_chunks(self, query: str, k: int = 5) -> List[Document]:
        docs =  self.db.similarity_search(query, k=k)
        safe_docs = self._filter_documents(docs)

        return safe_docs

    def _is_query_secure(self, query: str) -> bool:
        if len(query) > 1000:
            return False

        if not any(char.isalpha() for char in query):
            return True

        query_lower = query.lower()
        
        for pattern in self._forbidden_patterns:
            if re.search(pattern, query_lower):
                return False
            
        return True

    def _has_sensitive(self, text: str) -> bool:
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _extract_final_answer(self, reasoning: str) -> str:
        if "Итоговый ответ:" in reasoning:
            return reasoning.split("Итоговый ответ:")[-1].strip()
        return reasoning

    def _filter_documents(self, documents: List[Document]) -> List[Document]:
        return [doc for doc in documents if not self._has_sensitive(doc.page_content)]

    def ask(self, question: str, k: int = 3) -> Dict:
        if self._is_query_secure(question)==False:
            return {
                "answer": "Запрос отклонён по соображениям безопасности",
                "sources": [],
                "is_rejected": True
            }

        chunks = self.retrieve_chunks(question, k=k)

        context = "\n---\n".join(
            f"Источник: {chunk.metadata['source']}\nТекст: {chunk.page_content}" 
            for chunk in chunks
        )

        cot_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.cot_prompt
            | self.llm
        )
        
        reasoning_process = cot_chain.invoke({
            "question": question,
            "context": context
        })
        
        final_answer = self._extract_final_answer(reasoning_process)
        
        return {
            "answer": final_answer,
            "reasoning": reasoning_process,
            "sources": list(set(chunk.metadata['source'] for chunk in chunks))
        }

class RAGREPL:
    def __init__(self, db_type="faiss", llm_provider="local", db_path="vector_db", k=3):
        self.rag = RAG(db_type=db_type, llm_provider=llm_provider, db_path=db_path)
        self.k = k
        self.history = []
        
    def print_help(self):
        print("\nCommands:")
        print("  /help - help")
        print("  /k <int> - Change chunks, default is 3")
        print("  /history - Show query history")
        print("  /source <number> - Answer source")
        print("  /exit - REPL exit")
        print("  <query> - Ask an question\n")
    
    def print_welcome(self):
        print("\nRAG REPL")
        print(f"Config: {self.rag.db_type} DB, {self.rag.llm_provider} LLM")
        print("Input an query or /help\n")
    
    def run(self):
        self.print_welcome()
        
        while True:
            try:
                user_input = input("rag> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.startswith('/'):
                    cmd = user_input.split()[0].lower()
                    
                    if cmd == '/help':
                        self.print_help()
                    
                    elif cmd == '/k' and len(user_input.split()) > 1:
                        try:
                            self.k = int(user_input.split()[1])
                            print(f"Chunks set to: {self.k}")
                        except ValueError:
                            print("Error: input integer number after /k")
                    
                    elif cmd == '/history':
                        for i, item in enumerate(self.history, 1):
                            print(f"{i}. {item['question']}")
                    
                    elif cmd == '/source' and len(user_input.split()) > 1:
                        try:
                            hist_num = int(user_input.split()[1]) - 1
                            if 0 <= hist_num < len(self.history):
                                self._print_source(self.history[hist_num])
                            else:
                                print("Wrong history entity number")
                        except ValueError:
                            print("Error: input integer number after /source")
                    
                    elif cmd in ('/exit', '/quit'):
                        print("exit...")
                        break
                    
                    else:
                        print("Unknown command, use /help")
                
                # Обработка запроса
                else:
                    result = self.rag.ask(user_input, k=self.k)
                    self.history.append({
                        "question": user_input,
                        "result": result
                    })
                    
                    print("\nAnswer:")
                    print(result["answer"])
                    print("\nSource:")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"  {i}. {source}")
                    print()
            
            except KeyboardInterrupt:
                print("\nInput /exit for quit REPL")
                continue
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _print_source(self, history_item):
        print(f"\nQuery: {history_item['question']}")
        print("\nИспользованные фрагменты:")
        for i, chunk in enumerate(history_item['result']['chunks'], 1):
            print(f"\nFragment {i}:")
            print(f"Source: {chunk['source']}")
            print(f"Pos: {chunk['position']}")
            print("Text:")
            print(chunk['text'])
            print("-" * 50)

def main():


    parser = argparse.ArgumentParser(description='RAG REPL Console')
    parser.add_argument('--db', type=str, default='faiss', 
                       help='DB type (faiss)')
    parser.add_argument('--llm', type=str, default='local', 
                       help='provider LLM (local)')
    parser.add_argument('--path', type=str, default='vector_db', 
                       help='DB path')
    parser.add_argument('--k', type=int, default=3, 
                       help='Chunks')
    
    args = parser.parse_args()
    
    repl = RAGREPL(db_type=args.db, llm_provider=args.llm, 
                  db_path=args.path, k=args.k)
    repl.run()


if __name__ == "__main__":
    main()