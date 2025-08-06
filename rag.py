from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.llms import LlamaCpp
#from llama_cpp import Llama
import argparse

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
        
        # Инициализация эмбеддингов
        self._init_embeddings()
        
        # Загрузка векторной БД
        self._load_vector_db()
        
        # Инициализация LLM
        self._init_llm()
        
        # Шаблон промпта
        self.prompt_template = """Ответь на вопрос, используя только предоставленный контекст. 
Если ответа нет в контексте, скажи 'Не могу найти ответ в предоставленных документах'.

Контекст:
{context}

Вопрос: {question}
Ответ:"""
    
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
        """Поиск релевантных чанков
        
        Args:
            query: Текстовый запрос
            k: Количество возвращаемых чанков
            
        Returns:
            Список найденных документов с метаданными
        """
        return self.db.similarity_search(query, k=k)
    
    def format_prompt(self, question: str, chunks: List[Document]) -> str:
        """Формирование промпта с контекстом
        
        Args:
            question: Вопрос пользователя
            chunks: Найденные релевантные чанки
            
        Returns:
            Сформированный промпт
        """
        context = "\n\n".join([
            f"Источник: {chunk.metadata['source']}\n"
            f"Позиция: {chunk.metadata.get('start_index', 'N/A')}\n"
            f"Текст: {chunk.page_content}"
            for chunk in chunks
        ])
        
        return self.prompt_template.format(context=context, question=question)
    
    def generate_response(self, prompt: str) -> str:
        """Генерация ответа с помощью LLM
        
        Args:
            prompt: Сформированный промпт
            
        Returns:
            Ответ модели
        """
        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(self.prompt_template)
            | self.llm
        )
        
        return rag_chain.invoke({"context": prompt, "question": prompt})
    
    def ask(self, question: str, k: int = 3) -> Dict:
        """Полный цикл: поиск + генерация ответа
        
        Args:
            question: Вопрос пользователя
            k: Количество используемых чанков
            
        Returns:
            Словарь с ответом и метаданными
        """
        # Поиск релевантных чанков
        chunks = self.retrieve_chunks(question, k=k)
        
        # Формирование промпта
        prompt = self.format_prompt(question, chunks)
        
        # Генерация ответа
        response = self.generate_response(prompt)
        print(response)
        
        # Формирование результата
        return {
            "answer": response,
            "sources": [chunk.metadata["source"] for chunk in chunks],
            "chunks": [
                {
                    "text": chunk.page_content,
                    "source": chunk.metadata["source"],
                    "position": chunk.metadata.get("start_index", "N/A")
                }
                for chunk in chunks
            ]
        }

class RAGREPL:
    def __init__(self, db_type="faiss", llm_provider="local", db_path="vector_db", k=3):
        self.rag = RAG(db_type=db_type, llm_provider=llm_provider, db_path=db_path)
        self.k = k
        self.history = []
        
    def print_help(self):
        print("\nДоступные команды:")
        print("  /help - Показать это сообщение")
        print("  /k <число> - Изменить количество чанков (по умолчанию 3)")
        print("  /history - Показать историю запросов")
        print("  /source <номер> - Показать исходный текст для ответа")
        print("  /exit - Выйти из REPL")
        print("  <ваш запрос> - Задать вопрос системе\n")
    
    def print_welcome(self):
        print("\nДобро пожаловать в RAG REPL!")
        print(f"Конфигурация: {self.rag.db_type} DB, {self.rag.llm_provider} LLM")
        print("Введите ваш запрос или /help для списка команд\n")
    
    def run(self):
        self.print_welcome()
        
        while True:
            try:
                user_input = input("rag> ").strip()
                
                if not user_input:
                    continue
                    
                # Обработка команд
                if user_input.startswith('/'):
                    cmd = user_input.split()[0].lower()
                    
                    if cmd == '/help':
                        self.print_help()
                    
                    elif cmd == '/k' and len(user_input.split()) > 1:
                        try:
                            self.k = int(user_input.split()[1])
                            print(f"Установлено количество чанков: {self.k}")
                        except ValueError:
                            print("Ошибка: укажите целое число после /k")
                    
                    elif cmd == '/history':
                        for i, item in enumerate(self.history, 1):
                            print(f"{i}. {item['question']}")
                    
                    elif cmd == '/source' and len(user_input.split()) > 1:
                        try:
                            hist_num = int(user_input.split()[1]) - 1
                            if 0 <= hist_num < len(self.history):
                                self._print_source(self.history[hist_num])
                            else:
                                print("Неверный номер в истории")
                        except ValueError:
                            print("Ошибка: укажите номер после /source")
                    
                    elif cmd in ('/exit', '/quit'):
                        print("Выход...")
                        break
                    
                    else:
                        print("Неизвестная команда. Введите /help для списка команд")
                
                # Обработка запроса
                else:
                    result = self.rag.ask(user_input, k=self.k)
                    self.history.append({
                        "question": user_input,
                        "result": result
                    })
                    
                    print("\nОтвет:")
                    print(result["answer"])
                    print("\nИсточники:")
                    for i, source in enumerate(result["sources"], 1):
                        print(f"  {i}. {source}")
                    print()
            
            except KeyboardInterrupt:
                print("\nДля выхода введите /exit")
                continue
            except Exception as e:
                print(f"Ошибка: {str(e)}")
    
    def _print_source(self, history_item):
        print(f"\nЗапрос: {history_item['question']}")
        print("\nИспользованные фрагменты:")
        for i, chunk in enumerate(history_item['result']['chunks'], 1):
            print(f"\nФрагмент {i}:")
            print(f"Источник: {chunk['source']}")
            print(f"Позиция: {chunk['position']}")
            print("Текст:")
            print(chunk['text'])
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='RAG REPL Console')
    parser.add_argument('--db', type=str, default='faiss', 
                       help='Тип векторной БД (faiss)')
    parser.add_argument('--llm', type=str, default='local', 
                       help='Провайдер LLM (local)')
    parser.add_argument('--path', type=str, default='vector_db', 
                       help='Путь к векторной БД')
    parser.add_argument('--k', type=int, default=3, 
                       help='Количество чанков для поиска')
    
    args = parser.parse_args()
    
    repl = RAGREPL(db_type=args.db, llm_provider=args.llm, 
                  db_path=args.path, k=args.k)
    repl.run()



# Пример использования
if __name__ == "__main__":
    main()