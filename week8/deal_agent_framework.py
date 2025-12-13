import os
import sys
import logging
import json
from typing import List, Optional
from twilio.rest import Client
from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity
from sklearn.manifold import TSNE
import numpy as np
import traceback
import threading


# Colors for logging
BG_BLUE = '\033[44m'
WHITE = '\033[37m'
RESET = '\033[0m'

# Colors for plot
CATEGORIES = ['Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games']
COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'cyan']

def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    
    handler.setFormatter(formatter)

    print(f"in init_logging, before adding handler to root from instance #{DealAgentFramework.isntance_count}")
    traceback.print_stack()

    root.addHandler(handler)

class DealAgentFramework:
    isntance_count = 0
    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"
    _memory_lock = threading.Lock()

    def __init__(self):
        DealAgentFramework.isntance_count += 1
        print(f"Creating DealAgentFramework instance #{DealAgentFramework.isntance_count}")
        init_logging()
        load_dotenv()
        client = chromadb.PersistentClient(path=self.DB)
        # memory is the list of Opportunity instances backed up by the memory.json file
        self.memory = self.read_memory()
        # retrieve the ChromaDB collection for RAG
        self.collection = client.get_or_create_collection('products')
        self.planner = None
        

    def init_agents_as_needed(self):
        if self.planner is None:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")
        
    def read_memory(self) -> List[Opportunity]:
        with DealAgentFramework._memory_lock:
            print(f"In read_memory, DealAgentFramework._memory_lock acquired")
            if os.path.exists(self.MEMORY_FILENAME):
                with open(self.MEMORY_FILENAME, "r") as file:
                    data = json.load(file)
                opportunities = [Opportunity(**item) for item in data]
                print(f"In read_memory, DealAgentFramework._memory_lock return1")
                return opportunities
            print(f"In read_memory, DealAgentFramework._memory_lock return2")
            return []

    def write_memory(self) -> None:
        with DealAgentFramework._memory_lock:
            print(f"In write_memory, DealAgentFramework._memory_lock acquired")
            data = [opportunity.dict() for opportunity in self.memory]
            with open(self.MEMORY_FILENAME, "w") as file:
                json.dump(data, file, indent=2)
            print(f"In write_memory, DealAgentFramework._memory_lock context end")

    def log(self, message: str):
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def get_memory(self) -> List[Opportunity]:
        with DealAgentFramework._memory_lock:
            print(f"In get_memory, DealAgentFramework._memory_lock acquired and return")
            return self.memory.copy()

    # ENTRY POINT for the class DealAgentFramework

    def run(self) -> List[Opportunity]:
        '''
        Docstring for run
        :param self: Description
        :return: Description
        :rtype: List[Opportunity]
        updates the memory with newly found opportunities
        writes the updated memory to the memory.json file
        Returns the updated memory with newly found opportunities
        '''
        self.init_agents_as_needed()
        logging.info("Kicking off Planning Agent")
        # passing the List of already fully processessed Opportunities,
        # that has been pessisted in the memory.json file
        # the call to plan returns a single fully processed Opportunity
        result = self.planner.plan(memory=self.memory)
        logging.info(f"Planning Agent has completed and returned: {result}")
        if result:
            with DealAgentFramework._memory_lock:
                print(f"In run, DealAgentFramework._memory_lock acquired")
                self.memory.append(result)
                print(f"In run, DealAgentFramework._memory_lock acquired, context end")
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=10000):
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection('products')
        result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=max_datapoints)
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        categories = [metadata['category'] for metadata in result['metadatas']]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__=="__main__":
    DealAgentFramework().run()
    