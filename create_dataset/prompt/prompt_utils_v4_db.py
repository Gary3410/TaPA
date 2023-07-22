'''
This file is used to generate the prompt for GPT-3.5, using LangChain database.
'''
import os
import numpy as np
import json
from langchain.document_loaders import TextLoader
# from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# samples = [
#         {
#             "context": f"""List of objects: [wine, bowl, remote control, TV, table, desk, chair]
#                 Instruction: Give me a drink""", 
#                 "response": f"""Necessary actions:
#                 Step 1. Grasp a bottle of wine (Target: wine)
#                 Step 2. Grasp a glass (Target: bowl)
#                 Step 3. Place the cup on the table (Target: glass, table)
#                 Step 4. Pour the wine into the glass (Target: wine, glass)
#                 Step 5. Grasp the glass with wine (Target: glass)
#                 Step 6. Move to the person and hand over it
#                 Step 7. Done"""
#         }, 
#         {
#             "context": f"""List of objects: [wine, bowl, remote control, TV, table, desk, chair]
#                 Instruction: Please turn on the TV""", 
#                 "response": f"""Necessary actions:
#                 Step 1. Grasp the remote control (Target: remote control)
#                 Step 2. Move closer to the TV (Target: TV)
#                 Step 3. Rotate the remote control to point at the TV (Target: remote control, TV)
#                 Step 4. Press the power button to turn on the remote control (Target: remote control)
#                 Step 5. Done"""
#         }
#     ]


class PromptManager(object):
    def __init__(self, alfred_db_path) -> None:
        # FIXME: match the format in parse_json_2_alpaca.py
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            # chunk_size=1000,
            chunk_overlap=0,
            length_function=len,
        )
        # this is the original code from langchain document, slow
        # with open(alfred_train_data_path) as f:
        #     raw = f.read()
        # self.documents = text_splitter.create_documents(raw)

        loader = TextLoader(alfred_db_path, encoding="utf-8")
        raw = loader.load()
        self.documents = text_splitter.split_documents(raw)

        self.embeddings = HuggingFaceEmbeddings()
        self.db = Chroma.from_documents(self.documents, self.embeddings)
        print("database initialized!")

    def get_fewshot_sample():
        '''copied from prompt/prompt_utils.py'''
        samples = [{"context": "List of objects: [wine, bowl, remote control, TV, table, desk, chair]",
                    "response": f"""Generate the instruction: Give me a drink
    ---
    Necessary actions:
    Step 1. Grasp a bottle of wine (Target: wine)
    Step 2. Grasp a glass (Target: bowl)
    Step 3. Place the cup on the table (Target: glass, table)
    Step 4. Pour the wine into the glass (Target: wine, glass)
    Step 5. Grasp the glass with wine (Target: glass)
    Step 6. Move to the person and hand over it
    Step 7. Done
    ===
    Generate the instruction: Please turn on the TV
    ---
    Necessary actions:
    Step 1. Grasp the remote control (Target: remote control)
    Step 2. Move closer to the TV (Target: TV)
    Step 3. Rotate the remote control to point at the TV (Target: remote control, TV)
    Step 4. Press the power button to turn on the remote control (Target: remote control)
    Step 5. Done
    ==="""}]
        return samples
    
    def get_prompt(example=None):
        '''copied from prompt/prompt_utils.py'''
        messages = [{"role": "system", "content": f"""You are an indoor service robot named Garybot and you are inside a room. what you see is provided with a list of objects that contains all the objects in the room you are in. 
    The location of the objects in the list you are guided in advance, without reasoning about the spatial relations of the objects. Execute all the instructions as you are located in the room.

    Design a conversation between you and the person you are serving in the room. The answer should be the tone of the service robot located in the room and performing the action specifically. 
    The generated instructions can be described in different tones. Ask for various instructions and give the corresponding series of actions with a maximum of 15 steps.

    Only include instructions for their corresponding actions only utilizing atomic motions (Grasp, Release, Lift, Place, Rotate, Push, Pull, Align, Press, Pour, Move): 
    (1) Generate operation instructions using only the objects in the list with the actions that must be performed to complete the operating instructions;
    (2) Do not generate any instructions or actions that cannot be executed with confidence;
    (3) Do not generate any instructions or actions with (Target: xx), xx is outside the list of objects.

    Again, the object being manipulated cannot be located outside the list. 
    Please double-check that Target: xx is in the list at each step and that xx is in the list.
    When evaluating the existence of xx, consider its original part or component, its function, and whether it can be replaced by an object in the list,
    and if it is satisfied, you can iterate over each element in the list to find an alternative and replace xx."""}]
        return messages

    def get_fewshot_sample_db(self, query, k)->list:
        '''
        Use the database to find the most similar samples
        '''
        samples = []
        docs = self.db.similarity_search(query, k=k)
        for doc in docs:
            # load json from content
            sample = doc.page_content.split("\n\n")[0]  # for json.loads
            # print(f"sample: {sample}")
            json_data = json.loads(sample)
            sample = {
                "context": f"""List of objects: {json_data["input"]}""",
                "response": f"""Generate the instruction: {json_data["instruction"]}
    ---
    Necessary actions:
    {json_data["output"]}
    ==="""
            }
            samples.append(sample)
        return samples

    def get_prompt_as_ai2thor(self, example=None):
        '''
        Give executable actions
        '''
        messages = [{"role": "system", "content": f"""You are an indoor service robot named Garybot and you are inside a room. what you see is provided with a list of objects that contains all the objects in the room you are in. 
    The location of the objects in the list you are guided in advance, without reasoning about the spatial relations of the objects. Execute all the instructions as you are located in the room.

    Design a conversation between you and the person you are serving in the room. The answer should be the tone of the service robot located in the room and performing the action specifically. 
    The generated instructions can be described in different tones. Ask for various instructions and give the corresponding series of actions with a maximum of 15 steps.

    Only include instructions for their corresponding actions only utilizing these atomic motions: 
                     
    GotoLocation(object)
    Go to the subgoal object. This action is finished once the object is visible and reachable.
    Augments:
    - object: a string, the object to go to.

    PickupObject(object)
    Pick up an object.
    Augments:
    - object: a string, the object to pick.

    PutObject(object)
    Put down the holding object to an receptacle.
    Augments:
    - object: a string, the receptacle to put the object.

    OpenObject(receptacle)
    Open an openable receptacle.
    Augments:
    - receptacle: a string, the receptacle to open.

    CloseObject(receptacle)
    Close an openable receptacle.
    Augments:
    - receptacle: a string, the receptacle to close.

    ToggleObject(object)
    Toggle a toggleable object.
    Augments:
    - object: a string, the receptacle to toggle.

    SliceObject(object)
    Slice a sliceable object.
    Augments:
    - object: a string, the object to slice.

    CleanObject(object)
    Wash an object in the sink then pick it up.
    Augments:
    - object: a string, the object to clean.

    HeatObject(object)
    heat an object in the microwave then pick it up.
    Augments:
    - object: a string, the object to heat.

    CoolObject(object)
    Cool an object in the fridge then pick it up.
    Augments:
    - object: a string, the object to cool.

    (1) Generate operation instructions using only the objects in the list with the actions that must be performed to complete the operating instructions;
    (2) Do not generate any instructions or actions that cannot be executed with confidence;
    (3) Do not generate any instructions or actions with (Target: xx), xx is outside the list of objects.

    Again, the object being manipulated cannot be located outside the list. 
    Please double-check that Target: xx is in the list at each step and that xx is in the list.
    When evaluating the existence of xx, consider its original part or component, its function, and whether it can be replaced by an object in the list,
    and if it is satisfied, you can iterate over each element in the list to find an alternative and replace xx."""}]
        return messages
