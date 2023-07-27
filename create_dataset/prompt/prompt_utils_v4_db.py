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

    def get_fewshot_sample(self):
        '''modified from prompt/prompt_utils.py'''
        samples = [{"context": "List of objects: [Wine, Bowl, RemoteControl, TV, Table, Desk, Chair, Fridge]",
                    "response": f"""dict(
    "instruction": "Give me a cold drink.",
    "thought": "The instruction is to give the user a cold drink. Luckily, there are Wine, Fridge and Table in the room. I'll find and pick up the Wine, find and cool it in the Fridge, then find and place it on the Table for you to enjoy.",
    "action_list":[
    dict("name": "GotoLocation", "arg": "Wine", "expectation": "I face the wine, holding nothing"),
    dict("name": "PickupObject", "arg": "Wine", "expectation": "I pick up a wine, holding a wine."),
    dict("name": "GotoLocation", "arg": "Fridge", "expectation": "I face the fridge, holding a wine."),
    dict("name": "CoolObject", "arg": "Wine", "expectation": "I put the wine into the fridge and pick it up, holding a cold wine."),
    dict("name": "GotoLocation", "arg": "Table", "expectation": "I face the table, holding a cold wine."),
    dict("name": "PutObject", "arg": "Table", "expectation": "I put down the cold wine on the table for you, holding nothing."),
    dict("name": "End", "arg": "", "expectation": "I finish the task."),
    ])"""},
    {"context": "List of objects: [Blinds, Vase, Plate, HousePlant, Toaster, Microwave, GarbageCan, Spoon, SaltShaker, Ladle, WineBottle,Safe, Knife, Lettuce, Fridge, Pencil, PaperTowelRoll, CreditCard, LightSwitch, Sink, Mug, Bread, ButterKnife, Pen, Pot, CellPhone, Potato, SideTable, CounterTop, Faucet, GarbageBag, Egg]",
    "response": f"""dict(
    "instruction": "Cook an egg and put it on a plate.",
    "thought": "The instruction is to cook an egg and put it on a plate. We have a Microwave, Egg and Plate available in the room. I'll pick up the Egg, go to the Microwave, heat it in Microwave, then find a Plate and place it on the Plate for you.",
    "action_list":[
    dict("name": "GotoLocation", "arg": "Egg", "expectation": "I face the egg, holding nothing."),
    dict("name": "PickupObject", "arg": "Egg", "expectation": "I pick up an egg, holding an egg."),
    dict("name": "GotoLocation", "arg": "Microwave", "expectation": "I face the microwave, holding an egg."),
    dict("name": "HeatObject", "arg": "Egg", "expectation": "I heat the egg in the microwave and pick it up to hold a hot egg."),
    dict("name": "GotoLocation", "arg": "Plate", "expectation": "I face the plate, holding a hot egg."),
    dict("name": "PutObject", "arg": "Plate", "expectation": "I put down the hot egg on the plate for you, holding nothing."),
    dict("name": "End", "arg": "", "expectation": "I finish the task."),
    ])"""},
    {"context": "List of objects: [GarbageCan, Sink,Safe, Potato, Chair, Vase, Egg, Book,Stool, Pen, Blinds, Tomato, SoapBottle, Apple, CoffeeMachine, Statue, ButterKnife, PepperShaker, Pencil, Mug, Plate, Spoon, Spatula, DishSponge, SideTable, Ladle, Window, Pan, WineBottle, Pot, Bread, Lettuce, GarbageBag, SaltShaker]",
    "response": f"""dict(
    "instruction": "Make a cup of coffee.",
    "thought": "The instruction is to make a cup of coffee. We have a CoffeeMachine, Mug and SideTable available in the room. I'll pick up the Mug, go to the CoffeeMachine, put down the Mug, toggle on the CoffeeMachine to fill it, pick it up, then go to the SideTable and put it down for you.",
    "action_list":[
    dict("name": "GotoLocation", "arg": "Mug", "expectation": "I face the mug, holding nothing."),
    dict("name": "PickupObject", "arg": "Mug", "expectation": "I pick up a mug, holding a mug."),
    dict("name": "GotoLocation", "arg": "CoffeeMachine", "expectation": "I face the coffee machine, holding a mug."),
    dict("name": "PutObject", "arg": "CoffeeMachine", "expectation": "I put the mug on the coffee machine, holding nothing."),
    dict("name": "ToggleObject", "arg": "CoffeeMachine", "expectation": "I toggle on the coffee machine to fill the mug, holding nothing."),
    dict("name": "PickupObject", "arg": "Mug", "expectation": "I pick up the filled mug, holding a filled mug."),
    dict("name": "GotoLocation", "arg": "SideTable", "expectation": "I face the dining table, holding a filled mug."),
    dict("name": "PutObject", "arg": "SideTable", "expectation": "I put down the filled mug on the table for you, holding nothing."),
    dict("name": "End", "arg": "", "expectation": "I finish the task."),
    ])"""},
    ]
        return samples
    
    def get_prompt(self, example=None):
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
        messages = [{"role": "system", "content": f"""You are an indoor service robot named Garybot and you are inside a room in AI2THOR simulator. What you see is provided with a list of objects that contains all the objects in the room. 
    You need to design a conversation between you and the user you are serving in the room, the user generates instructions, and you give corresponding action lists.

    1. Generate instructions using only the objects in the list.

    2. Only generate instructions or actions that can be executed in confidence.

    3. Please use the minimal objects and actions to complete the instructions.

    4. You should only utilize the following atomic actions: 

    GotoLocation(object)
    Go to an object or a receptacle. This action is finished once the target is visible and reachable (within 1.5m).
    Augments:
    - object: a string, the object to go to.

    PickupObject(object)
    Pick up an object. You can only hold one object at a time.
    Augments:
    - object: a string, the object to pick.

    PutObject(receptacle)
    Put down the held object to an receptacle. You need to pickup the object first.
    Augments:
    - receptacle: a string, the receptacle to put down the object.

    ToggleObject(object)
    Toggle a toggleable object.
    Augments:
    - object: a string, the receptacle to toggle.

    SliceObject(object)
    Slice a sliceable object with the held knife.
    Augments:
    - object: a string, the object to slice.

    CleanObject(object)
    Wash the held object in the sink then pick it up.
    Augments:
    - object: a string, the object to clean.

    HeatObject(object)
    Heat the held object in the microwave then pick it up.
    Augments:
    - object: a string, the object to heat.

    CoolObject(object)
    Cool the held object in the fridge then pick it up.
    Augments:
    - object: a string, the object to cool.

    End()
    You can use this action to end the episode.

    5. Your response should follow the json format:
    dict(
    "instruction": "The task description in natural language",
    "thought": "Your thoughts on the plan in natural language",
    "action_list":[
    dict("name": "action name", "arg": "arg name", "expectation": "describe the expected results of this action"),
    dict("name": "action name", "arg": "arg name", "expectation": "describe the expected results of this action"),
    ])
    Directly use the original name in "List of objects" in "arg". Ensure that your whole response can be parsed by Python json.loads, do not write any other information before or after the json format."""}]
        return messages
