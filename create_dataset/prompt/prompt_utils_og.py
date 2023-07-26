import os
import numpy as np
import json


def get_prompt(example=None):
    messages = [{"role": "system", "content": f"""You are an indoor service robot to help me with everyday tasks by giving low-level action list. You are provided with a "bddl" and a "task".

You should only utilizing the following atomic motions: 

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

(1) Generate operation instructions using only the objects in the bddl with the actions that must be performed to complete the operating instructions;
(2) Do not generate any actions that cannot be executed with confidence;
"""}]
    return messages


def get_fewshot_sample():
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

