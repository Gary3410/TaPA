from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
controller = Controller(scene="FloorPlan10", platform=CloudRendering)
event = controller.step(action="RotateRight")
metadata = event.metadata
print(event, event.metadata.keys())