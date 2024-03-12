from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition


all_command = [
    'use the stick to drag the cube onto the red target',
    'use the stick to drag the cube onto the blue target',
    'use the stick to drag the cube onto the yellow target',
    'use the stick to drag the cube onto the green target',
    # 'close the red jar',
    # 'close the brown jar',
    'push the green button',
    'push the red button',
    'push the blue button',
    'stack the red blocks',
    'stack the yellow blocks',
     
]
class ReachSmallObjects(Task):

    def init_task(self) -> None:
        pass

    def init_episode(self, index: int) -> List[str]:
        len_command = len(all_command)
        command = all_command[index % len_command]
        
        return [
            command,
            command,
            command,]

    def variation_count(self) -> int:
        return len(all_command)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]


    def is_static_workspace(self) -> bool:
        return True
