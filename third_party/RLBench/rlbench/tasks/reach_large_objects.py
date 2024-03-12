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
    'open the drawer',
    'sweep the dirt to the dustpan',
    'turn tap',
]
class ReachLargeObjects(Task):

    def init_task(self) -> None:
        pass
        # self._waypoints = [Dummy('waypoint0')]
        # self.waypoint0 = Dummy('waypoint0')
        # self.success_sensor = ProximitySensor('success')
        # self.register_success_conditions(
        #     [DetectedCondition(self.robot.arm.get_tip(), self.success_sensor)])

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
