from typing import List
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from pyrep.objects.dummy import Dummy
from rlbench.backend.conditions import DetectedCondition
from rlbench.const import colors


class PutRubbishInColorBin(Task):

    COLORS = [colors[0], colors[3]]

    def init_task(self) -> None:
        self.rubbish = Shape('rubbish')
        self.register_graspable_objects([self.rubbish])

        self.register_waypoint_ability_start(3, self._move_above_bin)
        self.register_waypoint_ability_start(4, self._is_last)

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        success_sensor = ProximitySensor(f'success{self._variation_index+1}')
        self.register_success_conditions(
            [DetectedCondition(self.rubbish, success_sensor)])

        self.main_bin = Shape(f'bin{self._variation_index+1}')
        self.distractor_bin = Shape(f'bin{1 if self._variation_index+1 == 2 else 2}')

        color_bulb = Shape('color_bulb')
        color_name, color_rgb = self.COLORS[self._variation_index]
        color_bulb.set_color(color_rgb)

        x1, y1, z1 = color_bulb.get_position()
        x2, y2, z2 = self.rubbish.get_position()
        x3, y3, z3 = self.main_bin.get_position()
        x4, y4, z4 = self.distractor_bin.get_position()
        pos = np.random.randint(2)
        if pos == 0:
            self.main_bin.set_position([x4, y4, z4])
            self.distractor_bin.set_position([x3, y3, z3])

        return ['put rubbish in the bin with the same color as the light',
                'drop the rubbish into the bin with the color of the light',
                'pick up the trash and leave it in the bin with the same color as the light']

    def variation_count(self) -> int:
        return 2

    def _move_above_bin(self, _):
        w3 = Dummy('waypoint3')
        w4 = Dummy('waypoint4')
        
        if self._variation_index == 1:
            w3.set_position(w4.get_position())
            w3.set_orientation(w4.get_orientation())

    def _is_last(self, waypoint):
        waypoint.skip = True

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        pass

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        pass
