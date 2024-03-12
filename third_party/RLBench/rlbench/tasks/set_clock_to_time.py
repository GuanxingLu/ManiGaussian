from typing import List, Tuple
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.task import Task
from pyrep.objects.dummy import Dummy


TIMES = ['15', '30', '45']

class SetClockToTime(Task):

    def init_task(self) -> None:
        self._turn_points = [
            Dummy('point_15'),
            Dummy('point_30'),
            Dummy('point_45'),
        ]

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        time = TIMES[index]

        self.register_fail_conditions([
            DetectedCondition(Shape('clock_needle_crank'),
                              ProximitySensor('Panda_gripper_attachProxSensor'))
        ])

        self.register_success_conditions([
            DetectedCondition(Shape('clock_minute_head'),
                              ProximitySensor('detector_%s' % time)),
            NothingGrasped(self.robot.gripper)
        ])


        turn_point = self._turn_points[self._variation_index]
        waypoint3 = Dummy('waypoint2')
        waypoint3.set_pose(turn_point.get_pose())

        return [
            'change the clock to show time 12.%s' % (time),
            'adjust the time to 12.%s' % (time),
            'change the clock to 12.%s' % (time),
            'set the clock to 12.%s' % (time),
            'turn the knob on the back of the clock until the time shows 12.%s' % (time),
            'rotate the wheel on the clock to make it show 12.%s' % (time),
            'make the clock say 12.%s' % (time),
            'turn the knob on the back of the clock %d degrees' % (90*(index+1))
        ]

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -3.14/4 - 3.14/2], [0.0, 0.0, 3.14/4 - 3.14/2]