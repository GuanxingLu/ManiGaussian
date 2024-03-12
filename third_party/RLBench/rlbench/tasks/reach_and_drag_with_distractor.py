from typing import List
from rlbench.backend.task import Task
from rlbench.const import colors
from rlbench.backend.conditions import NothingGrasped, DetectedCondition
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor


class ReachAndDragWithDistractor(Task):

    def init_task(self) -> None:
        self.stick = Shape('stick0')
        self.register_graspable_objects([self.stick])
        self.cube = Shape('cube0')
        self.target = Shape('target0')
        self.distractor1 = Shape('distractor1_1')
        self.distractor2 = Shape('distractor2_2')
        self.distractor3 = Shape('distractor3_3')
        self.button = Shape('push_button_target')
        self.button2 = Shape('push_button_target_2')

    def init_episode(self, index: int) -> List[str]:
        self.register_success_conditions([
            DetectedCondition(self.cube, ProximitySensor('success0'))])
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)

        distractor1_rgb_name, distractor1_rgb = colors[(index + 5) % len(colors)]
        self.distractor1.set_color(distractor1_rgb)

        _, distractor2_rgb = colors[(index + 6) % len(colors)]
        self.distractor2.set_color(distractor2_rgb)

        _, distractor3_rgb = colors[(index + 7) % len(colors)]
        self.distractor3.set_color(distractor3_rgb)

        self.button.set_color(color_rgb)
        self.button2.set_color(distractor1_rgb)
            
        # if  index % 2 == 0:
        return ['use the stick to drag the cube onto the %s target'
                % color_name,
                'pick up the stick and use it to push or pull the cube '
                'onto the %s target' % color_name,
                'drag the block towards the %s square on the table top'
                % color_name,
                'grasping the stick by one end, pick it up and use the its '
                'other end to move the block onto the %s target' % color_name]
        # else:
            
        #     return ['push the %s button and then push the %s button' % (color_name, distractor1_rgb_name),
        #             'push the %s button and then push the %s button' % (color_name, distractor1_rgb_name),
        #             'push the %s button and then push the %s button' % (color_name, distractor1_rgb_name),]
    def variation_count(self) -> int:
        return len(colors)
