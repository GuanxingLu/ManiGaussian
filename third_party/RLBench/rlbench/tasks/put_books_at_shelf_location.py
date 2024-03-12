from typing import List, Tuple

from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped


class PutBooksAtShelfLocation(Task):

    def init_task(self) -> None:
        # self._success_sensor_top = ProximitySensor('success_top')
        # self._success_sensor_bottom = ProximitySensor('success_bottom')
        self._books = [Shape('book2'), Shape('book1'), Shape('book0')]
        self._waypoints_idxs = [5, 11, -1]
        self.register_graspable_objects(self._books)

        self._book2_waypoints = [Dummy('waypoint%d' % i) for i in [3, 4, 5]]
        self._book1_waypoints = [Dummy('waypoint%d' % i) for i in [9, 10, 11]]
        self._book0_waypoints = [Dummy('waypoint%d' % i) for i in [15, 16]]

        self._variations = {
            0: {
                'seq': ['top_left', 'top_middle', 'top_right'],
                'success': ['success_top', 'success_top', 'success_top'],
                'lang': "put all books on the top shelf"
            },
            1: {
                'seq': ['top_left', 'bottom_middle', 'top_right'],
                'success': ['success_top', 'success_bottom', 'success_top'],
                'lang': "put one book at the bottom and two books on the top"
            },
            2: {
                'seq': ['top_left', 'bottom_middle', 'bottom_right'],
                'success': ['success_top', 'success_bottom', 'success_bottom'],
                'lang': "put one book at the top and two books at the bottom"
            },
            3: {
                'seq': ['bottom_left', 'bottom_middle', 'bottom_right'],
                'success': ['success_bottom', 'success_bottom', 'success_bottom'],
                'lang': "put all books on the bottom shelf"
            },
        }

    def init_episode(self, index: int) -> List[str]:
        self._variation_index = index
        setup = self._variations[self._variation_index]

        # success condition
        success_conditions = []
        for book_idx, book in enumerate(self._books):
            success_conditions.append(
            DetectedCondition(
                book, ProximitySensor(setup['success'][book_idx])
                )
            )

        # waypoints
        for idx, w in enumerate(self._book2_waypoints):
            w.set_pose(Dummy("%s_%d" % (setup['seq'][0], idx+1)).get_pose())
        for idx, w in enumerate(self._book1_waypoints):
            w.set_pose(Dummy("%s_%d" % (setup['seq'][1], idx+1)).get_pose())
        for idx, w in enumerate(self._book0_waypoints):
            w.set_pose(Dummy("%s_%d" % (setup['seq'][2], idx+1)).get_pose())


        # language instruction
        return [setup['lang']]

        # self.register_success_conditions([
        #     DetectedCondition(
        #         b, self._success_sensor) for b in self._books[:index+1]
        # ])
        # self.register_stop_at_waypoint(self._waypoints_idxs[index])
        # return ['put %d books on bookshelf' % (index + 1),
        #         'pick up %d books and place them on the top shelf' % (index + 1),
        #         'stack %d books up on the top shelf' % (index + 1)]

    def variation_count(self) -> int:
        return 4

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -3.14/2], [0.0, 0.0, 3.14/2]
