"""The driver controller's pad guard (rc_controller.PadGuard, TODO AR2 / AQ4): a trigger that was never seen released, or a pad that
dropped off, must not drive the car."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


class PadGuardTest(unittest.TestCase):
    def setUp(self):
        import rc_controller as rc
        self.rc, self.g = rc, rc.PadGuard()

    def test_a_trigger_that_never_moved_reads_released(self):
        # SDL reports 0.0 (half pressed once scaled) until the trigger is first moved: this drove the car off on 11 Nov
        self.assertEqual(self.g.trigger(self.rc.AXIS_LEFT_TRIGGER, 0.0), 0.0)
        self.assertEqual(self.g.trigger(self.rc.AXIS_LEFT_TRIGGER, 0.6), 0.0)

    def test_armed_after_it_was_seen_at_rest(self):
        ax = self.rc.AXIS_RIGHT_TRIGGER
        self.assertEqual(self.g.trigger(ax, -1.0), 0.0)                      # at rest: armed, still zero
        self.assertAlmostEqual(self.g.trigger(ax, 0.0), 0.5)
        self.assertAlmostEqual(self.g.trigger(ax, 1.0), 1.0)
        self.assertEqual(self.g.trigger(self.rc.AXIS_LEFT_TRIGGER, 0.0), 0.0)   # each trigger on its own

    def test_a_lost_pad_commands_nothing_and_needs_arming_again(self):
        ax = self.rc.AXIS_RIGHT_TRIGGER
        self.g.trigger(ax, -1.0)
        self.assertGreater(self.g.trigger(ax, 0.8), 0.0)
        self.g.lost()
        self.assertEqual(self.g.trigger(ax, 0.8), 0.0)
        self.assertEqual(self.g.stick(0.7), 0.0)
        self.g.found()
        self.assertEqual(self.g.trigger(ax, 0.0), 0.0)                       # back, but not seen at rest yet
        self.assertEqual(self.g.stick(0.7), 0.7)
        self.g.trigger(ax, -1.0)
        self.assertGreater(self.g.trigger(ax, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
