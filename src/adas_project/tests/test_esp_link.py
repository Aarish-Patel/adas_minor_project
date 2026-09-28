"""The relay's supervised ESP32 link (pi/esp_link.py): healthy, silent, reboot banner, USB drop-out and reopen."""
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pi.esp_link import EspLink


class FakeEsp:
    """A serial port stand-in: answers PING with PONG if `alive`, can emit a boot banner, can be 'unplugged'."""

    def __init__(self, alive=True):
        self.alive, self.unplugged = alive, False
        self.out, self.lines = b"", []
        self.lock = threading.Lock()

    def write(self, data):
        if self.unplugged:
            raise OSError(5, "Input/output error")
        with self.lock:
            for line in data.decode().splitlines():
                self.lines.append(line)
                if line == "PING" and self.alive:
                    self.out += b"PONG\r\n"
        return len(data)

    @property
    def in_waiting(self):
        if self.unplugged:
            raise OSError(5, "Input/output error")
        return len(self.out)

    def read(self, n=1):
        with self.lock:
            r, self.out = self.out[:n], self.out[n:]
        return r

    def boot(self):
        with self.lock:
            self.out += b"ets Jun  8 2016 00:22:57\r\nROVER READY\r\n"

    def close(self):
        pass


class EspLinkTest(unittest.TestCase):
    def setUp(self):
        self.saved = (EspLink.PING_EVERY, EspLink.SILENT_AFTER, EspLink.REOPEN_EVERY)
        EspLink.PING_EVERY, EspLink.SILENT_AFTER, EspLink.REOPEN_EVERY = 0.1, 0.5, 0.1
        self.events = []

    def tearDown(self):
        EspLink.PING_EVERY, EspLink.SILENT_AFTER, EspLink.REOPEN_EVERY = self.saved

    def link(self, fake, candidates=None):
        return EspLink(lambda port: fake, "FAKE", on_event=lambda msg, **kw: self.events.append(msg),
                       candidates=candidates or (lambda: []))

    def wait_for(self, cond, timeout=3.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if cond():
                return True
            time.sleep(0.02)
        return False

    def test_healthy_esp32_is_ok(self):
        fake = FakeEsp(alive=True)
        link = self.link(fake)
        self.assertTrue(self.wait_for(lambda: link.state == "ok"), link.status())
        self.assertIn("PING", fake.lines)
        link.close()

    def test_silent_esp32_is_reported(self):
        """The 28 Sep failure: port present, chip not answering -> 'silent', and an event for the log."""
        fake = FakeEsp(alive=False)
        link = self.link(fake)
        self.assertTrue(self.wait_for(lambda: link.state == "silent"), link.status())
        self.assertIn("ESP32 silent", self.events)
        self.assertFalse(link.healthy())
        link.close()

    def test_reboot_banner_is_counted(self):
        fake = FakeEsp(alive=True)
        link = self.link(fake)
        fake.boot()
        self.assertTrue(self.wait_for(lambda: link.reboots == 1), link.status())
        link.close()

    def test_unplug_drops_writes_then_reopens(self):
        """USB drop-out: writes stop raising into the relay, the link is 'lost', then reopens on the new device."""
        first, second = FakeEsp(alive=True), FakeEsp(alive=True)
        devices = {"FAKE": first, "NEW": second}
        link = EspLink(lambda port: devices[port], "FAKE", on_event=lambda msg, **kw: self.events.append(msg),
                       candidates=lambda: ["NEW"] if first.unplugged else [])
        self.assertTrue(self.wait_for(lambda: link.state == "ok"))
        first.unplugged = True
        self.assertEqual(link.write(b"M 100\n"), 0)            # no exception into the control loop
        self.assertTrue(self.wait_for(lambda: link.reconnects == 1 and link.state == "ok"), link.status())
        self.assertIn("ESP32 link lost", self.events)
        self.assertEqual(link.port, "NEW")
        link.write(b"M 0\n")
        self.assertIn("M 0", second.lines)
        link.close()

    def test_no_esp32_at_start(self):
        link = EspLink(lambda port: FakeEsp(), None, candidates=lambda: [])
        self.assertEqual(link.state, "lost")
        self.assertEqual(link.write(b"M 50\n"), 0)
        self.assertEqual(link.status()["dropped_writes"], 1)
        link.close()


if __name__ == "__main__":
    unittest.main()
