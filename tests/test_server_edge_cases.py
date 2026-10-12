"""Additional unit tests for server watcher and lifecycle edge cases."""

# These tests intentionally invoke private watcher helpers to exercise their
# failure and lifecycle behavior without starting real threads or processes.
# pylint: disable=protected-access,line-too-long

import argparse
import os
import tempfile
import unittest
from unittest.mock import (
    MagicMock,
    patch,
)

from pumaguard import (
    server,
)
from pumaguard.presets import (
    Settings,
)


class TestFolderObserverEdgeCases(unittest.TestCase):
    """Exercise observer lifecycle and watcher failure paths."""

    def setUp(self):
        self.presets = Settings()
        self.webui = MagicMock()
        self.webui.plugs = {}
        self.observer = server.FolderObserver(
            "watched", "inotify", self.presets, self.webui
        )

    def test_start_does_not_duplicate_live_monitor(self):
        """A running monitor must not be replaced by a second monitor."""
        self.observer._monitor_thread = MagicMock()
        self.observer._monitor_thread.is_alive.return_value = True

        with patch("pumaguard.server.threading.Thread") as mock_thread:
            self.observer.start()

        mock_thread.assert_not_called()

    def test_stop_joins_live_threads(self):
        """Stopping waits briefly for both active worker threads."""
        self.observer._thread = MagicMock()
        self.observer._thread.is_alive.return_value = True
        self.observer._monitor_thread = MagicMock()
        self.observer._monitor_thread.is_alive.return_value = True

        self.observer.stop()

        self.observer._thread.join.assert_called_once_with(timeout=2)
        self.observer._monitor_thread.join.assert_called_once_with(timeout=2)

    def test_monitor_observer_exits_when_stopped(self):
        """The monitor stops after its observer worker has completed."""
        self.observer._stop_event = MagicMock()
        self.observer._stop_event.is_set.side_effect = [False, True]

        with patch("pumaguard.server.threading.Thread") as mock_thread:
            self.observer._monitor_observer()

        mock_thread.return_value.start.assert_called_once()
        mock_thread.return_value.join.assert_called_once()

    def test_monitor_observer_restarts_after_thread_error(self):
        """A thread startup failure is logged and retried."""
        self.observer._stop_event = MagicMock()
        self.observer._stop_event.is_set.side_effect = [False, False, True]
        with (
            patch("pumaguard.server.threading.Thread") as mock_thread,
            patch("pumaguard.server.time.sleep") as mock_sleep,
        ):
            mock_thread.return_value.start.side_effect = RuntimeError("broken")
            self.observer._monitor_observer()

        mock_sleep.assert_called_once_with(1)

    def test_wait_for_file_stability_rejects_invalid_timeout(self):
        """Non-positive stability timeouts are invalid."""
        with self.assertRaises(ValueError):
            self.observer._wait_for_file_stability("missing.jpg", timeout=0)

    def test_wait_for_file_stability_handles_missing_file(self):
        """A file that disappears while waiting is rejected immediately."""
        with patch(
            "pumaguard.server.Image.open", side_effect=FileNotFoundError()
        ) as mock_open:
            self.assertFalse(
                self.observer._wait_for_file_stability(
                    "missing.jpg", timeout=1
                )
            )

        mock_open.assert_called_once_with("missing.jpg")

    def test_wait_for_file_stability_retries_missing_module(self):
        """A transient optional-decoder import failure is retried."""
        with (
            patch("pumaguard.server.Image.open") as mock_open,
            patch("pumaguard.server.FolderObserver._sleep") as mock_sleep,
            patch("pumaguard.server.FolderObserver._get_time") as mock_time,
        ):
            mock_time.side_effect = [0, 0.1, 0.2]
            mock_open.side_effect = [ModuleNotFoundError(), MagicMock()]
            result = self.observer._wait_for_file_stability(
                "image.jpg", timeout=1
            )

        self.assertTrue(result)
        mock_sleep.assert_called_once_with(0.5)

    def test_configure_subparser_accepts_server_options(self):
        """The server command exposes its documented CLI options."""
        parser = argparse.ArgumentParser()
        server.configure_subparser(parser)

        options = parser.parse_args(
            [
                "first",
                "second",
                "--sound-path",
                "/sounds",
                "--no-play-sound",
                "--no-download-progress",
                "--watch-method",
                "inotify",
            ]
        )

        self.assertEqual(options.FOLDER, ["first", "second"])
        self.assertEqual(options.sound_path, "/sounds")
        self.assertTrue(options.no_play_sound)
        self.assertTrue(options.no_download_progress)
        self.assertEqual(options.watch_method, "inotify")

    def test_inotify_observer_requires_stdout(self):
        """An inotify process without stdout cannot be consumed."""
        with (
            patch("pumaguard.server.acquire_lock") as mock_lock,
            patch("pumaguard.server.cache_model_two_stage") as mock_cache,
            patch("pumaguard.server.subprocess.Popen") as mock_popen,
        ):
            mock_popen.return_value.__enter__.return_value.stdout = None
            with self.assertRaises(ValueError):
                self.observer._observe()

        mock_cache.assert_called_once()
        mock_lock.return_value.release.assert_called_once()

    def test_inotify_observer_terminates_when_stopped(self):
        """A stop request terminates inotify before handling files."""
        with (
            patch("pumaguard.server.acquire_lock") as mock_lock,
            patch("pumaguard.server.cache_model_two_stage") as mock_cache,
            patch("pumaguard.server.subprocess.Popen") as mock_popen,
        ):
            process = mock_popen.return_value.__enter__.return_value
            process.stdout = iter(["watched/image.jpg\n"])
            self.observer._stop_event.set()
            self.observer._observe()

        process.terminate.assert_called_once()
        mock_cache.assert_called_once()
        mock_lock.return_value.release.assert_called_once()

    def test_inotify_observer_ignores_unstable_file(self):
        """Inotify does not dispatch files that remain unreadable."""
        with (
            patch("pumaguard.server.acquire_lock") as mock_lock,
            patch("pumaguard.server.cache_model_two_stage") as mock_cache,
            patch("pumaguard.server.subprocess.Popen") as mock_popen,
            patch.object(
                self.observer, "_wait_for_file_stability", return_value=False
            ),
        ):
            process = mock_popen.return_value.__enter__.return_value
            process.stdout = iter(["watched/image.jpg\n"])
            self.observer._observe()

        mock_cache.assert_called_once()
        mock_lock.return_value.release.assert_called_once()

    def test_os_observer_handles_stable_new_file(self):
        """The polling watcher dispatches stable files and honors extra wait."""
        self.observer.method = "os"
        self.presets.file_stabilization_extra_wait = 0.25
        with (
            patch("pumaguard.server.cache_model_two_stage"),
            patch("pumaguard.server.os.listdir", side_effect=[[], ["image.jpg"]]),
            patch("pumaguard.server.threading.Thread") as mock_thread,
            patch("pumaguard.server.time.sleep") as mock_sleep,
            patch.object(
                self.observer, "_wait_for_file_stability", return_value=True
            ),
        ):
            mock_sleep.side_effect = lambda _: self.observer._stop_event.set()
            self.observer._observe()

        mock_thread.assert_called_once_with(
            target=self.observer._handle_new_file,
            args=("watched/image.jpg",),
        )
        mock_thread.return_value.start.assert_called_once()
        mock_sleep.assert_any_call(0.25)

    def test_os_observer_ignores_unstable_file(self):
        """The polling watcher skips files that fail stability checks."""
        self.observer.method = "os"
        with (
            patch("pumaguard.server.cache_model_two_stage"),
            patch("pumaguard.server.os.listdir", side_effect=[[], ["image.jpg"]]),
            patch("pumaguard.server.time.sleep") as mock_sleep,
            patch.object(
                self.observer, "_wait_for_file_stability", return_value=False
            ),
        ):
            mock_sleep.side_effect = lambda _: self.observer._stop_event.set()
            self.observer._observe()

    def test_observe_rejects_unknown_method(self):
        """Unsupported watcher implementations fail clearly after model setup."""
        self.observer.method = "unknown"
        with (
            patch("pumaguard.server.acquire_lock") as mock_lock,
            patch("pumaguard.server.cache_model_two_stage") as mock_cache,
        ):
            with self.assertRaises(ValueError):
                self.observer._observe()

        mock_cache.assert_called_once()
        mock_lock.return_value.release.assert_called_once()

    def test_control_plug_switch_uses_default_hostname(self):
        """The shared Shelly helper receives plug data unchanged."""
        with patch("pumaguard.server.set_shelly_switch") as mock_switch:
            self.observer._control_plug_switch({}, True)

        mock_switch.assert_called_once_with(None, True, "unknown")

    def test_late_puma_detection_skips_deterrents(self):
        """Delayed classifications retain images but never activate deterrents."""
        lock = MagicMock()
        lock.time_waited.return_value = 61
        with (
            patch("pumaguard.server.acquire_lock", return_value=lock),
            patch(
                "pumaguard.server.classify_image_two_stage", return_value=0.9
            ) as mock_classify,
            patch("pumaguard.server.playsound") as mock_play,
            patch("pumaguard.server.set_shelly_switch") as mock_switch,
        ):
            self.observer._handle_new_file("late.jpg")

        mock_classify.assert_called_once()
        mock_play.assert_not_called()
        mock_switch.assert_not_called()
        lock.release.assert_called_once()

    def test_handle_new_file_logs_visualization_move_failure(self):
        """A visualization move failure does not prevent lock cleanup."""
        lock = MagicMock()
        lock.time_waited.return_value = 0
        with (
            patch("pumaguard.server.acquire_lock", return_value=lock),
            patch(
                "pumaguard.server.classify_image_two_stage", return_value=0.1
            ) as mock_classify,
            patch("pumaguard.server.shutil.move", side_effect=[None, OSError()]) as mock_move,
            patch("pumaguard.server.Path.exists", return_value=True),
        ):
            self.observer._handle_new_file("image.jpg")

        mock_classify.assert_called_once()
        self.assertEqual(mock_move.call_count, 2)
        lock.release.assert_called_once()

    def test_handle_new_file_releases_lock_after_classification_failure(self):
        """Unexpected classification failures are contained and release locks."""
        lock = MagicMock()
        lock.time_waited.return_value = 0
        with (
            patch("pumaguard.server.acquire_lock", return_value=lock),
            patch(
                "pumaguard.server.classify_image_two_stage",
                side_effect=RuntimeError("model failed"),
            ) as mock_classify,
        ):
            self.observer._handle_new_file("image.jpg")

        mock_classify.assert_called_once()
        lock.release.assert_called_once()


class TestServerMain(unittest.TestCase):
    """Unit tests for server startup and orderly shutdown orchestration."""

    def _presets(self, base: str) -> Settings:
        presets = Settings()
        presets.default_watch_dir = os.path.join(base, "watch")
        presets.classified_puma_dir = os.path.join(base, "puma")
        presets.classified_other_dir = os.path.join(base, "other")
        presets.intermediate_dir = os.path.join(base, "intermediate")
        presets.intermediate_puma_dir = os.path.join(base, "intermediate-puma")
        presets.intermediate_other_dir = os.path.join(base, "intermediate-other")
        return presets

    def _options(self, folders=None, sound_path=None):
        return argparse.Namespace(
            FOLDER=[] if folders is None else folders,
            sound_path=sound_path,
            no_play_sound=True,
            no_download_progress=True,
            watch_method="os",
        )

    def test_main_configures_and_stops_server(self):
        """Startup wires components and KeyboardInterrupt shuts them down."""
        with tempfile.TemporaryDirectory() as base:
            presets = self._presets(base)
            watch = os.path.join(base, "custom-watch")
            with (
                patch("pumaguard.server.threading.stack_size") as mock_stack,
                patch("pumaguard.server.WebUI") as mock_webui_class,
                patch("pumaguard.server.FolderManager") as mock_manager_class,
                patch("pumaguard.server.acquire_lock") as mock_lock,
                patch("pumaguard.server.cache_model_two_stage") as mock_cache,
                patch("pumaguard.server.signal.signal") as mock_signal,
                patch(
                    "pumaguard.server.time.sleep", side_effect=KeyboardInterrupt
                ) as mock_sleep,
            ):
                server.main(self._options([watch], "/sounds"), presets)

        webui = mock_webui_class.return_value
        manager = mock_manager_class.return_value
        self.assertEqual(presets.sound_path, "/sounds")
        self.assertFalse(presets.play_sound)
        self.assertFalse(presets.print_download_progress)
        webui.start.assert_called_once()
        manager.register_folder.assert_called_once_with(watch, "os", start=False)
        webui.add_image_directory.assert_called_once_with(watch)
        self.assertEqual(webui.add_classification_directory.call_count, 4)
        mock_cache.assert_called_once_with(
            yolo_model_filename=presets.yolo_model_filename,
            classifier_model_filename=presets.classifier_model_filename,
            print_progress=False,
            warm=True,
        )
        mock_lock.return_value.release.assert_called_once()
        manager.start_all.assert_called_once()
        self.assertEqual(mock_signal.call_count, 2)
        manager.stop_all.assert_called_once()
        webui.stop.assert_called_once()
        mock_stack.assert_called_once_with(64 * 1024 * 1024)
        mock_sleep.assert_called_once_with(1)

    def test_main_uses_default_folder_after_stack_size_failure(self):
        """Defaults are used when options omit folders and sound path."""
        with tempfile.TemporaryDirectory() as base:
            presets = self._presets(base)
            options = self._options()
            with (
                patch.dict(os.environ, {"PUMAGUARD_SOUND_PATH": "/env-sounds"}),
                patch(
                    "pumaguard.server.threading.stack_size",
                    side_effect=RuntimeError("unsupported"),
                ) as mock_stack,
                patch("pumaguard.server.WebUI") as mock_webui_class,
                patch("pumaguard.server.FolderManager") as mock_manager_class,
                patch("pumaguard.server.acquire_lock"),
                patch("pumaguard.server.cache_model_two_stage"),
                patch("pumaguard.server.signal.signal"),
                patch(
                    "pumaguard.server.time.sleep", side_effect=KeyboardInterrupt
                ),
            ):
                server.main(options, presets)

        self.assertEqual(presets.sound_path, "/env-sounds")
        mock_manager_class.return_value.register_folder.assert_called_once_with(
            presets.default_watch_dir, "os", start=False
        )
        mock_webui_class.return_value.add_image_directory.assert_called_once_with(
            presets.default_watch_dir
        )
        mock_stack.assert_called_once()

    def test_main_termination_handler_stops_manager(self):
        """Registered signal handlers stop watchers before exiting."""
        with tempfile.TemporaryDirectory() as base:
            presets = self._presets(base)
            with (
                patch("pumaguard.server.WebUI"),
                patch("pumaguard.server.FolderManager") as mock_manager_class,
                patch("pumaguard.server.acquire_lock"),
                patch("pumaguard.server.cache_model_two_stage"),
                patch("pumaguard.server.sys.exit", side_effect=SystemExit(0)) as mock_exit,
                patch("pumaguard.server.signal.signal") as mock_signal,
            ):
                def register_signal(_, handler):
                    if mock_signal.call_count == 2:
                        handler(server.signal.SIGTERM, None)

                mock_signal.side_effect = register_signal
                with self.assertRaises(SystemExit) as raised:
                    server.main(self._options([presets.default_watch_dir]), presets)

        self.assertEqual(raised.exception.code, 0)
        mock_manager_class.return_value.stop_all.assert_called_once()
        mock_exit.assert_called_once_with(0)
