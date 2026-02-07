"""
Progress tracking mixin for MainWindow.
Handles tile progress, batch progress, smooth interpolation, and timing.
"""

import os
import time
from pathlib import Path

from ..utils import format_time_hms


class ProgressTrackerMixin:
    """Mixin providing progress tracking and smooth progress bar interpolation."""

    def _update_time_display(self):
        """Update elapsed time and ETA labels."""
        if self._batch_start_time <= 0:
            return

        elapsed = time.perf_counter() - self._batch_start_time
        elapsed_str = format_time_hms(elapsed)

        if self._current_avg_per_image > 0 and self._completed_files_in_batch > 0:
            remaining_files = self._total_files_in_batch - self._completed_files_in_batch
            eta_seconds = remaining_files * self._current_avg_per_image
            eta_str = format_time_hms(eta_seconds)
        else:
            eta_str = "--:--:--"

        self._time_label.setText(f"Elapsed: {elapsed_str} | ETA: {eta_str}")

    def _on_progress(self, current: int, total: int):
        """Update tile progress - records timing for smooth interpolation."""
        now = time.perf_counter()

        # Calculate time per tile unit for interpolation
        if current > self._last_tile_current and self._last_tile_current > 0:
            delta_tiles = current - self._last_tile_current
            delta_time = now - self._last_tile_update_time
            if delta_tiles > 0 and delta_time > 0:
                new_rate = delta_time / delta_tiles
                if self._tile_time_per_unit > 0:
                    self._tile_time_per_unit = 0.7 * self._tile_time_per_unit + 0.3 * new_rate
                else:
                    self._tile_time_per_unit = new_rate

        # Store current state for next update
        self._last_tile_current = current
        self._last_tile_total = total
        self._last_tile_update_time = now

        # Set base progress (will be interpolated by smooth timer)
        self._interpolated_tile_progress = current / total if total > 0 else 0.0

    def _update_smooth_progress(self):
        """Interpolate and update progress bars smoothly (called every 50ms)."""
        now = time.perf_counter()
        time_on_current = now - self._current_image_start_time

        # Interpolate tile progress using multiple strategies
        tile_progress = 0.0

        # Strategy 1: Use tile-based timing if we have real tile progress
        if self._tile_time_per_unit > 0 and self._last_tile_total > 0 and self._last_tile_current > 0:
            time_since_update = now - self._last_tile_update_time
            expected_progress = time_since_update / self._tile_time_per_unit

            base_progress = self._last_tile_current / self._last_tile_total
            tile_progress = base_progress + (expected_progress / self._last_tile_total)

            tile_progress = min(tile_progress, 1.0)
            tile_progress = max(tile_progress, base_progress)

        # Strategy 2: Use time-based extrapolation from previous image duration
        elif self._last_image_duration > 0:
            raw_progress = time_on_current / self._last_image_duration
            raw_progress = min(raw_progress, 0.98)
            tile_progress = 1.0 - (1.0 - raw_progress) ** 2
            tile_progress = max(0.0, min(tile_progress, 0.98))

        # Strategy 3: Use average time per image if no other data
        elif self._current_avg_per_image > 0:
            raw_progress = time_on_current / self._current_avg_per_image
            raw_progress = min(raw_progress, 0.98)
            tile_progress = 1.0 - (1.0 - raw_progress) ** 2
            tile_progress = max(0.0, min(tile_progress, 0.98))

        # Strategy 4: Fallback to discrete progress
        else:
            tile_progress = self._interpolated_tile_progress

        tile_value = int(tile_progress * 1000)
        self._progress_bar.setValue(tile_value)

        # Interpolate batch progress using time-based estimation
        if self._total_files_in_batch > 0:
            completed_fraction = self._completed_files_in_batch / self._total_files_in_batch
            current_file_fraction = tile_progress / self._total_files_in_batch

            time_interpolation = 0.0
            if self._current_avg_per_image > 0 and self._current_image_start_time > 0:
                time_on_current = now - self._current_image_start_time
                time_based_progress = min(time_on_current / self._current_avg_per_image, 1.0)

                if tile_progress > 0:
                    blended_progress = 0.7 * tile_progress + 0.3 * time_based_progress
                else:
                    blended_progress = time_based_progress

                current_file_fraction = blended_progress / self._total_files_in_batch

            batch_progress = completed_fraction + current_file_fraction
            batch_value = int(batch_progress * 1000)
            batch_value = min(batch_value, 1000)

            self._batch_progress_bar.setValue(batch_value)

            files_done = self._completed_files_in_batch
            files_total = self._total_files_in_batch
            pct = int(batch_progress * 100)
            self._batch_progress_bar.setFormat(f"Batch: {pct}% ({files_done}/{files_total} files)")

    def _on_file_progress(self, current: int, total: int, file_path: str):
        """Update file progress and thumbnail for current image."""
        # Reset tile progress state for new file
        self._last_tile_current = 0
        self._last_tile_total = 1
        self._last_tile_update_time = time.perf_counter()
        self._interpolated_tile_progress = 0.0
        self._current_image_start_time = time.perf_counter()

        # Update thumbnail to show current file being processed
        if file_path and file_path != self._thumbnail_label.get_current_path():
            self._update_thumbnail(file_path)

        # Update labels
        file_name = Path(file_path).name if file_path else ""
        self._progress_label.setText(f"Processing image {current}/{total}: {file_name}")
        self._current_file_label.setText(f"Processing {current}/{total}")

    def _on_file_done(self, input_path: str, output_path: str, elapsed_time: float):
        """Handle completed file."""
        self._completed_files_in_batch += 1
        self._last_output_path = output_path

        # Record this image's duration for smooth progress extrapolation
        image_duration = time.perf_counter() - self._current_image_start_time
        if image_duration > 0:
            if self._last_image_duration > 0:
                self._last_image_duration = 0.7 * self._last_image_duration + 0.3 * image_duration
            else:
                self._last_image_duration = image_duration

        # Update average using total elapsed time
        total_elapsed = time.perf_counter() - self._batch_start_time
        self._current_avg_per_image = total_elapsed / self._completed_files_in_batch

        # Reset tile progress for next file
        self._interpolated_tile_progress = 0.0
        self._last_tile_current = 0

        # Add to processing log
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({elapsed_time:.2f}s)"
        self._log_entries.append(log_entry)

        # Update display
        self._current_file_label.setText(
            f"Processed {self._completed_files_in_batch}/{self._total_files_in_batch}"
        )
        self._avg_label.setText(f"Avg per image: {self._current_avg_per_image:.2f}s")

        # Update time display immediately
        self._update_time_display()

        # Progress label
        self._progress_label.setText(f"Saved: {os.path.basename(output_path)}")

    def _on_file_skipped(self, input_path: str, reason: str):
        """Handle skipped file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] SKIPPED: {os.path.basename(input_path)} ({reason})"
        self._log_entries.append(log_entry)
        self._progress_label.setText(f"Skipped: {os.path.basename(input_path)} ({reason})")

    def _on_checkpoint_updated(self, current_index: int, remaining_files: list):
        """Handle checkpoint update - save state for resume."""
        self._checkpoint_index = current_index
        self._checkpoint_files = list(remaining_files)
        self._checkpoint_onnx = self.onnx_edit.text()
        self._checkpoint_input_root = self.input_root

        if self._checkpoint_files:
            self._resume_button.setEnabled(True)
